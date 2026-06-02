"""
source: https://github.com/yuval-alaluf/Attend-and-Excite/issues/44
"""

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.utils import deprecate, logging

# from composition import get_inpaint_mask_by_attn_maps
from utils import (  # get_mask_by_attn_maps,
    AttentionStore,
    AttnProcessor,
    compute_layout_guidance_loss,
    get_inpaint_mask_by_attn_maps,
    min_max_normalize,
)

logger = logging.get_logger(__name__)


class GaussianSmoothing(torch.nn.Module):
    """Apply Gaussian smoothing using a depthwise convolution kernel.

    This module builds a separable Gaussian kernel for the given number of
    dimensions and applies it to an input tensor via grouped convolution.

    Args:
        channels: number of channels in the input tensor (depthwise convolution).
        kernel_size: int or sequence of ints specifying the kernel spatial size
            along each dimension.
        sigma: standard deviation for the Gaussian kernel (float or sequence).
        dim: number of spatial dimensions (1, 2 or 3).
    """
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int | Sequence[int] = 3,
        sigma: float | Sequence[float] = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        kernel = 1.0
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying the Gaussian kernel to `x`.

        Args:
            x: input tensor of shape (N, C, ...spatial_dims...).

        Returns:
            The smoothed tensor with the same shape as `x`.
        """
        return self.conv(x, weight=self.weight.to(x.dtype), groups=self.groups)


class AttendAndExciteSDXL(StableDiffusionXLPipeline):
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        **kwargs,
    ):
        return super().encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

    def _register_attention_control(self, store: AttentionStore) -> None:
        """Install `AttnProcessor` instances on the U-Net to capture attention.

        Args:
            store: an `AttentionStore` instance where attention maps are accumulated.
        """
        procs, count = {}, 0
        for name in self.unet.attn_processors.keys():
            place = (
                "mid"
                if name.startswith("mid_block")
                else "up" if name.startswith("up_blocks") else "down"
            )
            procs[name] = AttnProcessor(store, place)
            count += 1
        self.unet.set_attn_processor(procs)
        store.num_att_layers = count

    def get_indices(self, prompt: str) -> Dict[str, int]:
        ids = self.tokenizer(prompt).input_ids
        return {
            tok: i for i, tok in enumerate(self.tokenizer.convert_ids_to_tokens(ids))
        }

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices_to_alter: List[int],
        normalize_eot: bool = False,
    ) -> List[torch.Tensor]:
        """Compute per-token maximum attention scores.

        Args:
            attention_maps: tensor of aggregated attention maps with shape
                (batch, spatial, tokens) or similar depending on the store.
            indices_to_alter: list of token indices (or lists of indices) to inspect.
            normalize_eot: whether to normalize indexing relative to the prompt EOT.

        Returns:
            A list of scalar tensors containing the maximum attention value for
            each requested token index.
        """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        # indices_to_alter = [index - 1 for index in indices_to_alter]
        shifted_indices = []
        for index in indices_to_alter:
            if isinstance(index, list):
                shifted_indices.extend([_index - 1 for _index in index])
            else:
                shifted_indices.append(index - 1)
        indices_to_alter = shifted_indices

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            smoothing = GaussianSmoothing().to(attention_maps.device)
            _input = F.pad(
                image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
            )
            image = smoothing(_input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())

        return max_indices_list

    def _compute_composition_score(
        self,
        keep_ratio: float,
        categories: List[str],
        tok_map: Dict[str, List[int]],
        add_eot: bool = True,
    ) -> Tuple[List[int], float, torch.Tensor]:
        """Compute a composition score and a candidate inpaint mask for categories.

        Args:
            keep_ratio: fraction of attention mass to keep when creating masks.
            categories: list of category strings to evaluate.
            tok_map: mapping from subtoken string to token index list.
            add_eot: whether to consider the end-of-text token as an extra category.

        Returns:
            A tuple of (return_token_ids, score, mask) where `return_token_ids`
            is a list of token indices to target, `score` is the composition score
            (float), and `mask` is a torch.Tensor inpaint mask.
        """
        categories_cp = categories.copy()
        EOT = "<|endoftext|>"
        if add_eot:
            categories_cp.append(EOT)

        attn_maps = self.attention_store.aggregate(where=["up", "mid"]).detach()
        token_ids = []
        normalized_maps = []
        for cat in categories_cp:
            if cat != EOT:
                cat += "</w>"
            if cat not in tok_map:
                subtoks = [
                    idx
                    for tok in self.tokenizer.tokenize(cat.strip("</w>"))
                    if tok in tok_map
                    for idx in tok_map[tok]
                ]
                _attn_map = attn_maps[
                    :,
                    :,
                    subtoks,
                ].mean(-1)
            else:
                subtoks = tok_map[cat]
                _attn_map = attn_maps[:, :, subtoks].mean(-1)
            token_ids.append(subtoks)
            normalized_maps.append(min_max_normalize(_attn_map))

        scores = []
        masks = []
        for i, attn_map in enumerate(normalized_maps):
            # Get attention map from other tokens
            other_tok_map = torch.stack(
                [normalized_maps[j] for j in range(len(normalized_maps)) if j != i],
                dim=-1,
            ).mean(dim=-1)
            mask = get_inpaint_mask_by_attn_maps(
                other_tok_map,
                keep_ratio=keep_ratio,
                target_H=32,
                target_W=32,
                smoothed=True,
                return_mask_only=True,
            )
            mask = -1 * mask + 1  # invert mask to get free space
            masks.append(mask)
            outside = attn_map * mask
            score = torch.sum(outside) / (torch.sum(attn_map) + 1e-5)
            scores.append(score)

        # sort indices by ascending score
        sorted_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]))
        return_tok_ids = None
        for idx in sorted_idx:
            if not isinstance(token_ids[idx], list):
                return_tok_ids = [token_ids[idx]]
            elif isinstance(token_ids[idx], list) and len(token_ids[idx]) > 0:
                return_tok_ids = token_ids[idx]
            if return_tok_ids is not None:
                break

        if return_tok_ids is None:
            raise ValueError(
                f"All token_ids are empty. categories={categories}, tok_map keys={tok_map}"
            )

        return (
            return_tok_ids,
            scores[idx],
            masks[idx],
        )

    def _compute_loss(
        self,
        indices_to_alter: List[int],
        normalize_eot: bool = False,
        return_losses: bool = False,
        return_max_attention_per_index: bool = False,
        add_joint_composition_loss: bool = False,
        keep_ratio: float = 0.5,
        categories: Optional[List[str]] = None,
        tok_map: Optional[Dict[str, List[int]]] = None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[float]],
        Tuple[torch.Tensor, List[float], List[torch.Tensor]],
        Tuple[torch.Tensor, List[torch.Tensor]],
    ]:

        """Compute the Attend-and-Excite loss for the requested tokens.

        Args:
            indices_to_alter: token indices to compute the loss for.
            normalize_eot: whether to normalize indices with respect to EOT.
            return_losses: if True, also return per-token loss values.
            return_max_attention_per_index: if True, return max attention values.
            add_joint_composition_loss: if True, include layout-guidance loss.
            keep_ratio: parameter forwarded to composition scoring mask computation.
            categories: optional categories list for joint composition loss.
            tok_map: optional token map for joint composition loss.

        Returns:
            Either a single scalar loss tensor or a tuple containing the loss and
            optional auxiliary information depending on the flags.
        """

        attn_maps = self.attention_store.aggregate(("up", "down", "mid"))
        max_attention_per_index = self._compute_max_attention_per_index(
            attn_maps, indices_to_alter, normalize_eot
        )

        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)

        if add_joint_composition_loss:
            target_toks, composition_score, inpaint_mask = (
                self._compute_composition_score(keep_ratio, categories, tok_map)
            )
            if not isinstance(target_toks, list):
                target_toks = [target_toks]
            bboxes = len(target_toks) * [None]
            lg_loss = compute_layout_guidance_loss(
                self, target_toks, bboxes, inpaint_mask
            )
            loss = (loss + lg_loss) / 2

        if return_losses and return_max_attention_per_index:
            return loss, losses, max_attention_per_index
        elif not return_losses and return_max_attention_per_index:
            return loss, max_attention_per_index
        elif return_losses and not return_max_attention_per_index:
            return loss, losses
        else:
            return loss

    def _unet_forward_hook_attnmaps(
        self,
        latents: torch.Tensor,
        t: int,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> None:
        """
        Peform a forward pass through the unet to collect attention maps.
        """
        latent_input = self.scheduler.scale_model_input(latents, t)
        self.unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": add_text_embeds,
                "time_ids": time_ids,
            },
        )
        self.unet.zero_grad()

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[int],
        loss: torch.Tensor,
        threshold: float,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
        step_size: float,
        t: int,
        max_refinement_steps: int = 20,
        normalize_eot: bool = False,
        text_inputs: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_loss = max(0, 1.0 - threshold)

        while loss > target_loss:
            iteration += 1
            latents = latents.clone().detach().requires_grad_(True)
            self._unet_forward_hook_attnmaps(
                latents,
                t,
                prompt_embeds,
                add_text_embeds,
                time_ids,
            )
            loss, losses, max_attention_per_index = self._compute_loss(
                indices_to_alter,
                normalize_eot,
                return_losses=True,
                return_max_attention_per_index=True,
            )

            if loss != 0:
                grad_cond = torch.autograd.grad(loss, [latents])[0]
                latents = latents - step_size * grad_cond
            else:
                break

            if iteration >= max_refinement_steps:
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        self._unet_forward_hook_attnmaps(
            latents,
            t,
            prompt_embeds,
            add_text_embeds,
            time_ids,
        )
        loss, losses = self._compute_loss(
            indices_to_alter,
            normalize_eot,
            return_losses=True,
        )

        return loss, latents

    def _prepare_for_attend_and_excite(
        self,
        width: int,
        height: int,
        attn_res: Optional[Tuple[int, int]],
        scale_factor: float,
        batch_size: int,
        num_images_per_prompt: int,
        timesteps: Sequence[int],
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        do_classifier_free_guidance: bool,
        prompt: Union[str, List[str]],
        token_indices: Union[List[int], List[List[int]]],
        optimized_layers: Optional[List[int]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        Union[List[int], List[List[int]]],
        Dict[str, torch.Tensor],
    ]:
        """Prepare embeddings, step-sizes and token batches for refinement.

        Returns a tuple of (prompt_embeds_refine, add_txt_refine, time_ids_refine,
        step_sizes, token_indices, text_inputs) used by the refinement loop.
        """
        # 1: hook attention
        if attn_res is None:
            attn_res = (int(np.ceil(width / 32)), int(np.ceil(height / 32)))
        self.attention_store = AttentionStore(attn_res, optimized_layers)
        self.original_procs = self.unet.attn_processors
        self._register_attention_control(self.attention_store)

        # 2: step size schedule
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(
            scale_range
        )  # https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/pipeline_attend_and_excite.py#L500C1-L500C97

        # 3: Split embeds (ignore uncoditional part) for refinement batch
        prompt_embeds_refine = (
            prompt_embeds[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else prompt_embeds
        )
        add_txt_refine = (
            add_text_embeds[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else add_text_embeds
        )
        time_ids_refine = (
            add_time_ids[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else add_time_ids
        )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        if isinstance(token_indices[0], int):
            token_indices = [token_indices]
        token_ids_batched: List[List[int]] = []
        for ind in token_indices:
            token_ids_batched += [ind] * num_images_per_prompt

        return (
            prompt_embeds_refine,
            add_txt_refine,
            time_ids_refine,
            step_sizes,
            token_indices,
            text_inputs,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        token_indices: Optional[Union[List[int], List[List[int]]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image=None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_iter_to_alter: int = 25,
        thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 30,
        attn_res: Optional[Tuple[int, int]] = None,
        optimized_layers: Optional[List[int]] = list(range(70, 82)),
        add_joint_composition_loss: bool = False,
        keep_ratio: float = 0.5,
        categories: Optional[List[str]] = None,
        tok_map: Optional[Dict[str, List[int]]] = None,
        progress_to_save_attn: Optional[float] = None,
        **kwargs,
    ):
        """Generate images while applying Attend-and-Excite refinements.

        Args: (only key arguments listed)
            prompt: text prompt or list of prompts.
            token_indices: token index or list of token indices to target.
            num_inference_steps: number of denoising steps.
            guidance_scale: classifier-free guidance scale.
            progress_to_save_attn: optional fraction of progress to save attention maps.

        Returns:
            A `StableDiffusionXLPipelineOutput` containing generated images.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        self.default_sample_size = (
            self.unet.config.sample_size
            if hasattr(self, "unet")
            and self.unet is not None
            and hasattr(self.unet.config, "sample_size")
            else 128
        )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = (
            guidance_scale > 1.0 and self.unet.config.time_cond_proj_dim is None
        )

        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7 Prepare added time ids & embeddings for sdxl
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # 8.1 Apply denoising_end
        if (
            denoising_end is not None
            and isinstance(denoising_end, float)
            and denoising_end > 0
            and denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # Prepare for attend and excite
        (
            prompt_embeds_refine,
            add_txt_refine,
            time_ids_refine,
            step_sizes,
            token_indices,
            text_inputs,
        ) = self._prepare_for_attend_and_excite(
            width,
            height,
            attn_res,
            scale_factor,
            batch_size,
            num_images_per_prompt,
            timesteps,
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            do_classifier_free_guidance,
            prompt,
            token_indices,
            optimized_layers=optimized_layers,
        )

        # Start denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)
                    new_latents = []
                    for (
                        _latent,
                        _prompt_emb,
                        _add_text_emb,
                        _time_ids,
                        _token_ids,
                    ) in zip(
                        latents,
                        prompt_embeds_refine,
                        add_txt_refine,
                        time_ids_refine,
                        token_indices,
                    ):

                        # Add dimensions for batch
                        _latent = _latent.unsqueeze(0)
                        _prompt_emb = _prompt_emb.unsqueeze(0)
                        _add_text_emb = _add_text_emb.unsqueeze(0)
                        _time_ids = _time_ids.unsqueeze(0)

                        _latent = _latent.clone().detach().requires_grad_(True)
                        self._unet_forward_hook_attnmaps(
                            _latent,
                            t,
                            _prompt_emb,
                            _add_text_emb,
                            _time_ids,
                        )
                        _loss = self._compute_loss(
                            _token_ids,
                            add_joint_composition_loss=add_joint_composition_loss,
                            keep_ratio=keep_ratio,
                            categories=categories,
                            tok_map=tok_map,
                        )

                        # refine if under threshold
                        if i in thresholds and _loss > 1.0 - thresholds[i]:
                            _loss, _latent = self._perform_iterative_refinement_step(
                                _latent,
                                _token_ids,
                                _loss,
                                thresholds[i],
                                _prompt_emb,
                                _add_text_emb,
                                _time_ids,
                                step_sizes[i],
                                t,
                                text_inputs=text_inputs,
                            )
                        # gradient update
                        if i < max_iter_to_alter and _loss != 0:
                            grad_cond = torch.autograd.grad(
                                _loss, [_latent], retain_graph=False
                            )[0]
                            _latent = _latent - step_sizes[i] * grad_cond
                        new_latents.append(_latent)

                    latents = torch.cat(new_latents)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if latent_model_input.dtype != self.unet.dtype:
                    latent_model_input = latent_model_input.to(self.unet.dtype)

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://huggingface.co/papers/2305.08891
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if (
                    progress_to_save_attn is not None
                    and i > int(len(timesteps) * progress_to_save_attn)
                    and getattr(self, "saved_attn_maps", None) is None
                ):
                    self.saved_attn_maps = (
                        self.attention_store.aggregate(["up", "mid", "down"])
                        .detach()
                        .cpu()
                    )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    add_text_embeds = callback_outputs.pop(
                        "add_text_embeds", add_text_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            )

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype
                )
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = (
                hasattr(self.vae.config, "latents_mean")
                and self.vae.config.latents_mean is not None
            )
            has_latents_std = (
                hasattr(self.vae.config, "latents_std")
                and self.vae.config.latents_std is not None
            )
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, 4, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = (
                    latents * latents_std / self.vae.config.scaling_factor
                    + latents_mean
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()
        self.unet.set_attn_processor(self.original_procs)  # restore

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def efficient_mode(self):
        self.enable_attention_slicing()

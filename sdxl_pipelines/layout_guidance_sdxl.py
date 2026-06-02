"""
source: https://github.com/nipunjindal/diffusers-layout-guidance
"""

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from utils import AttentionStore, AttnProcessor

logger = logging.get_logger(__name__)


class LayoutGuidanceSDXL(StableDiffusionXLPipeline):
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

    def check_inputs(
        self,
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
        bboxes,
    ):
        # Check inputs for sdxl base
        super().check_inputs(
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

        if bboxes is not None:
            if isinstance(bboxes, list):
                if isinstance(bboxes[0], list):
                    if (
                        isinstance(bboxes[0][0], list)
                        and len(bboxes[0][0]) == 4
                        and all(isinstance(x, float) for x in bboxes[0][0])
                    ):
                        bboxes_batch_size = len(bboxes)
                    elif (
                        isinstance(bboxes[0], list)
                        and len(bboxes[0]) == 4
                        and all(isinstance(x, float) for x in bboxes[0])
                    ):
                        bboxes_batch_size = 1
                    else:
                        print(isinstance(bboxes[0], list), len(bboxes[0]))
                        raise TypeError(
                            "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                        )
                else:
                    print(isinstance(bboxes[0], list), len(bboxes[0]))
                    raise TypeError(
                        "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                    )
            else:
                print(isinstance(bboxes[0], list), len(bboxes[0]))
                raise TypeError(
                    "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if bboxes_batch_size != prompt_batch_size:
            raise ValueError(
                f"bbox batch size must be same as prompt batch size. bbox batch size: {bboxes_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def get_add_time_ids(
        self,
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
        dtype: torch.dtype,
        text_embed_dim: int,
    ) -> torch.Tensor:
        return self._get_add_time_ids(
            orig_size,
            (0, 0),  # crop top‑left
            target_size,
            dtype,
            text_embed_dim,
        )

    def _register_attention_control(self, store: AttentionStore) -> None:
        """Replace the UNet's attention processors with AttnProcessor wrappers.

        This hooks an :class:`AttentionStore` into the model so attention maps
        can be recorded during a forward pass. The original processors are
        replaced with ``AttnProcessor(store, place)`` instances which write
        into the provided ``store``.

        Args:
            store: An :class:`AttentionStore` instance used to collect
                attention maps during the forward pass.
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

    def _prepare_for_layout_guidance(
        self,
        width: int,
        height: int,
        attn_res: Optional[Tuple[int, int]],
        batch_size: int,
        num_images_per_prompt: int,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        do_classifier_free_guidance: bool,
        token_indices: Union[List[int], List[List[int]]],
        bboxes: Union[List[List[List[float]]], List[List[float]]],
        optimized_layers: Optional[List[int]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[List[List[float]]],
        List[List[int]],
    ]:
        """Prepare tensors and batched metadata for layout-guided refinement.

        This method hooks the attention store into the UNet, splits out the
        conditional parts of the prompt embeddings for the refinement batch,
        and expands/duplicates the provided ``token_indices`` and ``bboxes``
        to match ``num_images_per_prompt``.

        Returns a tuple of:
          - ``prompt_embeds_refine``: embeddings used for refinement passes.
          - ``add_txt_refine``: additional text embeddings used for added cond.
          - ``time_ids_refine``: time id tensors for refinement.
          - ``bboxes_batched``: list of per-image bounding boxes.
          - ``token_ids_batched``: list of per-image token index lists.

        Args:
            width: Target image width in pixels.
            height: Target image height in pixels.
            attn_res: Optional attention resolution override (W_cells, H_cells).
            batch_size: Number of prompt batches.
            num_images_per_prompt: Number of images generated per prompt.
            prompt_embeds: Encoded prompt embeddings for the full batch.
            add_text_embeds: Pooled prompt embeddings used for added conditioning.
            add_time_ids: Time id tensors corresponding to the added condition.
            do_classifier_free_guidance: Whether classifier-free guidance is used
                (controls splitting of unconditional/conditional parts).
            token_indices: Token index or list of token index lists indicating
                which attention head positions correspond to objects.
            bboxes: Bounding boxes specified either as a single list for the
                batch or as a list-of-lists for multiple batches.
            optimized_layers: Optional list of attention layers to optimize.
        """
        # 1: hook attention
        if attn_res is None:
            attn_res = (int(np.ceil(width / 32)), int(np.ceil(height / 32)))
        self.attention_store = AttentionStore(attn_res, optimized_layers)
        self.original_procs = self.unet.attn_processors
        self._register_attention_control(self.attention_store)

        # 2: Split embeds (ignore uncoditional part) for refinement batch
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

        # 3: Make token indices & bboxes batched for num_images_per_prompt
        if isinstance(token_indices[0], int):
            token_indices = [token_indices]
        token_ids_batched: List[List[int]] = []
        for ind in token_indices:
            token_ids_batched += [ind] * num_images_per_prompt

        if isinstance(bboxes[0][0], float):
            bboxes = [bboxes]  # single batch
        bboxes_batched: List[List[List[float]]] = []
        for bbox in bboxes:
            bboxes_batched += [bbox] * num_images_per_prompt

        assert (
            len(bboxes_batched) == prompt_embeds_refine.shape[0]
        ), f"{len(bboxes_batched)} vs {prompt_embeds_refine.shape[0]}"

        return (
            prompt_embeds_refine,
            add_txt_refine,
            time_ids_refine,
            bboxes_batched,
            token_ids_batched,
        )

    def _compute_loss(
        self, token_indices: List[int], bboxes: List[List[float]]
    ) -> torch.Tensor:
        """Compute a loss that encourages attention mass inside the bboxes.

        For each attention map and each object box this computes the fraction
        of attention falling inside the box and returns the mean squared
        deviation from 1.0 (i.e. encourage full attention inside box).

        Args:
            token_indices: List of token positions (int) identifying object
                tokens in the cross-attention maps.
            bboxes: List of bounding boxes in normalized coordinates
                (x_min, y_min, x_max, y_max) where values are in [0, 1].

        Returns:
            A scalar :class:`torch.Tensor` representing the averaged loss.
        """
        loss = 0
        object_number = len(bboxes)
        total_maps = 0
        for location in [
            "mid",
            "up",
        ]:  # In paper they only update mid and up blocks
            for attn_map_integrated in self.attention_store.maps(location):
                b, i, j = attn_map_integrated.shape
                H = W = int(np.sqrt(i))

                total_maps += 1
                for obj_idx in range(object_number):
                    obj_loss = 0
                    obj_box = bboxes[obj_idx]

                    x_min, y_min, x_max, y_max = (
                        obj_box[0] * W,
                        obj_box[1] * H,
                        obj_box[2] * W,
                        obj_box[3] * H,
                    )
                    mask = torch.zeros((H, W), device=self._execution_device)
                    mask[round(y_min) : round(y_max), round(x_min) : round(x_max)] = 1
                    assert isinstance(token_indices[obj_idx], int)
                    obj_position = token_indices[obj_idx]
                    ca_map_obj = attn_map_integrated[:, :, obj_position].reshape(
                        b, H, W
                    )
                    activation_value = (ca_map_obj * mask).reshape(b, -1).sum(
                        dim=-1
                    ) / ca_map_obj.reshape(b, -1).sum(dim=-1)

                    obj_loss += torch.mean((1 - activation_value) ** 2)
                    loss += obj_loss

        loss /= object_number * total_maps
        return loss

    def _perform_iterative_refinement_step(
        self,
        sigmas: float,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        add_text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
        t: int,
        bboxes: List[List[float]],
        token_indices: List[int],
        max_guidance_iter_per_step: int,
        scale_factor: float,
    ) -> torch.Tensor:
        """Run a few gradient-based refinement iterations on latents.

        This performs up to ``max_guidance_iter_per_step`` steps of iterative
        refinement: it enables gradients on the latents, computes attention
        maps via a forward pass, computes the attention loss w.r.t. ``bboxes``
        and ``token_indices``, backprops to obtain gradients on the latents,
        and applies a simple Euler-like update scaled by ``sigmas``.

        Args:
            sigmas: Noise scale for the current timestep (used to scale updates).
            latents: Input latent tensor for the refinement step (batch dim may be 1).
            prompt_embeds: Conditional prompt embeddings for the forward pass.
            add_text_embeds: Additional conditioning text embeddings.
            time_ids: Time id tensor for added conditioning.
            t: Current timestep index (scheduler timestep).
            bboxes: List of normalized bounding boxes for objects.
            token_indices: List of token indices corresponding to objects.
            max_guidance_iter_per_step: Maximum number of refinement iterations.
            scale_factor: Factor to scale the computed attention loss.

        Returns:
            The refined latents as a :class:`torch.Tensor`.
        """

        loss = None
        for guidance_iter in range(max_guidance_iter_per_step):
            if loss is not None and loss.item() / scale_factor < 0.2:
                break
            latents = latents.clone().detach().requires_grad_(True)
            latent_input = self.scheduler.scale_model_input(latents, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": time_ids}
            self.unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            )
            self.unet.zero_grad()
            loss = self._compute_loss(token_indices, bboxes) * scale_factor
            grad_cond = torch.autograd.grad(
                loss,
                [latents],
                retain_graph=False,
            )[0]
            latents = latents - grad_cond * sigmas**2

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        token_indices: Union[List[int], List[List[int]]] = None,
        bboxes: Union[
            List[List[List[float]]],
            List[List[float]],
        ] = None,
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
        max_guidance_iter: int = 10,
        max_guidance_iter_per_step: int = 5,
        scale_factor: int = 30,
        attn_res: Optional[Tuple[int, int]] = None,
        optimized_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[StableDiffusionXLPipelineOutput, Tuple[torch.Tensor, ...]]:
        """Generate images with optional layout-guided refinement.

        This overrides the base pipeline call to add layout-guided iterative
        refinement steps. It supports the usual SDXL inputs plus ``bboxes``
        and ``token_indices`` which are used to guide attention maps during
        early denoising steps.

        Returns either a :class:`StableDiffusionXLPipelineOutput` when
        ``return_dict`` is True, or a tuple containing the images when
        ``return_dict`` is False.
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
            bboxes,
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

        # Prepare for latent refinement
        (
            prompt_embeds_refine,
            add_txt_refine,
            time_ids_refine,
            bboxes_batched,
            token_ids_batched,
        ) = self._prepare_for_layout_guidance(
            width,
            height,
            attn_res,
            batch_size,
            num_images_per_prompt,
            prompt_embeds,
            add_text_embeds,
            add_time_ids,
            do_classifier_free_guidance,
            token_indices,
            bboxes,
            optimized_layers=optimized_layers,
        )

        # Start denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < max_guidance_iter:
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        new_latents = []
                        for (
                            _latent,
                            _prompt_emb,
                            _add_text_emb,
                            _time_ids,
                            _bboxes,
                            _tok_ids,
                        ) in zip(
                            latents,
                            prompt_embeds_refine,
                            add_txt_refine,
                            time_ids_refine,
                            bboxes_batched,
                            token_ids_batched,
                        ):

                            new_latents.append(
                                self._perform_iterative_refinement_step(
                                    self.scheduler.sigmas[i],
                                    _latent.unsqueeze(0),
                                    _prompt_emb.unsqueeze(0),
                                    _add_text_emb.unsqueeze(0),
                                    _time_ids.unsqueeze(0),
                                    t,
                                    _bboxes,
                                    _tok_ids,
                                    max_guidance_iter_per_step,
                                    scale_factor,
                                )
                            )

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

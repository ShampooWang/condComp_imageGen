from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import stanza
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from diffusers.utils import deprecate, logging
from nltk.tree import Tree
from stanza.pipeline.core import DownloadMethod
from transformers.tokenization_utils import BatchEncoding

logger = logging.get_logger(__name__)


STRUCT_ATTENTION_TYPE = Literal["extend_str", "extend_seq", "align_seq", "none"]


@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]


@dataclass
class KeyValueTensors(object):
    k: torch.Tensor
    v: torch.Tensor

    @property
    def shape(self):
        """Return (k.shape, v.shape) as a tuple."""
        assert self.k.shape == self.v.shape
        return self.k.shape

    def to(self, device):
        return KeyValueTensors(k=self.k.to(device), v=self.v.to(device))


@dataclass
class StructPromptEmbeddings(object):
    full: torch.Tensor
    concepts: List[torch.Tensor]

    def to(self, device):
        return StructPromptEmbeddings(
            full=self.full.to(device),
            concepts=[embed.to(device) for embed in self.concepts],
        )


class StructuredAttnProcessor(AttnProcessor2_0):
    def __init__(
        self,
        struct_attention: bool = False,
    ) -> None:
        super().__init__()
        self.struct_attention = struct_attention

    def struct_qkv(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        uncond_context: torch.Tensor,
        full_prompt_context: torch.Tensor,
        concepts_contexts: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, L, D = full_prompt_context.shape

        # Uncoditional attention
        q_uncond = attn.head_to_batch_dim(attn.to_q(hidden_states[:B]))
        M_uncond = attn.get_attention_scores(
            q_uncond, attn.head_to_batch_dim(attn.to_k(uncond_context)), mask
        )
        v_uncond = attn.head_to_batch_dim(attn.to_v(uncond_context))
        hidden_states_uncond = torch.matmul(
            M_uncond, attn.head_to_batch_dim(attn.to_v(uncond_context))
        )

        # Structured attention for conditional embeddings (eq.4 in the paper)
        q_cond = attn.head_to_batch_dim(attn.to_q(hidden_states[B:]))
        K_p = attn.head_to_batch_dim(attn.to_k(full_prompt_context))
        M = attn.get_attention_scores(q_cond, K_p, mask)
        V = [attn.head_to_batch_dim(attn.to_v(ctx)) for ctx in concepts_contexts]
        hidden_state_cond = []
        for v_i in V:
            hidden_state_cond.append(torch.matmul(M, v_i))
        hidden_state_cond = sum(hidden_state_cond) / len(V)  # average (equation 7)
        hidden_states = torch.cat([hidden_states_uncond, hidden_state_cond], dim=0)

        # back to normal dims
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def get_kv(self, context: torch.Tensor) -> KeyValueTensors:
        return KeyValueTensors(k=self.to_k(context), v=self.to_v(context))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[Tuple[torch.Tensor, KeyValueTensors]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ):

        # hidden_states: x
        batch_size, sequence_length, _ = hidden_states.shape

        # same preparation as default processor
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        # we expect encoder_hidden_states to either be:
        # - tuple like (uc_context, KeyValueTensors(...)) (structured)
        # - None or a Tensor (normal)
        if isinstance(encoder_hidden_states, tuple):
            assert isinstance(encoder_hidden_states[1], StructPromptEmbeddings)
            hidden_states = self.struct_qkv(
                attn,
                hidden_states,
                encoder_hidden_states[0],
                encoder_hidden_states[1].full,
                encoder_hidden_states[1].concepts,
                attention_mask,
            )
        else:
            # standard self or cross-attention
            hidden_states = super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        return hidden_states


class StructureGuidanceSDXL(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        image_encoder=None,
        feature_extractor=None,
        force_zeros_for_empty_prompt=True,
        add_watermarker=None,
    ):
        super().__init__(
            vae,
            text_encoder,
            text_encoder_2,
            tokenizer,
            tokenizer_2,
            unet,
            scheduler,
            image_encoder,
            feature_extractor,
            force_zeros_for_empty_prompt,
            add_watermarker,
        )

        # Build a dict of processors: one per attention layer
        processors = {}
        for name, proc in self.unet.attn_processors.items():
            if "attn2" in name:  # cross attention
                processors[name] = StructuredAttnProcessor(struct_attention=True)
            else:  # self attention (attn1)
                processors[name] = StructuredAttnProcessor(struct_attention=False)

        self.unet.set_attn_processor(processors)

        self.nlp = stanza.Pipeline(
            lang="en",
            processors="tokenize,pos,constituency",
            use_gpu=False,
            dir="./stanza_resources",
            download_method=DownloadMethod.REUSE_RESOURCES,  # DO NOT go online
        )

    def encode_text_ids(self, text_ids, return_last=False):
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
            if self.text_encoder is not None
            else [self.text_encoder_2]
        )
        if return_last:
            prompt_embeds = text_encoders[-1](
                text_ids.to(self.device), output_hidden_states=True
            )[0]
        else:
            prompt_embeds_list = []
            for text_encoder in text_encoders:
                prompt_embeds = text_encoder(
                    text_ids.to(self.device), output_hidden_states=True
                )
                prompt_embeds_list.append(prompt_embeds.hidden_states[-2])
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        return prompt_embeds

    def preprocess_prompt(self, prompt: str) -> str:
        return prompt.lower().strip().strip(".").strip()

    def get_token_alignment_map(
        self, tree: Tree, tokens: Optional[List[str]]
    ) -> Dict[int, List[int]]:
        """Create a mapping from constituency tree leaf indices to token indices.

        The returned dict maps each leaf index in ``tree.leaves()`` to a list
        of token indices (positions in the tokenizer output) that correspond
        to that leaf. When ``tokens`` is ``None``, a trivial identity mapping
        is returned.

        Args:
            tree: An NLTK :class:`Tree` representing constituency parse.
            tokens: Optional token list produced by the tokenizer for the
                sentence. If provided, alignment consolidates subword tokens
                back to tree leaves.

        Returns:
            A dictionary mapping leaf indices (int) to lists of token indices.
        """

        if tokens is None:
            return {i: [i] for i in range(len(tree.leaves()) + 1)}

        def _get_token(token: str):
            return token[:-4] if token.endswith("</w>") else token

        idx_map: Dict[int, List[int]] = {}
        j = 0
        max_offset = abs(len(tokens) - len(tree.leaves()))
        tree_prev_leaf = ""
        for i, w in enumerate(tree.leaves()):
            token = _get_token(tokens[j])
            idx_map[i] = [j]
            if token == tree_prev_leaf + w:
                tree_prev_leaf = ""
                j += 1
            else:
                if len(token) < len(w):
                    prev = ""
                    while prev + token != w:
                        prev += token
                        j += 1
                        token = _get_token(tokens[j])
                        idx_map[i].append(j)
                        assert j - i <= max_offset
                else:
                    tree_prev_leaf += w
                    j -= 1
                j += 1
        idx_map[i + 1] = [j]
        return idx_map

    def get_sub_nps(
        self,
        tree: Tree,
        full_sent: str,
        left: int,
        right: int,
        idx_map: Dict[int, List[int]],
        highest_only: bool = False,
    ) -> List[SubNP]:
        """Recursively extract noun-phrase subtrees as SubNP objects.

        Walks the provided constituency ``tree`` between the ``left`` and
        ``right`` leaf indices and returns a list of ``SubNP`` dataclasses
        containing the noun phrase text and its token span (using ``idx_map``
        for token alignment). When ``highest_only`` is True, only top-level
        NP nodes (excluding the full sentence) are returned.

        Args:
            tree: An NLTK :class:`Tree` node.
            full_sent: The full sentence string used for comparison.
            left: Left leaf index (inclusive) for this subtree.
            right: Right leaf index (exclusive) for this subtree.
            idx_map: Mapping from leaf indices to token index lists.
            highest_only: If True, return only highest-level NP nodes.

        Returns:
            A list of :class:`SubNP` instances representing noun phrases.
        """

        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []

        sub_nps: List[SubNP] = []

        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
        assert right - left == n_leaves

        if tree.label() == "NP" and n_leaves > 1:
            sub_np = SubNP(
                text=" ".join(tree.leaves()),
                span=Span(left=int(min(idx_map[left])), right=int(min(idx_map[right]))),
            )
            sub_nps.append(sub_np)

            if highest_only and sub_nps[-1].text != full_sent:
                return sub_nps

        for i, subtree in enumerate(tree):
            sub_nps += self.get_sub_nps(
                subtree,
                full_sent,
                left=left + offset[i],
                right=left + offset[i] + n_subtree_leaves[i],
                idx_map=idx_map,
            )
        return sub_nps

    def get_all_nps(
        self,
        tree: Tree,
        full_sent: str,
        tokens: Optional[List[str]] = None,
        highest_only: bool = False,
        lowest_only: bool = False,
    ) -> AllNPs:
        """Extract all noun phrases (NPs) from a constituency tree.

        This returns an :class:`AllNPs` object containing the list of NP
        strings, their token spans, and the subset of lowest-level NPs.

        Args:
            tree: An NLTK :class:`Tree` representing the sentence parse.
            full_sent: The full sentence string corresponding to the tree.
            tokens: Optional token list from the tokenizer to assist alignment.
            highest_only: If True, restrict extraction to highest-level NPs.
            lowest_only: If True, return only the lowest-level NPs as strings
                in the ``nps`` field.

        Returns:
            An :class:`AllNPs` instance with extracted noun phrases and spans.
        """

        start = 0
        end = len(tree.leaves())

        idx_map = self.get_token_alignment_map(tree=tree, tokens=tokens)

        all_sub_nps = self.get_sub_nps(
            tree,
            full_sent,
            left=start,
            right=end,
            idx_map=idx_map,
            highest_only=highest_only,
        )

        lowest_nps: List[SubNP] = []
        for i in range(len(all_sub_nps)):
            span = all_sub_nps[i].span
            lowest = True
            for j in range(len(all_sub_nps)):
                span2 = all_sub_nps[j].span
                if span2.left >= span.left and span2.right <= span.right:
                    lowest = False
                    break
            if lowest:
                lowest_nps.append(all_sub_nps[i])

        if lowest_only:
            all_nps = [lowest_np.text for lowest_np in lowest_nps]

        all_nps = [all_sub_np.text for all_sub_np in all_sub_nps]
        spans = [all_sub_np.span for all_sub_np in all_sub_nps]

        if full_sent and full_sent not in all_nps:
            all_nps = [full_sent] + all_nps
            spans = [Span(left=start, right=end)] + spans

        return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

    def tokenize(self, prompt: Union[str, List[str]]) -> BatchEncoding:
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input

    def get_concepts_embeddings(self, prompt):
        """Extract concept embeddings (noun-phrase based) from a prompt.

        The function parses the prompt into noun phrases (concepts), tokenizes
        them, encodes with the text encoder(s), and returns a list of
        concept-aligned embedding tensors. The first returned entry corresponds
        to the full prompt embedding; subsequent entries represent aligned
        concept embeddings suitable for structured attention.

        Args:
            prompt: Input prompt string.

        Returns:
            A list of :class:`torch.Tensor` embeddings, where each tensor is
            shaped like a prompt embedding and can be used in structured
            attention modules.
        """

        # Extract concepts (noun phrases) from the prompt
        preprocessed_prompt = self.preprocess_prompt(prompt)
        doc = self.nlp(preprocessed_prompt)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))
        all_nps = self.get_all_nps(tree=tree, full_sent=preprocessed_prompt)
        assert all_nps.nps[0] == preprocessed_prompt, "Full prompt mismatch"

        concepts = all_nps.nps
        concept_spans = all_nps.spans
        input_ids = self.tokenize(concepts).input_ids
        embeds = self.encode_text_ids(input_ids)  # shape: (num_concepts, 77, dim)
        prompt_embedding = embeds[0]  # The first concept is always the full prompt
        prompt_toks = input_ids[0]

        # Sequnce alignment in Fig. 3 (mid)
        concept_embed_list = []
        for toks, embed, span in zip(input_ids[1:], embeds[1:], concept_spans[1:]):
            start, end = span.left + 1, span.right + 1  # include sot
            seg_length = end - start
            prompt_embedding[start:end] = embed[1 : 1 + seg_length]
            concept_embed_list.append(prompt_embedding.unsqueeze(0))

        return concept_embed_list

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
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
        **kwargs,
    ) -> Union[StableDiffusionXLPipelineOutput, Tuple[torch.Tensor, ...]]:
        """Generate images using structured-attention guidance.

        This overrides the base pipeline ``__call__`` to support structured
        attention by accepting prompts and internally generating concept
        embeddings used by :class:`StructuredAttnProcessor`. It mirrors the
        base pipeline's behavior but wraps/returns results compatible with
        :class:`StableDiffusionXLPipelineOutput` unless ``return_dict`` is
        False (in which case a tuple is returned).

        Returns:
            Either a :class:`StableDiffusionXLPipelineOutput` (when
            ``return_dict`` is True) or a tuple containing the generated
            images when ``return_dict`` is False.
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

        # We create a new dataclass which has the embedding of full prompt and concepts in the prompt
        prompt_embeds = StructPromptEmbeddings(
            full=prompt_embeds,
            concepts=self.get_concepts_embeddings(prompt),
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
            prompt_embeds.full.dtype,
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
            dtype=prompt_embeds.full.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.full.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_embeds = (negative_prompt_embeds, prompt_embeds)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = (
            prompt_embeds[0].to(device),
            prompt_embeds[1].to(device),
        )
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

        # Start denoising loop
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
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

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def efficient_mode(self):
        self.unet.enable_gradient_checkpointing()

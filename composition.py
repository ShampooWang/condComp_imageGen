import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sdxl_pipelines.sdxl_img_inpaint import customStableDiffusionXLInpaintPipeline
from utils import (
    get_category_token_indices,
    get_inpaint_mask_by_attn_maps,
    get_tok2id_map,
    min_max_normalize,
)

SOT = "<|startoftext|>"
EOT = "<|endoftext|>"


def multiobj_conditional_composition(
    sdxl_pipe: Any,
    inpaint_pipe: Any,
    prompts_list: List[str],
    categories_list: List[List[str]],
    generator: Optional[torch.Generator] = None,
    inpaint_generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    inpaint_noise: Optional[torch.Tensor] = None,
    resample_noise: Optional[torch.Tensor] = None,
    use_layout_guidance: bool = True,
    keep_ratios: Union[float, List[float]] = 0.5,
    add_eot: bool = True,
    scale_factor: int = 50,
    resample_steps: int = 1,
    strength: float = 0.9,
    use_ae_for_joint_stage: bool = False,
    return_img_per_stage: bool = False,
    return_mask_and_map_per_stage: bool = False,
    saved_joint_attn_progress: float = 0.4,
    saved_compose_attn_progress: float = 0.7,
) -> Union[
    Tuple[List[Any]], Tuple[List[Any], List[torch.Tensor], List[Optional[torch.Tensor]]]
]:
    """Run multi-object conditional composition across staged prompts.

    Args:
        sdxl_pipe: SDXL pipeline used to generate initial image and saved attention maps.
        inpaint_pipe: Inpainting pipeline used to iteratively add objects.
        prompts_list: List of prompts for each stage (first entry is the initial prompt).
        categories_list: List of category lists corresponding to each prompt.
        generator: RNG generator for the main pipeline.
        inpaint_generator: RNG generator for the inpainting pipeline.
        latents: Optional latent tensor to seed generation.
        inpaint_noise: Optional noise tensor for inpainting.
        resample_noise: Optional noise tensor for resampling stages.
        use_layout_guidance: Whether to enable layout guidance in composition stages.
        keep_ratios: Per-stage keep ratios or a single float applied to all stages.
        add_eot: Whether to append an end-of-text token when computing masks.
        scale_factor: Scale factor passed to pipelines when relevant.
        resample_steps: Number of resample steps used in inpainting stages.
        strength: Strength parameter for inpainting refinement.
        use_ae_for_joint_stage: Whether to use Attend-and-Excite for the initial joint stage.
        return_img_per_stage: If True, return a list of images for each stage; otherwise return final image only.
        return_mask_and_map_per_stage: If True, also return attention masks and maps per stage.
        saved_joint_attn_progress: Saved attention progress for the joint stage.
        saved_compose_attn_progress: Saved attention progress for composition stages.

    Returns:
        Either `(img_per_stage,)` or `(img_per_stage, attn_map_per_stage, attn_mask_per_stage)` depending on flags.
    """

    prompts_list_copy = prompts_list.copy()
    categories_list_copy = categories_list.copy()
    keep_ratios_copy = keep_ratios.copy()

    if not isinstance(keep_ratios, list):
        keep_ratios = [keep_ratios] * len(prompts_list_copy)

    init_prompt = prompts_list_copy.pop(0)
    init_categories = categories_list_copy.pop(0)
    init_keep_ratio = keep_ratios_copy.pop(0)
    base_token_indices = get_tok2id_map(sdxl_pipe, init_prompt)
    base_target_indices = get_category_token_indices(
        category=init_categories,
        token_map=base_token_indices,
        tokenizer=sdxl_pipe.tokenizer,
    )
    img_per_stage = []
    attn_map_per_stage = []
    attn_mask_per_stage = []

    if use_ae_for_joint_stage:
        init_image = sdxl_pipe(
            prompt=init_prompt,
            num_images_per_prompt=1,
            generator=generator,
            latents=latents,
            token_indices=base_target_indices,
            categories=init_categories,
            tok_map=base_token_indices,
            progress_to_save_attn=saved_joint_attn_progress,
            scale_factor=30,
        ).images[0]
    else:
        init_image = sdxl_pipe(
            prompt=init_prompt,
            num_images_per_prompt=1,
            generator=generator,
            latents=latents,
            token_indices=base_target_indices,
            progress_to_save_attn=saved_joint_attn_progress,
        ).images[0]
    img_per_stage.append(init_image)
    joint_attn_map = sdxl_pipe.saved_attn_maps.to(sdxl_pipe.device)
    attn_map_per_stage.append(joint_attn_map)
    attn_mask_per_stage.append(None)

    if len(prompts_list_copy) == 0:
        if return_img_per_stage:
            return_reult = [img_per_stage]
        else:
            return_reult = [img_per_stage[-1]]

        if return_mask_and_map_per_stage:
            return_reult.extend([attn_map_per_stage, attn_mask_per_stage])

        return_reult = tuple(return_reult)
        return return_reult
    else:
        hard_mask = compute_mask_for_inpaint(
            sdxl_pipe.attention_store,
            get_tok2id_map(sdxl_pipe, init_prompt),
            sdxl_pipe.tokenizer,
            keep_ratio=init_keep_ratio,
            add_eot=add_eot,
            categories=init_categories,
            return_attn_map=False,
        )
        composed_imgs, composed_attn_maps, composed_attn_masks = composition_stage(
            inpaint_pipe=inpaint_pipe,
            prompts=prompts_list_copy,
            add_categories=categories_list_copy,
            init_image=init_image,
            init_mask=hard_mask,
            scale_factor=scale_factor,
            init_categories=init_categories,
            generator=inpaint_generator,
            inpaint_noise=inpaint_noise,
            resample_noise=resample_noise,
            add_eot=add_eot,
            use_layout_guidance=use_layout_guidance,
            keep_ratios=keep_ratios_copy,
            resample_steps=resample_steps,
            strength=strength,
            saved_compose_attn_progress=saved_compose_attn_progress,
            return_img_per_stage=True,
            return_mask_and_map_per_stage=True,
        )
        img_per_stage.extend(composed_imgs)
        attn_map_per_stage.extend(composed_attn_maps)
        attn_mask_per_stage.extend(composed_attn_masks)
    del sdxl_pipe.saved_attn_maps

    if return_img_per_stage:
        return_reult = [img_per_stage]
    else:
        return_reult = [img_per_stage[-1]]

    if return_mask_and_map_per_stage:
        return_reult.extend([attn_map_per_stage, attn_mask_per_stage])

    return_reult = tuple(return_reult)
    return return_reult


def composition_stage(
    inpaint_pipe,
    prompts,
    add_categories,
    init_image,
    init_mask,
    init_categories,
    scale_factor=50,
    use_layout_guidance=True,
    inpaint_noise=None,
    generator=None,
    resample_noise=None,
    strength=0.7,
    keep_ratios=0.4,
    add_eot=True,
    resample_steps=1,
    return_img_per_stage=False,
    return_mask_and_map_per_stage=False,
    saved_compose_attn_progress=0.4,
):
    """Perform iterative composition (inpainting) stages to add categories to an image.

    Args:
        inpaint_pipe: Inpainting pipeline used to refine images.
        prompts: List of prompts for each refinement stage.
        add_categories: List of categories to add at each stage.
        init_image: Initial PIL image produced by the base pipeline.
        init_mask: Initial inpaint mask (torch.Tensor) where new content should be placed.
        init_categories: List of categories already present in the image.
        scale_factor: Scale factor passed to the inpainting pipeline.
        use_layout_guidance: Whether to use layout guidance during inpainting.
        inpaint_noise: Optional noise tensor for inpainting.
        generator: Optional torch.Generator for reproducibility.
        resample_noise: Optional resampling noise tensor.
        strength: Strength of inpainting refinement.
        keep_ratios: Per-stage keep ratios or single float applied to all stages.
        add_eot: Whether to include end-of-text token when computing masks.
        resample_steps: Number of resample steps used during inpainting.
        return_img_per_stage: If True, return images for each stage instead of only the final image.
        return_mask_and_map_per_stage: If True, also return masks and attention maps per stage.
        saved_compose_attn_progress: Saved attention progress value for composition stages.

    Returns:
        Either `(img_per_stage,)` or `(img_per_stage, attn_mask_per_stage, attn_map_per_stage)` depending on flags.
    """

    # Type hints for local variables (kept for clarity)
    # num_inference_steps: int
    # prev_image: Any
    # prev_mask: torch.Tensor

    num_inference_steps = int(50 / strength)
    prev_image = init_image
    prev_mask = init_mask
    categories = init_categories.copy()
    img_per_stage = []
    attn_mask_per_stage = []
    attn_map_per_stage = []

    if not isinstance(keep_ratios, list):
        keep_ratios = [keep_ratios] * len(add_categories)

    for i, (category, prompt) in enumerate(zip(add_categories, prompts)):
        refine_tok_map = get_tok2id_map(inpaint_pipe, prompt)
        token_indices = get_category_token_indices(
            category=category,
            token_map=refine_tok_map,
            tokenizer=inpaint_pipe.tokenizer,
        )
        refine_img = inpaint_pipe(
            prompt=prompt,
            image=prev_image,
            mask_image=prev_mask,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            register_attn=True,
            resample_steps=resample_steps,
            layout_guidance=use_layout_guidance,
            token_indices=token_indices,
            scale_factor=scale_factor,
            inpaint_noise=inpaint_noise,
            resample_noise=resample_noise,
            progress_to_save_attn=saved_compose_attn_progress,
        ).images[0]

        img_per_stage.append(refine_img)
        prev_image = refine_img
        categories.append(category)

        inpaint_maps = inpaint_pipe.attention_store.aggregate(
            where=["up", "mid", "down"] if not use_layout_guidance else ["up", "mid"]
        )[:, :, token_indices].mean(dim=-1)
        attn_map_per_stage.append(inpaint_maps)
        attn_mask_per_stage.append(inpaint_pipe.attn_map_mask[0, 0])

        if hasattr(inpaint_pipe, "saved_attn_maps"):
            del inpaint_pipe.saved_attn_maps
        torch.cuda.empty_cache()

        # Update mask for next iteration
        if i != len(add_categories) - 1:
            prev_mask = compute_mask_for_inpaint(
                inpaint_pipe.attention_store,
                refine_tok_map,
                inpaint_pipe.tokenizer,
                keep_ratio=keep_ratios[i],
                categories=categories,
                add_eot=add_eot,
            )

    if return_img_per_stage:
        return_reult = [img_per_stage]
    else:
        return_reult = [refine_img]

    if return_mask_and_map_per_stage:
        return_reult.extend([attn_map_per_stage, attn_mask_per_stage])

    return_reult = tuple(return_reult)

    return return_reult


def get_inpaint_model(
    device: Union[torch.device, str],
    sdxl_pipe: Optional[Any] = None,
    use_inpaint_ckpt: bool = False,
) -> Any:
    """Return a Stable Diffusion XL inpainting pipeline on the requested device.

    Args:
        device: Device (e.g., 'cuda:0' or torch.device) to move the model to.
        sdxl_pipe: Optional existing SDXL pipeline instance to reuse components from when
            `use_inpaint_ckpt` is False. Required if not using a pretrained inpaint checkpoint.
        use_inpaint_ckpt: If True, load the official pretrained inpainting checkpoint from
            the Diffusers hub. Otherwise construct an inpaint pipeline by reusing components
            from `sdxl_pipe`.

    Returns:
        An inpainting pipeline instance ready for inference (in eval mode) on `device`.
    """

    if use_inpaint_ckpt:
        inpaint_pipe = customStableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
        ).to(device)
    else:
        assert (
            sdxl_pipe is not None
        ), "sdxl_pipe must be provided if not using inpaint_ckpt"

        # check whether sdxl_pipe is in eval mode
        assert not sdxl_pipe.unet.training
        assert not sdxl_pipe.vae.training
        assert not sdxl_pipe.text_encoder.training
        assert not sdxl_pipe.text_encoder_2.training

        inpaint_pipe = customStableDiffusionXLInpaintPipeline(
            vae=sdxl_pipe.vae,
            text_encoder=sdxl_pipe.text_encoder,
            text_encoder_2=sdxl_pipe.text_encoder_2,
            tokenizer=sdxl_pipe.tokenizer,
            tokenizer_2=sdxl_pipe.tokenizer_2,
            unet=sdxl_pipe.unet,
            scheduler=sdxl_pipe.scheduler,
        ).to(device)

    return inpaint_pipe


def compute_mask_for_inpaint(
    attention_store: Any,
    tok_ids: Dict[str, Any],
    tokenizer: Any,
    keep_ratio: float = 0.2,
    add_eot: bool = True,
    categories: List[str] = ["person", "apple", "cat"],
    eot_only: bool = False,
    aggregate_method: str = "mean",
    return_attn_map: bool = False,
    target_H: int = 1024,
    target_W: int = 1024,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute an inpainting mask from attention maps for given categories.

    Args:
        attention_store: Attention store object exposing `aggregate()`.
        tok_ids: Mapping from token strings to token indices.
        tokenizer: Tokenizer to split category names when needed.
        keep_ratio: Fraction of area to keep (not inpainted).
        add_eot: Whether to include end-of-text token in the aggregation.
        categories: List of category names to use when computing attention maps.
        eot_only: If True, only use the end-of-text token.
        aggregate_method: How to aggregate per-category maps ('mean' or 'max').
        return_attn_map: If True, also return the aggregated attention map.
        target_H: Target height for upsampled attention maps.
        target_W: Target width for upsampled attention maps.

    Returns:
        If `return_attn_map` is False, returns the inverted hard mask (torch.Tensor).
        If True, returns a tuple `(mask, target_attn_map)` where both are torch.Tensors on CPU.
    """
    categories_copy = categories.copy()
    attn_maps = attention_store.aggregate(where=["up", "mid", "down"])

    # Extract attention map of target tokens
    attn_map_list = []
    if add_eot:
        categories_copy.append(EOT)
    if eot_only:
        categories_copy = [EOT]

    for category in categories_copy:
        if category == EOT:
            category = "<|endoftext|>"
        else:
            category += "</w>"

        if category not in tok_ids:
            subtoks = [
                idx
                for tok in tokenizer.tokenize(category.strip("</w>"))
                if tok in tok_ids
                for idx in tok_ids[tok]
            ]
            _attn_map = attn_maps[
                :,
                :,
                subtoks,
            ].mean(-1)
        else:
            if isinstance(tok_ids[category], int):
                subtoks = [tok_ids[category]]
            else:
                subtoks = tok_ids[category]
            _attn_map = attn_maps[:, :, subtoks].mean(-1)
        _attn_map = min_max_normalize(_attn_map)
        attn_map_list.append(_attn_map)

    if aggregate_method == "mean":
        target_attn_map = torch.stack(attn_map_list, dim=-1).mean(-1)
    elif aggregate_method == "max":
        target_attn_map = torch.stack(attn_map_list, dim=-1).max(-1)[0]
    else:
        raise ValueError(f"Unknown aggregate_method: {aggregate_method}")

    # Compute hard mask from attention map, 1 for inpaint area, 0 for keep area
    target_mask, target_attn_map_up = get_inpaint_mask_by_attn_maps(
        attn_maps=target_attn_map,
        keep_ratio=keep_ratio,
        target_H=target_H,
        target_W=target_W,
        smoothed=True,
    )
    target_mask = target_mask.to("cpu")
    target_mask = (-1 * target_mask) + 1  # invert mask for inpainting
    target_attn_map_up = target_attn_map_up.float().to("cpu")
    target_attn_map_up = min_max_normalize(target_attn_map_up)
    target_attn_map_up = (-1 * target_attn_map_up) + 1

    if return_attn_map:
        target_attn_map = min_max_normalize(target_attn_map)
        return target_mask, target_attn_map
    else:
        return target_mask

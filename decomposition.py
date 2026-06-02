import os
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.transforms.functional import pil_to_tensor

from composition import (
    multiobj_conditional_composition,
)
from utils import (
    get_tok2id_map,
    min_max_normalize,
    prepare_condComp_prompts,
)

faster_rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT


def attnScore_decomposition(
    sdxl_pipe: Any,
    inpaint_pipe: Any,
    categories: List[str],
    keep_ratio_offset: int,
    latents: torch.Tensor,
    inpaint_noise: torch.Tensor,
    resample_noise: Optional[torch.Tensor] = None,
    t_joint: float = 0.4,
    t_comp: float = 0.7,
    resample_steps: int = 1,
    layout_guidance: bool = True,
    strength: float = 1.0,
    return_scores: bool = False,
    attn_threshold: float = 0.5,
    scale_factor: int = 50,
    decomposition_size: int = 1,
    use_ae_for_joint_stage: bool = False,
    drop_stages: Optional[List[bool]] = None,
) -> Union[Tuple[Any, List[str]], Tuple[Any, List[str], Dict[str, Any]]]:
    """Estimate attention-based decomposition scores for candidate added categories.

    Args:
        sdxl_pipe: SDXL pipeline used to compute attention maps.
        inpaint_pipe: Inpainting pipeline used in composition stages.
        categories: List of existing category names in the original prompt.
        keep_ratio_offset: Offset added to the keep-ratio denominator to reserve capacity for future composition stages.
        latents: Latent tensor used for generation.
        inpaint_noise: Noise tensor fed to the inpainting model.
        resample_noise: Optional noise used for resampling stages.
        t_joint: Timestep to save for joint attention progress.
        t_comp: Timestep to save for composition attention progress.
        resample_steps: Number of resample steps to run.
        layout_guidance: Whether to enable layout guidance during composition.
        strength: Strength parameter passed to composition routine.
        return_scores: If True, return a dict of per-category scores as a third element.
        attn_threshold: Threshold to binarize attention maps when computing scores.
        scale_factor: Scale factor used by certain pipelines.
        decomposition_size: Number of objects are removed from the base stage and added later.
        use_ae_for_joint_stage: Whether to use Attend-and-Excite for joint stage.
        drop_stages: Optional list of booleans indicating dropped stages.

    Returns:
        Either a tuple `(img_per_stage, decomp_cat)` or `(img_per_stage, decomp_cat, category_scores)` when `return_scores` is True.
    """

    category_scores = {}
    sampled_imgs = {}
    for cat in permutations(categories, decomposition_size):
        cat = list(cat)
        prompts_list, categories_list, keep_ratios = prepare_condComp_prompts(
            categories,
            cat,
            keep_ratio_offset=keep_ratio_offset,
            drop_stages=drop_stages,
        )

        if len(cat) > 1:
            added_cats = "_".join(cat)
        else:
            added_cats = cat[0]

        category_scores[added_cats] = compute_attnScore(
            sdxl_pipe=sdxl_pipe,
            inpaint_pipe=inpaint_pipe,
            prompts_list=prompts_list,
            categories_list=categories_list,
            keep_ratios=keep_ratios,
            latents=latents,
            inpaint_noise=inpaint_noise,
            resample_noise=resample_noise,
            t_joint=t_joint,
            t_comp=t_comp,
            resample_steps=resample_steps,
            use_layout_guidance=layout_guidance,
            strength=strength,
            attn_threshold=attn_threshold,
            scale_factor=scale_factor,
            use_ae_for_joint_stage=use_ae_for_joint_stage,
        )
        sampled_imgs[added_cats] = category_scores[added_cats]["img_per_stage"]
        del category_scores[added_cats]["img_per_stage"]
        torch.cuda.empty_cache()

    best_cat = max(
        category_scores,
        key=lambda x: category_scores[x]["full_score"],
    )
    decomp_cat = best_cat.split("_")
    if return_scores:
        return sampled_imgs[best_cat], decomp_cat, category_scores

    return sampled_imgs[best_cat], decomp_cat


def compute_joint_score(
    attn_maps: torch.Tensor,
    category: List[str],
    token_map: Dict[str, Any],
    tokenizer: Any,
    threshold: float = 0.4,
    scale_by_max_attn: bool = False,
) -> float:
    """Compute a 'joint' score measuring how distinct each category's attention is from others.

    Args:
        attn_maps: Attention maps tensor of shape (heads, tokens, seq_len) or similar.
        category: Sequence of category names to evaluate.
        token_map: Mapping from token strings to tokenizer ids/indices.
        tokenizer: Tokenizer instance used to split category names when needed.
        threshold: Minimum attention value to consider when computing maps.
        scale_by_max_attn: If True, scale per-token outside score by its max attention weight.

    Returns:
        float: The computed joint score (lower is better separation in this formulation).
    """

    all_tok_maps = []
    for _cat in category:
        if f"{_cat}</w>" in token_map:
            all_tok_maps.append(attn_maps[:, :, token_map[f"{_cat}</w>"]].mean(-1))
        else:
            tokens = tokenizer.tokenize(_cat)
            tokens = [tok for tok in tokens if tok in token_map]
            tok_ids = []
            for tok in tokens:
                ids = token_map[tok]
                if isinstance(ids, list):
                    tok_ids.extend(ids)
                else:
                    tok_ids.append(ids)
            agg_map = attn_maps[:, :, tok_ids].mean(dim=-1)
            all_tok_maps.append(agg_map)

    all_tok_maps = torch.stack(all_tok_maps, dim=-1)
    original_max_weight = [torch.max(tok_maps).item() for tok_maps in all_tok_maps]
    all_tok_maps = torch.stack(
        [
            min_max_normalize(all_tok_maps[:, :, i])
            for i in range(all_tok_maps.shape[-1])
        ],
        dim=-1,
    )
    all_tok_maps[all_tok_maps < threshold] = 0
    outside_scores = []
    for i in range(all_tok_maps.shape[-1]):
        other_maps = all_tok_maps[
            :, :, [j for j in range(all_tok_maps.shape[-1]) if j != i]
        ]
        if other_maps.shape[-1] == 0:
            max_other = torch.zeros_like(all_tok_maps[i])
        else:
            max_other, _ = torch.max(other_maps, dim=-1)

        outsides = torch.clamp(all_tok_maps[:, :, i] - max_other, min=0)
        outside_score = outsides.sum() / (all_tok_maps[:, :, i].sum() + 1e-8)
        if scale_by_max_attn:
            outside_score = outside_score * original_max_weight[i]
        outside_scores.append(outside_score.item())
    score = min(outside_scores)

    return score


def compute_composition_score(
    inpaint_mask: torch.Tensor,
    new_attn_map: torch.Tensor,
    threshold: float = 0.6,
    eps: float = 1e-8,
) -> float:
    """Compute a score indicating how much the new attention overlaps the inpaint mask.

    Args:
        inpaint_mask: Binary or soft mask indicating the inpaint region.
        new_attn_map: Attention map produced at the composition stage.
        threshold: Minimum normalized attention to include in the computation.
        eps: Small epsilon to avoid division by zero.

    Returns:
        float: Composition overlap score in [0, 1].
    """

    new_attn_map = new_attn_map.float()
    inpaint_mask = inpaint_mask.float().to(new_attn_map.device)
    new_attn_map = min_max_normalize(new_attn_map, eps)

    # set value < threshold to 0 to focus on main attention area
    new_attn_map = torch.where(new_attn_map < threshold, 0.0, new_attn_map)
    outside = new_attn_map * inpaint_mask
    score = torch.sum(outside) / (torch.sum(new_attn_map) + eps)

    if isinstance(score, torch.Tensor):
        return float(score.item())

    return float(score)


def compute_attnScore(
    sdxl_pipe: Any,
    inpaint_pipe: Any,
    prompts_list: List[str],
    categories_list: List[Any],
    keep_ratios: List[float],
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    inpaint_noise: Optional[torch.Tensor] = None,
    resample_noise: Optional[torch.Tensor] = None,
    add_eot: bool = True,
    t_joint: float = 0.7,
    t_comp: float = 0.7,
    resample_steps: int = 1,
    use_layout_guidance: bool = True,
    strength: float = 1.0,
    scale_factor: int = 50,
    attn_threshold: float = 0.6,
    use_ae_for_joint_stage: bool = False,
) -> Dict[str, Any]:
    """Compute attention-based scores for a sequence of composition stages.

    This function runs the multi-object conditional composition pipeline and computes
    a joint attention score plus per-stage composition scores.

    Args:
        sdxl_pipe: SDXL pipeline used for forward passes and obtaining attention maps.
        inpaint_pipe: Inpainting pipeline used during composition.
        prompts_list: Sequence of prompts used during staged composition.
        categories_list: Sequence of category lists corresponding to each prompt.
        keep_ratios: Ratios indicating how much to keep from previous stages.
        generator: Optional torch.Generator for reproducibility.
        latents: Optional latents tensor passed to the pipeline.
        inpaint_noise: Optional inpaint noise tensor.
        resample_noise: Optional resample noise tensor.
        add_eot: Whether to append an end-of-text token for tokenizer alignment.
        t_joint: Timestep to save for joint attention progress.
        t_comp: Timestep to save for composition attention progress.
        resample_steps: Number of resample steps.
        use_layout_guidance: Whether to enable layout guidance during composition.
        strength: Strength parameter for composition.
        scale_factor: Scale factor to pass to the pipeline.
        attn_threshold: Threshold used for attention-based computations.
        use_ae_for_joint_stage: Whether to use Attend-and-Excite for the joint stage.

    Returns:
        Dict[str, Any]: Dictionary containing keys 'img_per_stage', 'joint_score', 'comp_score', and 'full_score'.
    """

    img_per_stage, attn_map_per_stage, attn_mask_per_stage = (
        multiobj_conditional_composition(
            sdxl_pipe=sdxl_pipe,
            inpaint_pipe=inpaint_pipe,
            prompts_list=prompts_list,
            categories_list=categories_list,
            latents=latents,
            inpaint_noise=inpaint_noise,
            resample_noise=resample_noise,
            keep_ratios=keep_ratios,
            resample_steps=resample_steps,
            generator=generator,
            strength=strength,
            add_eot=add_eot,
            use_layout_guidance=use_layout_guidance,
            return_img_per_stage=True,
            use_ae_for_joint_stage=use_ae_for_joint_stage,
            return_mask_and_map_per_stage=True,
            scale_factor=scale_factor,
            saved_joint_attn_progress=t_joint,
            saved_compose_attn_progress=t_comp,
        )
    )
    base_token_indices = get_tok2id_map(sdxl_pipe, prompts_list[0])
    s_joint = compute_joint_score(
        attn_map_per_stage[0].to(sdxl_pipe.device),
        categories_list[0],
        base_token_indices,
        sdxl_pipe.tokenizer,
        threshold=attn_threshold,
    )

    s_comp = []
    for j in range(1, len(prompts_list)):
        comp_score = compute_composition_score(
            inpaint_mask=attn_mask_per_stage[j],
            new_attn_map=attn_map_per_stage[j],
            threshold=attn_threshold,
        )
        s_comp.append(comp_score)
    s_or = s_joint * np.prod(s_comp)

    return {
        "img_per_stage": img_per_stage,
        "joint_score": s_joint,
        "comp_score": s_comp,
        "full_score": s_or,
    }


def objDetect_decomposition(
    sdxl_pipe: Any,
    inpaint_pipe: Any,
    latents: torch.Tensor,
    inpaint_noise: torch.Tensor,
    categories: List[str],
    object_detector: Any,
    keep_ratio_offset: int,
    resample_noise: Optional[torch.Tensor] = None,
    decomposition_size: int = 1,
    resample_steps: int = 1,
    strength: float = 0.9,
    scale_factor: int = 50,
    use_layout_guidance: bool = True,
    drop_stages: Optional[List[bool]] = None,
    use_ae_for_joint_stage: bool = False,
) -> Tuple[Any, str, Dict[str, float]]:
    """Perform decomposition using an object detector (Faster R-CNN) to score candidates.

    Args:
        sdxl_pipe: SDXL pipeline used for generating candidates.
        inpaint_pipe: Inpainting pipeline used during composition.
        latents: Latent tensor for generation.
        inpaint_noise: Noise tensor for inpainting.
        categories: List of original categories in the prompt.
        object_detector: Object detection model (e.g., Faster R-CNN) returning class names and scores.
        keep_ratio_offset: Offset added to the keep-ratio denominator to reserve capacity for future composition stages.
        resample_noise: Optional resample noise tensor.
        decomposition_size: Number of objects are removed from the base stage and added later.
        resample_steps: Number of resampling steps.
        strength: Strength parameter for composition.
        scale_factor: Scale factor for pipelines.
        use_layout_guidance: Whether to enable layout guidance.
        drop_stages: Optional list indicating dropped stages.
        use_ae_for_joint_stage: Whether to use Attend-and-Excite for joint stage.

    Returns:
        Tuple containing `(sampled_img_per_stage, best_category_str, category_scores_dict)`.
    """

    best_cat = None
    sampled_imgs = {}
    category_scores = {}
    for cat in permutations(categories, decomposition_size):
        cat = list(cat)
        prompts_list, categories_list, keep_ratios = prepare_condComp_prompts(
            categories,
            cat,
            keep_ratio_offset=keep_ratio_offset,
            drop_stages=drop_stages,
        )

        if len(cat) > 1:
            added_cats = "_".join(cat)
        else:
            added_cats = cat[0]

        img_per_stage = multiobj_conditional_composition(
            sdxl_pipe=sdxl_pipe,
            inpaint_pipe=inpaint_pipe,
            prompts_list=prompts_list,
            categories_list=categories_list,
            latents=latents,
            inpaint_noise=inpaint_noise,
            resample_noise=resample_noise,
            keep_ratios=keep_ratios,
            resample_steps=resample_steps,
            strength=strength,
            use_layout_guidance=use_layout_guidance,
            return_img_per_stage=True,
            use_ae_for_joint_stage=use_ae_for_joint_stage,
            scale_factor=scale_factor,
        )[0]
        img = img_per_stage[-1]
        sampled_imgs[added_cats] = img_per_stage

        detect_classes, confidence_scores = get_pred_from_fast_rcnn(
            object_detector, img, sdxl_pipe.device
        )
        det_names = set(detect_classes)

        if set(categories).issubset(det_names) and best_cat is None:
            best_cat = added_cats
            # break
        else:
            conf_dict = {}
            for cls, conf in zip(detect_classes, confidence_scores):
                conf_dict[cls] = max(conf_dict.get(cls, 0), conf)
            category_scores[added_cats] = (
                min(conf_dict.values()) if len(conf_dict) > 0 else 0.0
            )

    if best_cat is None:
        best_cat = max(
            category_scores,
            key=category_scores.get,
        )

    return sampled_imgs[best_cat], best_cat, category_scores


@torch.inference_mode()
def get_pred_from_fast_rcnn(
    model: Any,
    image: Any,
    device: Union[torch.device, str],
    conf_threshold: float = 0.7,
) -> Tuple[List[str], List[float]]:
    """Run Faster R-CNN on a single image and return detected class names and confidences.

    Args:
        model: The Faster R-CNN model instance.
        image: PIL Image to run detection on.
        device: Target device for tensors (e.g., 'cuda:0' or torch.device).
        conf_threshold: Confidence threshold to filter detections.

    Returns:
        Tuple[List[str], List[float]]: Detected class names and corresponding confidence scores.
    """
    preprocess = faster_rcnn_weights.transforms()
    img = pil_to_tensor(image).to(device)
    batch = [preprocess(img)]
    pred = model(batch)[0]
    scores = pred["scores"]
    labels = pred["labels"]
    det_names = [
        faster_rcnn_weights.meta["categories"][label]
        for label, score in zip(labels, scores)
        if score > conf_threshold
    ]
    scores = [score.item() for score in pred["scores"] if score > conf_threshold]

    return det_names, scores


def get_fastrcnn(device: Union[torch.device, str]) -> Any:
    """Instantiate and return a Faster R-CNN model on the given device.

    Args:
        device: Device to move the model to.

    Returns:
        The Faster R-CNN model ready for inference (in eval mode).
    """

    import torchvision

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=faster_rcnn_weights
    ).to(device)
    model.eval()

    return model

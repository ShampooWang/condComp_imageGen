from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def min_max_normalize(tensor, eps=1e-6):
    """Normalize a tensor to the [0,1] range using min-max scaling.

    Args:
        tensor: Input tensor (torch or numpy supported via duck-typing).
        eps: Small epsilon to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)

    return normalized_tensor


def prepare_for_layout_guidance(
    width: int,
    height: int,
    attn_res: Optional[Tuple[int, int]],
    batch_size: int,
    num_images_per_prompt: int,
    prompt_embeds: "torch.Tensor",
    add_text_embeds: "torch.Tensor",
    add_time_ids: "torch.Tensor",
    do_classifier_free_guidance: bool,
    token_indices: Union[List[int], List[List[int]]],
    bboxes: Optional[Union[List[List[List[float]]], List[List[float]]]] = None,
):
    """Prepare prompt embeddings and batched token/bbox lists for refinement.

    Returns (prompt_embeds_refine, add_txt_refine, time_ids_refine,
    bboxes_batched, token_ids_batched).
    """

    if attn_res is None:
        attn_res = (int(np.ceil(width / 32)), int(np.ceil(height / 32)))

    # 1: Split embeds (ignore uncoditional part) for refinement batch
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

    # 2: Make token indices & bboxes batched for num_images_per_prompt
    if isinstance(token_indices[0], int):
        token_indices = [token_indices]
    token_ids_batched: List[List[int]] = []
    for ind in token_indices:
        token_ids_batched += [ind] * num_images_per_prompt

    if bboxes is None:
        bboxes = [[None, None, None, None]] * len(token_indices[0])

    if bboxes[0][0] is None or isinstance(bboxes[0][0], float):
        bboxes = [bboxes]  # single batch
        bboxes_batched: List[List[List[float, None]]] = []
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


def compute_layout_guidance_loss(
    model,
    token_indices: List[int],
    bboxes: List[List[float]],
    mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Compute layout-guidance loss from model.attention_store.

    The loss encourages attention mass for the specified ``token_indices``
    to fall inside the corresponding ``bboxes``. Returns a scalar tensor.
    """

    loss = 0
    object_number = len(bboxes)
    total_maps = 0

    for location in [
        "mid",
        "up",
    ]:  # In Layout Guidance paper they only update mid and up blocks
        for _, attn_map_integrated in enumerate(model.attention_store.maps(location)):
            b, i, j = attn_map_integrated.shape
            H = W = int(np.sqrt(i))

            total_maps += 1
            for obj_idx in range(object_number):
                obj_loss = 0
                obj_box = bboxes[obj_idx]

                if mask is None:
                    x_min, y_min, x_max, y_max = (
                        obj_box[0] * W,
                        obj_box[1] * H,
                        obj_box[2] * W,
                        obj_box[3] * H,
                    )
                    mask = torch.zeros((H, W), device=model._execution_device)
                    mask[round(y_min) : round(y_max), round(x_min) : round(x_max)] = 1

                assert isinstance(token_indices[obj_idx], int)
                obj_position = token_indices[obj_idx]
                ca_map_obj = attn_map_integrated[:, :, obj_position].reshape(b, H, W)
                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(
                    dim=-1
                ) / ca_map_obj.reshape(b, -1).sum(dim=-1)
                obj_loss += torch.mean((1 - activation_value) ** 2)
                loss += obj_loss

    assert (
        total_maps > 0
    ), f"{model.attention_store.maps('up')}, {model.attention_store.maps('mid')}"
    loss /= object_number * total_maps

    return loss


def perform_iterative_refinement_step(
    model,
    sigmas: float,
    latents: "torch.Tensor",
    mask: "torch.Tensor",
    masked_image_latents: "torch.Tensor",
    num_channels_unet: int,
    prompt_embeds: "torch.Tensor",
    add_text_embeds: "torch.Tensor",
    time_ids: "torch.Tensor",
    t: int,
    bboxes: List[List[float]],
    token_indices: Union[List[int], List[List[int]]],
    max_guidance_iter_per_step: int,
    scale_factor: float,
    attn_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Perform iterative latent refinement using attention-based loss.

    Runs up to ``max_guidance_iter_per_step`` gradient refinement iterations
    on the provided ``latents`` to increase attention inside ``bboxes``.
    Returns the refined latents tensor.
    """

    loss = None
    for guidance_iter in range(max_guidance_iter_per_step):
        if loss is not None and loss.item() / scale_factor < 0.2:
            break
        latents = latents.clone().detach().requires_grad_(True)
        latent_input = model.scheduler.scale_model_input(latents, t)

        if num_channels_unet == 9:
            latent_input = torch.cat(
                [
                    latent_input,
                    mask[0].unsqueeze(0),
                    masked_image_latents[0].unsqueeze(0),
                ],
                dim=1,
            )

        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": time_ids}
        model.attention_store.clear()  # Clear stored attention maps before forward pass
        model.unet(
            latent_input,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
        )
        model.unet.zero_grad()

        loss = (
            compute_layout_guidance_loss(model, token_indices, bboxes, attn_mask)
            * scale_factor
        )
        grad_cond = torch.autograd.grad(
            loss,
            [latents],
            retain_graph=False,
        )[0]
        latents = latents - grad_cond * sigmas**2

    return latents

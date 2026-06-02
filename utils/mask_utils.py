from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_binary_mask(
    mask: "torch.Tensor", target_H: int, target_W: int
) -> "torch.Tensor":
    """Resize a binary mask using nearest-neighbor interpolation.

    Args:
        mask: Tensor of shape [H, W] with values in {0,1}.
        target_H: Target height.
        target_W: Target width.

    Returns:
        Resized mask tensor of shape [1,1,target_H,target_W].
    """
    mask = mask[None, None].float()  # [1,1,H,W]
    mask_resized = F.interpolate(mask, size=(target_H, target_W), mode="nearest")
    return mask_resized


def get_inpaint_mask_by_attn_maps(
    attn_maps: "torch.Tensor",
    threshold: Optional[float] = None,
    keep_ratio: float = 0.7,
    target_H: Optional[int] = None,
    target_W: Optional[int] = None,
    smoothed: bool = True,
    return_mask_only: bool = False,
) -> "torch.Tensor":
    """Generate an inpaint mask from attention maps.

    Args:
        attn_maps: 2D tensor of attention scores.
        threshold: Optional explicit threshold; if None, quantile is used.
        keep_ratio: Fraction of attention to keep when threshold is None.
        target_H, target_W: Optional resize targets for the attention map.
        smoothed: Whether to apply a small average pooling to smooth maps.
        return_mask_only: If True, return only the binary mask.

    Returns:
        If ``return_mask_only`` is True, returns a binary mask tensor.
        Otherwise returns a tuple (mask, attn_maps_rescaled).
    """

    if smoothed:
        attn_maps = F.avg_pool2d(attn_maps[None, None], 3, 1, 1)[0, 0]

    if target_H is not None and target_W is not None:
        attn_maps = (
            F.interpolate(
                attn_maps[None, None, :, :],
                size=(target_H, target_W),
                mode="bilinear",
            )
            .squeeze(0)
            .squeeze(0)
        )

    if threshold is None:
        # threshold at (1 - keep_ratio) quantile => top keep_ratio kept
        flat = attn_maps.flatten().float()
        q = 1.0 - keep_ratio
        threshold = torch.quantile(flat, q)

    def morphological_close(mask, kernel_size=3, iters=1):
        mask = mask[None, None].float()  # [1,1,H,W]
        for _ in range(iters):
            mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
            mask = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size // 2)
        return mask[0, 0]

    # Use >= to get closer to keep_ratio when there are ties
    attn_mask = (attn_maps >= threshold).float()
    attn_mask = morphological_close(attn_mask, kernel_size=3)

    if return_mask_only:
        return attn_mask

    return attn_mask, attn_maps

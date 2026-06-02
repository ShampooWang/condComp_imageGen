"""
source: https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/vis_utils.py
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


def show_cross_attention(
    prompt: str,
    tokenizer,
    indices_to_alter: List[int],
    res: int,
    select: int = 0,
    orig_image: Optional[Image.Image] = None,
    return_list: bool = False,
    global_normalized_heatmap: bool = False,
    from_where: List[str] = None,
    attention_store=None,
    attention_maps=None,
    average_for_all_tokens: bool = False,  # whether to average attention maps of all tokens or not
):
    """Visualize cross-attention maps for specified token indices.

    Args:
        prompt: Input prompt string.
        tokenizer: Tokenizer with ``encode`` and ``decode`` methods.
        indices_to_alter: Token indices to visualize.
        res: Resolution parameter for resizing visualizations.
        select: Unused selection index.
        orig_image: Optional PIL image to overlay heatmaps onto.
        return_list: If True, return a list of PIL images instead of a grid.
        global_normalized_heatmap: If True, normalize heatmaps globally.
        from_where: List of attention locations to aggregate (e.g. ["mid", "up"]).
        attention_store: Optional AttentionStore to pull maps from.
        attention_maps: Optional precomputed attention maps tensor.
        average_for_all_tokens: If True, average maps over tokens.

    Returns:
        Either a list of PIL images (if ``return_list`` True) or a single
        PIL image grid.
    """

    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    if attention_maps is None:
        assert attention_store is not None, "Attention store must be provided."
        attention_maps = attention_store.aggregate(from_where).detach().cpu()

    global_attn_max = attention_maps.max() if global_normalized_heatmap else None
    global_attn_min = attention_maps.min() if global_normalized_heatmap else None

    # show spatial attention for indices of tokens to strengthen
    images = []
    if average_for_all_tokens:
        image = attention_maps[:, :, indices_to_alter].mean(dim=-1)
        image = show_image_relevance(
            image,
            orig_image,
            global_max=global_attn_max,
            global_min=global_attn_min,
        )
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res**2, res**2)))
        images.append(Image.fromarray(image))
    else:
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            if i in indices_to_alter:
                image = show_image_relevance(
                    image,
                    orig_image,
                    global_max=global_attn_max,
                    global_min=global_attn_min,
                )
                image = image.astype(np.uint8)
                image = np.array(Image.fromarray(image).resize((res**2, res**2)))
                if return_list:
                    images.append(Image.fromarray(image))
                else:
                    image = text_under_image(image, decoder(int(tokens[i])))
                    images.append(image)

    if return_list:
        return images
    else:
        return view_images(np.stack(images, axis=0))


def view_images(
    images: Union[np.ndarray, List],
    num_rows: int = 1,
    offset_ratio: float = 0.02,
    display_image: bool = True,
) -> Image.Image:
    """Display a list or array of images in a grid and return a PIL image.

    Args:
        images: Either a numpy array of shape [N,H,W,C] or a Python list of
            image arrays.
        num_rows: Number of rows in the grid.
        offset_ratio: Spacing between images as fraction of image height.
        display_image: Unused; kept for API compatibility.

    Returns:
        A PIL Image containing the image grid.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)

    return pil_img


def text_under_image(
    image: np.ndarray,
    text: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 2.0,  # make text larger
    space_ratio: float = 0.1,  # space between image & text (smaller)
) -> np.ndarray:
    h, w, c = image.shape

    # smaller space vs. image height
    offset = int(h * space_ratio)

    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    img[:h] = image

    font = cv2.FONT_HERSHEY_SIMPLEX

    # recompute size using larger font_scale
    textsize = cv2.getTextSize(text, font, font_scale, thickness=int(font_scale * 1.5))[
        0
    ]

    # center horizontally
    text_x = (w - textsize[0]) // 2

    # place text close to image bottom
    text_y = h + offset - (textsize[1] // 4)

    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness=2)
    return img


def show_image_relevance(
    image_relevance: Union[torch.Tensor, np.ndarray],
    image: Image.Image,
    relevnace_res: int = 16,
    global_max: Optional[float] = None,
    global_min: Optional[float] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Render an attention relevance map over an image and return a BGR array.

    Args:
        image_relevance: 1D/2D relevance tensor or array.
        image: PIL Image to overlay.
        relevnace_res: Resolution of relevance map (square grid length).
        global_max/global_min: Optional global normalization bounds.
        eps: Small value to avoid division by zero.

    Returns:
        A NumPy BGR image (uint8) with the heatmap overlay.
    """

    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res**2, relevnace_res**2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(
        1, 1, image_relevance.shape[-1], image_relevance.shape[-1]
    )
    image_relevance = (
        image_relevance.cuda()
    )  # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(
        image_relevance, size=relevnace_res**2, mode="bilinear"
    )
    image_relevance = image_relevance.cpu()  # send it back to cpu

    if global_max is not None and global_min is not None:
        image_relevance = (image_relevance - global_min) / (global_max - global_min)
    else:
        if image_relevance.max() - image_relevance.min() > eps:
            diff = image_relevance.max() - image_relevance.min()
        else:
            diff = eps
        image_relevance = (image_relevance - image_relevance.min()) / diff

    image_relevance = image_relevance.reshape(relevnace_res**2, relevnace_res**2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def pil_list_to_grid(pil_images: List[Image.Image], nrow: int = 3) -> Image.Image:
    """Convert a list of PIL images into a single grid PIL image.

    Args:
        pil_images: List of PIL images.
        nrow: Number of images per row in the grid.

    Returns:
        A PIL Image with the image grid.
    """
    # Convert PIL → tensor and stack
    to_tensor = transforms.ToTensor()
    tensors = [to_tensor(img) for img in pil_images]  # each: [C,H,W]
    batch = torch.stack(tensors, dim=0)  # [B,C,H,W]

    # Make grid
    grid = make_grid(batch, nrow=nrow)  # [C, H_grid, W_grid]

    # Convert back to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(grid)

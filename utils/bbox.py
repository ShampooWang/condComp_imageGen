import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def iou(box1: list, box2: list) -> float:
    """Compute Intersection-over-Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def random_non_overlapping_boxes(
    width: int,
    height: int,
    n: int,
    min_size: int = 50,
    max_size: int = 200,
    iou_threshold: float = 0.0,
    max_tries: int = 2000,
    normalize: bool = False,
) -> list:
    """Sample up to ``n`` axis-aligned non-overlapping boxes.

    Boxes are returned as lists [x1, y1, x2, y2]. If ``normalize`` is
    True coordinates are in [0,1]. The function may return fewer than
    ``n`` boxes if sampling fails after ``max_tries`` attempts.
    """
    boxes = []

    tries = 0
    while len(boxes) < n and tries < max_tries:
        tries += 1

        # Random width/height
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)

        # Random location
        x1 = random.randint(0, width - w)
        y1 = random.randint(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        new_box = [x1, y1, x2, y2]

        # Check overlap
        success_create = True
        for b in boxes:
            if iou(new_box, b) > iou_threshold:  # e.g., > 0 (strict)
                success_create = False
                break

        if success_create:
            boxes.append(new_box)

    if len(boxes) < n:
        print(
            f"Warning: Only generated {len(boxes)} non-overlapping boxes (requested {n})."
        )

    if normalize:
        boxes = [
            [x1 / width, y1 / height, x2 / width, y2 / height]
            for x1, y1, x2, y2 in boxes
        ]

    return boxes


def draw_boxes(boxes: list, img_size: int = 512, save_path: str = "bboxes.png") -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.set_xlim(0, img_size)
    ax.set_ylim(0, img_size)
    ax.invert_yaxis()  # Match typical image coordinate system

    colors = plt.cm.tab20.colors  # different colors

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"Box {i}", color=colors[i % len(colors)], fontsize=9)

    plt.title(f"Sampled Bounding Boxes ({len(boxes)})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[Saved] {save_path}")
    plt.close()


def draw_bboxes_on_image(
    image_path: str,
    labels: list,
    bboxes: list,
    font_size: int = 16,
    font_type: str = "Hack-Regular.ttf",
    img=None,
) -> Image.Image:
    """Draw labeled bounding boxes on an image and return a PIL image.

    Args:
        image_path: Path to the source image (used if ``img`` is None).
        labels: Sequence of label strings corresponding to ``bboxes``.
        bboxes: Sequence of bounding boxes in normalized coordinates
            (x_min, y_min, x_max, y_max) where values are in [0,1].
        font_size: Font size for labels.
        font_type: Font filename to use; falls back to default font if not found.
        img: Optional PIL image to draw on; if provided, ``image_path`` is ignored.

    Returns:
        A PIL Image with boxes and labels drawn.
    """
    if img is None:
        img = Image.open(image_path).convert("RGB")
    else:
        img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Load font
    try:
        font = ImageFont.truetype(font_type, size=font_size)
    except:
        print(f"{font_type} font not found. Using default font.")
        font = ImageFont.load_default()

    # 1. Group labels by (approx) identical bounding boxes
    #    Use rounded coordinates as a key to avoid tiny float noise
    grouped = {}
    for bbox, label in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = bbox
        key = tuple(round(v, 4) for v in (x_min, y_min, x_max, y_max))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(label)

    # 2. Draw each unique box with combined labels
    for (x_min, y_min, x_max, y_max), lab_list in grouped.items():
        # Combine labels for the same box
        label = ", ".join(lab_list)

        # Convert normalized coords → pixels
        x_min_px = int(x_min * W)
        y_min_px = int(y_min * H)
        x_max_px = int(x_max * W)
        y_max_px = int(y_max * H)

        # Draw rectangle
        draw.rectangle(
            [(x_min_px, y_min_px), (x_max_px, y_max_px)],
            outline="red",
            width=4,
        )

        # Compute text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Position label above the box
        text_x = x_min_px
        text_y = y_min_px - text_h
        if text_y < 0:
            text_y = 0  # clamp to top

        # Background for text
        draw.rectangle(
            [text_x, text_y, text_x + text_w, text_y + text_h],
            fill="red",
        )

        # Draw label text
        draw.text((text_x, text_y), label, fill="white", font=font)

    return img


if __name__ == "__main__":
    # Test
    boxes = random_non_overlapping_boxes(512, 512, 5, min_size=100, max_size=300)
    draw_boxes(boxes, img_size=512, save_path="bboxes.png")

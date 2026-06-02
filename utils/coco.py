import random
from collections import defaultdict

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


def get_coco_annotations(ann_file="instances_val2014.json"):
    coco = COCO(ann_file)

    img_dict = {}

    # optional: cache category id -> name
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}

    for img in coco.dataset["images"]:
        img_id = img["id"]
        h, w = img["height"], img["width"]
        img_area = h * w

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        categories = set()
        is_crowd_list = []
        rles = []

        for ann in anns:
            categories.add(cat_id_to_name[ann["category_id"]])
            is_crowd_list.append(ann["iscrowd"])

            # always convert annotation to a proper RLE
            rle = coco.annToRLE(ann)
            rles.append(rle)

        if len(rles) == 0:
            fg_ratio = 0.0
        else:
            merged = maskUtils.merge(rles)
            fg_area = float(maskUtils.area(merged))
            fg_ratio = fg_area / img_area

        img_dict[img_id] = {
            "categories": categories,
            "is_crowd": max(is_crowd_list) if is_crowd_list else False,
            "fg_area": fg_ratio,
        }

    return img_dict


def oxford_join(items, article="a"):
    """
    Join items with an Oxford comma and add an article before each item.

    Examples:
        oxford_join(["person", "tie", "tv"]) -> "a person, a tie, and a tv"
        oxford_join(["dog", "cat"], article="the") -> "the dog and the cat"
    """
    if isinstance(items, str):
        items = [items]
    items = [str(i).strip() for i in items if i]
    if not items:
        return ""
    # prefix article
    items = [f"{article} {item}" for item in items]

    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def load_coco_filtered(
    instances_json: str,
    total_images: int,
    *,
    class_count: int = None,
    one_caption_per_image: str = "random",  # "first" or "random"
    shuffle: bool = True,
    use_coco_caption: bool = False,
    compose_class: bool = False,
    seed: int = 322,
):
    """
    Returns a list of dicts:
      { "image_id": int, "caption": str, "categories": [names] }
    filtered to images whose class count matches `class_count`.
    """

    coco_det = COCO(instances_json)

    # index annotations per image
    img_to_anns = defaultdict(list)
    for ann in coco_det.dataset["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    if shuffle:
        # deterministic shuffle with a fixed seed
        rng = random.Random(seed)
        img_ids = list(img_to_anns.keys())
        rng.shuffle(img_ids)
        img_to_anns = {img_id: img_to_anns[img_id] for img_id in img_ids}

    out = []
    for img_id, anns in img_to_anns.items():
        cat_ids = sorted({a["category_id"] for a in anns})

        if len(cat_ids) == 0:
            continue
        if class_count is not None and len(cat_ids) != class_count:
            continue

        names = [coco_det.loadCats(cid)[0]["name"] for cid in sorted({*cat_ids})]

        if compose_class:
            # build caption for composable sdxl
            caption = " | ".join(
                [
                    f"A photo with realistic lighting, showing a {name}."
                    for name in names
                ]
            )
        else:
            # build caption from categories
            caption = (
                "A photo with realistic lighting, showing " + oxford_join(names) + "."
            )

        out.append({"image_id": img_id, "caption": caption, "categories": names})

    if total_images is not None and total_images > 0:
        print(len(out), "images found, truncating to", total_images)
        out = out[:total_images]

    return out

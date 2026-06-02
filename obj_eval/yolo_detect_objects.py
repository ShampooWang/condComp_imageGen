import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ultralytics import YOLO


def yolo_eval(
    image_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    image_dir: Optional[Union[str, Path]] = None,
    target_img_name: Optional[str] = None,
    categories: List[str] = ["person", "apple", "dog", "cat"],
    print_results: bool = False,
    model: Optional[Any] = None,
    yolo_ckpt: str = "./ckpts/yolo/yolo11l.pt",
    return_individual_results: bool = False,
) -> Union[float, List[bool]]:
    """Evaluate presence of specified categories in images using YOLO.

    Args:
        image_path: Single path or list of image paths to evaluate. If None,
            images are collected from ``image_dir`` matching ``target_img_name``.
        image_dir: Directory to search for images if ``image_path`` is None.
        target_img_name: Glob or filename to match when searching ``image_dir``.
        categories: List of category names to check for presence.
        print_results: If True, print summary statistics.
        model: Optional pre-loaded YOLO model instance; if None a model is
            loaded from ``yolo_ckpt``.
        yolo_ckpt: Path to YOLO checkpoint used when ``model`` is None.
        return_individual_results: If True, returns per-image boolean results.

    Returns:
        If ``return_individual_results`` is True, returns a list of booleans
        indicating whether each image contains all requested categories.
        Otherwise returns the percentage of images containing all categories.
    """

    if model is None:
        from ultralytics import YOLO

        model = YOLO(yolo_ckpt)

    if image_path is None:  # Get paths of images from image_dir with name image_name
        image_path: list = list(Path(image_dir).rglob(target_img_name))

    if not isinstance(categories, list):
        categories = [categories]
    categories: list = categories

    if not isinstance(image_path, list):
        image_path = [image_path]
    image_path: list = image_path

    sucess_list: list[bool] = []
    sucess_num: int = 0
    for p in image_path:
        r = model(p, conf=0.25, iou=0.6, imgsz=640, verbose=False)[0]
        det_names: set = set([r.names[int(c)] for c in r.boxes.cls.tolist()])
        suc: bool = set(categories).issubset(det_names)
        sucess_list.append(suc)
        if suc:
            sucess_num += 1

    if print_results:
        print(f"# of images to detect: {len(image_path)}")
        print(f"Sucess rate: {sucess_num * 100 / len(image_path):.2f}")

    if return_individual_results:
        return sucess_list
    return sucess_num * 100 / len(image_path)


def yolo_eval_stagewise_results(
    init_img_path: Optional[Union[str, Path]],
    img_path: Union[str, Path],
    categories_list: List[Union[str, List[str]]],
    yolo_model: Optional[Any] = None,
    yolo_ckpt: str = "./ckpts/yolo/yolo11l.pt",
) -> Tuple[Optional[bool], Optional[bool], bool]:
    """Compute stage-wise YOLO detection results for an initial and final image.

    The function checks whether the final image contains all categories and
    whether an additional target category appears between the initial and
    final images.

    Returns a tuple (base_result, add_one_result, full_result) where each is
    a boolean or None.
    """
    if yolo_model is None:
        from ultralytics import YOLO

        yolo_model = YOLO(yolo_ckpt)

    all_cateogories: list = []
    for _cat in categories_list:
        if isinstance(_cat, list):
            all_cateogories.extend(_cat)
        else:
            all_cateogories.append(_cat)

    full_result = yolo_eval([img_path], categories=all_cateogories, model=yolo_model)
    full_result: bool = full_result >= 100

    if init_img_path is None:
        return None, None, full_result
    else:
        base_result = yolo_eval(
            [init_img_path],
            categories=categories_list[0],
            model=yolo_model,
        )
        add_one_base = yolo_eval(
            image_path=[init_img_path],
            categories=categories_list[-1],
            model=yolo_model,
        )
        add_one_final = yolo_eval(
            image_path=[img_path],
            categories=categories_list[-1],
            model=yolo_model,
        )

        base_result: bool = base_result >= 100
        add_one_result: bool = (add_one_final >= 100) and (add_one_base < 100)

        return base_result, add_one_result, full_result


def eval_coco_results(
    img_dir: Union[str, Path],
    coco_instances_json: Union[str, Path],
    out_json_path: Union[str, Path],
    args: Any,
) -> None:
    """Run YOLO on images and compare detected classes against COCO annotations.

    Args:
        img_dir: Directory with PNG images named like "...id<image_id>.png".
        coco_instances_json: Path to COCO annotations JSON.
        out_json_path: Output path to write per-image detection results JSON.
        args: Namespace with additional runtime args (used for locating caption file).

    Side effects:
        Prints success_rate statistics and writes ``out_json_path``.
    """
    from pycocotools.coco import COCO

    # Load COCO ground truth
    coco = COCO(coco_instances_json)

    # Load a pretrained YOLO model (COCO-trained)
    model = YOLO(args.yolo_ckpt)

    img_paths: list = sorted(Path(img_dir).glob("*.png"))
    per_image_presence: list[dict] = []
    for path in img_paths:
        res = model(path, conf=0.25, iou=0.6, imgsz=640, verbose=False)[0]
        # Extract numeric ID (e.g. "12345" from "12345.png" or "sdxl_id12345.png")
        image_id: int = int(path.stem.split("id")[-1])

        # Expected classes from COCO
        ann_ids: list = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        cat_ids: list = [a["category_id"] for a in anns]
        expected: set = {coco.loadCats(cid)[0]["name"] for cid in cat_ids}

        # Detected classes from YOLO
        det_names: set = set([res.names[int(c)] for c in res.boxes.cls.tolist()])
        intersects: set = expected & det_names
        unexpected: set = det_names - expected
        missing: set = expected - det_names
        delta: int = len(intersects) - len(expected)

        per_image_presence.append(
            {
                "image_id": image_id,
                "file": str(path),
                "expected": sorted(expected),
                "present": sorted(det_names),
                "missing": sorted(missing),
                "unexpected": sorted(unexpected),
                "delta": delta,
                "num_detections": len(det_names),
            }
        )

    # Aggregate success_rate
    num_images: int = len(per_image_presence)
    num_complete: int = sum(1 for r in per_image_presence if len(r["missing"]) == 0)
    success_rate: float = (num_complete / num_images * 100) if num_images > 0 else 0.0

    print(
        f"success_rate: {success_rate:.2f}% ({num_complete}/{num_images} images have all expected objects)"
    )

    with open(out_json_path, "w") as f:
        json.dump(per_image_presence, f, indent=4)

    if os.path.exists(os.path.join(args.img_dir, "coco_captions.json")):
        coco_capts: list[dict] = json.load(
            open(os.path.join(args.img_dir, "coco_captions.json"), "r")
        )
        if "target_tokens" in coco_capts[0]:
            imgid2miss: dict = {r["image_id"]: r["missing"] for r in per_image_presence}
            completness_of_full_target: int = 0
            total: int = 0
            for capt in coco_capts:
                img_id: int = capt["image_id"]
                miss = imgid2miss[img_id]
                if len(capt["target_tokens"]) < len(capt["categories"]):
                    total += 1
                    if len(miss) == 0:
                        completness_of_full_target += 1
            if total > 0:
                print(
                    f"success_rate of full target: {(completness_of_full_target/total*100):.2f}%. Total: {total}"
                )
            else:
                print("No captions with missing target tokens found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect objects in images using YOLO and compare against expected captions"
    )
    parser.add_argument(
        "--yolo_ckpt",
        default="./ckpts/yolo/yolo11l.pt",
        help="Path to YOLO checkpoint",
    )
    parser.add_argument(
        "--img_dir",
        required=True,
        help="Use other expected captions instead of COCO",
    )
    parser.add_argument(
        "--coco_instances_json",
        type=str,
        default="./instances_val2014.json",
        help="Path to COCO annotations JSON file",
    )
    parser.add_argument(
        "--out_json_path",
        type=str,
        default=None,
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    if args.out_json_path is None:
        args.out_json_path = os.path.join(args.img_dir, "eval_results.json")

    eval_coco_results(
        img_dir=args.img_dir,
        coco_instances_json=args.coco_instances_json,
        out_json_path=args.out_json_path,
        args=args,
    )

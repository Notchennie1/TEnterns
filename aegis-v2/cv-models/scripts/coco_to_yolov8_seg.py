"""
COCO Segmentation -> YOLOv8 Segmentation Format Converter
==========================================================
Converts COCO-format annotations (from Roboflow) into the YOLOv8
segmentation format required for training.

COCO format:
  - annotations.json with "images", "annotations", "categories"
  - Each annotation has "segmentation" as list of polygon point arrays

YOLOv8-seg format:
  - One .txt label file per image
  - Each line: <class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
  - All coordinates normalized to [0, 1]

Usage:
    python coco_to_yolov8_seg.py \
        --coco-json path/to/annotations.json \
        --images-dir path/to/images \
        --output-dir path/to/yolov8_dataset \
        --split-ratio 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_coco_json(json_path: str) -> dict[str, Any]:
    """Load and validate a COCO-format annotation file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    required_keys = {"images", "annotations", "categories"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"COCO JSON missing required keys: {missing}")

    logger.info(
        "Loaded COCO JSON: %d images, %d annotations, %d categories",
        len(data["images"]),
        len(data["annotations"]),
        len(data["categories"]),
    )
    return data


def build_category_map(categories: list[dict]) -> dict[int, int]:
    """Map COCO category IDs to contiguous 0-indexed class IDs."""
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    mapping = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    for cat in sorted_cats:
        logger.info("  Category %d -> class %d: %s", cat["id"], mapping[cat["id"]], cat["name"])
    return mapping


def coco_seg_to_yolo_line(
    annotation: dict,
    img_width: int,
    img_height: int,
    cat_map: dict[int, int],
) -> str | None:
    """
    Convert one COCO annotation into a YOLOv8-seg label line.

    Returns None if the annotation has no valid segmentation.
    """
    seg = annotation.get("segmentation")
    if not seg or not isinstance(seg, list):
        return None

    # COCO segmentation can be RLE or polygon — we only handle polygon
    if isinstance(seg[0], dict):
        logger.debug("Skipping RLE segmentation (annotation %d)", annotation["id"])
        return None

    class_id = cat_map.get(annotation["category_id"])
    if class_id is None:
        return None

    # Flatten all polygon parts and normalize
    parts: list[str] = [str(class_id)]
    for polygon in seg:
        # polygon is [x1, y1, x2, y2, ...]
        if len(polygon) < 6:  # Need at least 3 points
            continue
        for i in range(0, len(polygon), 2):
            x_norm = polygon[i] / img_width
            y_norm = polygon[i + 1] / img_height
            # Clamp to [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            parts.append(f"{x_norm:.6f}")
            parts.append(f"{y_norm:.6f}")

    if len(parts) < 4:  # class_id + at least 3 points (6 values)
        return None

    return " ".join(parts)


def convert_coco_to_yolov8_seg(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """
    Full conversion pipeline: COCO JSON -> YOLOv8-seg dataset.

    Creates:
        output_dir/
          images/
            train/
            val/
          labels/
            train/
            val/
          data.yaml
    """
    random.seed(seed)

    coco = load_coco_json(coco_json_path)
    cat_map = build_category_map(coco["categories"])

    # Build image lookup
    img_lookup: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Prepare output directories
    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    image_ids = list(img_lookup.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * split_ratio)
    splits = {
        "train": image_ids[:split_idx],
        "val": image_ids[split_idx:],
    }

    stats = {"train": 0, "val": 0, "skipped": 0}
    images_dir_path = Path(images_dir)

    for split_name, ids in splits.items():
        for img_id in ids:
            img_info = img_lookup[img_id]
            img_w = img_info["width"]
            img_h = img_info["height"]
            img_filename = img_info["file_name"]

            # Find source image
            src_img = images_dir_path / img_filename
            if not src_img.exists():
                # Try without subdirectory prefix
                src_img = images_dir_path / Path(img_filename).name
            if not src_img.exists():
                logger.warning("Image not found: %s — skipping", img_filename)
                stats["skipped"] += 1
                continue

            # Convert annotations to YOLO lines
            annotations = anns_by_image.get(img_id, [])
            lines: list[str] = []
            for ann in annotations:
                line = coco_seg_to_yolo_line(ann, img_w, img_h, cat_map)
                if line:
                    lines.append(line)

            if not lines:
                stats["skipped"] += 1
                continue

            # Copy image
            dst_img = out / "images" / split_name / Path(img_filename).name
            shutil.copy2(src_img, dst_img)

            # Write label file
            label_name = Path(img_filename).stem + ".txt"
            label_path = out / "labels" / split_name / label_name
            label_path.write_text("\n".join(lines) + "\n")

            stats[split_name] += 1

    # Write data.yaml
    category_names = [
        cat["name"]
        for cat in sorted(coco["categories"], key=lambda c: c["id"])
    ]
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"# YOLOv8 Segmentation Dataset\n"
        f"# Auto-generated from COCO annotations\n\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(category_names)}\n"
        f"names: {category_names}\n"
    )

    logger.info("Conversion complete!")
    logger.info("  Train: %d images", stats["train"])
    logger.info("  Val:   %d images", stats["val"])
    logger.info("  Skipped: %d images", stats["skipped"])
    logger.info("  data.yaml: %s", data_yaml)


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO segmentation annotations to YOLOv8-seg format"
    )
    parser.add_argument(
        "--coco-json", required=True,
        help="Path to COCO annotations JSON file (from Roboflow export)",
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Directory containing the source images",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for the YOLOv8 dataset",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits",
    )
    args = parser.parse_args()

    convert_coco_to_yolov8_seg(
        coco_json_path=args.coco_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert LabelMe annotations into crops + labels.csv alongside existing data.

It scans all LabelMe JSON files under data/annotation/, finds matching images
in data/image/, crops each polygon/rectangle, and appends entries to
data/dataset/labels.csv while saving crops to data/dataset/crops/.
"""

import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


def iter_annotation_files(root: Path) -> Iterable[Path]:
    """Yield all .json files under the given directory (recursively)."""
    return root.rglob("*.json")


def normalize_polygon(shape: dict) -> List[Tuple[float, float]]:
    """
    Return a list of (x, y) points for the shape.
    Handles polygons and rectangles; ignores unsupported shapes.
    """
    points = shape.get("points", []) or []
    shape_type = shape.get("shape_type", "polygon")

    if shape_type == "rectangle" and len(points) == 2:
        (x1, y1), (x2, y2) = points
        return [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ]

    # Treat everything else as polygon
    return [(float(x), float(y)) for x, y in points]


def normalize_text(text: str) -> str:
    """Apply basic normalization for Khmer labels."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width and similar artifacts
    for ch in ["\u200b", "\u200c", "\u200d", "\ufeff"]:
        text = text.replace(ch, "")
    return text.strip()


def crop_and_mask(
    image: Image.Image, polygon: List[Tuple[float, float]]
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Apply a polygon mask to the image, returning the masked crop and the crop box.
    Background outside the polygon is white.
    """
    if len(polygon) < 3:
        raise ValueError("Polygon needs at least 3 points")

    img_w, img_h = image.size
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x = max(0, int(np.floor(min(xs))))
    max_x = min(img_w, int(np.ceil(max(xs))))
    min_y = max(0, int(np.floor(min(ys))))
    max_y = min(img_h, int(np.ceil(max(ys))))

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid crop box")

    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, outline=255, fill=255)

    crop_box = (min_x, min_y, max_x, max_y)
    cropped_image_raw = image.crop(crop_box)
    cropped_mask = mask.crop(crop_box)

    final_image = Image.new("RGB", cropped_image_raw.size, (255, 255, 255))
    final_image.paste(cropped_image_raw, (0, 0), mask=cropped_mask)
    return final_image, crop_box


def process_labelme_file(
    json_path: Path, image_dir: Path, crops_dir: Path
) -> List[Tuple[str, str]]:
    """
    Process a single LabelMe JSON; returns list of (filename, text) label pairs.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    raw_image_path = str(data.get("imagePath", "")).replace("\\", "/")
    image_name = Path(raw_image_path).name
    if not image_name:
        raise ValueError(f"Missing imagePath in {json_path}")

    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for {json_path}: {image_path}")

    image = Image.open(image_path).convert("RGB")

    rows: List[Tuple[str, str]] = []
    for idx, shape in enumerate(data.get("shapes", [])):
        raw_text = shape.get("label") or ""
        text = normalize_text(raw_text)
        if not text:
            continue

        polygon = normalize_polygon(shape)
        try:
            crop_img, _ = crop_and_mask(image, polygon)
        except Exception as exc:  # skip bad shapes, but continue
            print(f"Warning: skipping shape {idx} in {json_path} ({exc})")
            continue

        base = image_path.stem
        crop_name = f"{base}_{idx:04d}.jpg"

        # Avoid collisions by nudging the suffix
        collision_idx = 1
        while (crops_dir / crop_name).exists():
            crop_name = f"{base}_{idx:04d}_{collision_idx}.jpg"
            collision_idx += 1

        crop_img.save(crops_dir / crop_name)
        rows.append((crop_name, text))

    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_ann_dir = repo_root / "data" / "annotation"
    default_image_dir = repo_root / "data" / "image"
    default_output_dir = repo_root / "data" / "dataset"

    parser = argparse.ArgumentParser(
        description="Convert LabelMe annotations into crops and labels.csv"
    )
    parser.add_argument(
        "-a",
        "--annotation-dir",
        type=Path,
        default=default_ann_dir,
        help="Directory containing LabelMe JSON files.",
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for crops/ and labels.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite labels.csv instead of appending if it exists.",
    )

    args = parser.parse_args()

    ann_dir = args.annotation_dir
    image_dir = args.image_dir
    output_dir = args.output_dir
    crops_dir = output_dir / "crops"
    labels_path = output_dir / "labels.csv"

    crops_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Tuple[str, str]] = []
    for json_file in iter_annotation_files(ann_dir):
        # Skip the Label Studio export if present
        if json_file.name == "annotated_data.json":
            continue
        try:
            rows = process_labelme_file(json_file, image_dir=image_dir, crops_dir=crops_dir)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"Warning: skipping {json_file} ({exc})")

    if not all_rows:
        print("No annotations processed.")
        return

    mode = "w" if args.overwrite or not labels_path.exists() else "a"
    write_header = mode == "w" or labels_path.stat().st_size == 0

    with labels_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["file_name", "text"])
        writer.writerows(all_rows)

    print(
        f"Processed {len(all_rows)} labeled regions from LabelMe files. "
        f"Crops saved to {crops_dir}, labels written to {labels_path} ({'overwrite' if mode=='w' else 'append'})."
    )


if __name__ == "__main__":
    main()

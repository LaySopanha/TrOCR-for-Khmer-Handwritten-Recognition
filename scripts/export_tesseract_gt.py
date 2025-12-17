#!/usr/bin/env python3
"""
Export Tesseract ground truth pairs from train.csv.

For each entry in data/dataset/train.csv, copy the image from data/dataset/crops
to a target folder and write a matching .gt.txt with the label.

Example:
python scripts/export_tesseract_gt.py \
  --csv data/dataset/train.csv \
  --root data/dataset/crops \
  --output data/tess/train
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tesseract GT files from train.csv.")
    parser.add_argument("--csv", type=Path, default=Path("data/dataset/train.csv"))
    parser.add_argument("--root", type=Path, default=Path("data/dataset/crops"))
    parser.add_argument("--output", type=Path, default=Path("data/tess/train"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["text"].notna()]

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for _, row in df.iterrows():
        fn = row["file_name"]
        label = str(row["text"])
        src_img = args.root / fn
        if not src_img.exists():
            print(f"[skip] missing image: {src_img}")
            skipped += 1
            continue

        dst_img = out_dir / fn
        dst_img.write_bytes(src_img.read_bytes())
        (out_dir / f"{dst_img.stem}.gt.txt").write_text(label, encoding="utf-8")
        copied += 1

    print(f"Exported {copied} pairs to {out_dir} (skipped {skipped} missing images).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark Tesseract OCR on the held-out test set and report CER.

Requirements:
  - Tesseract binary installed with Khmer traineddata (lang code: khm)
    Ubuntu/Debian example: sudo apt-get install tesseract-ocr tesseract-ocr-khm
  - Python packages: pytesseract, evaluate, pillow
    pip install pytesseract evaluate pillow

Example:
python scripts/benchmark_tesseract.py \
  --csv data/dataset/test.csv \
  --root data/dataset/crops \
  --lang khm \
  --psm 6 \
  --max-samples 0
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import evaluate
import pytesseract
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Tesseract OCR on cropped dataset.")
    parser.add_argument("--csv", type=Path, default=Path("data/dataset/test.csv"), help="CSV with file_name,text.")
    parser.add_argument("--root", type=Path, default=Path("data/dataset/crops"), help="Directory containing crop images.")
    parser.add_argument("--lang", type=str, default="khm", help="Tesseract language code (e.g., khm, khm+eng).")
    parser.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode.")
    parser.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract binary if not in PATH.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 for all).")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the CSV.")
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("file_name")
            text = row.get("text", "")
            if not fn or text is None:
                continue
            rows.append((fn, text.strip()))
    return rows


def main() -> None:
    args = parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    rows = load_rows(args.csv)
    if args.offset:
        rows = rows[args.offset :]
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]

    cer_metric = evaluate.load("cer")
    preds, refs = [], []

    config = f"--psm {args.psm}"

    for idx, (fn, label) in enumerate(rows, start=1):
        img_path = args.root / fn
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            pred = pytesseract.image_to_string(image, lang=args.lang, config=config)
            pred = pred.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {fn}: {exc}")
            pred = ""

        preds.append(pred)
        refs.append(label)
        print(f"[{idx}/{len(rows)}] {fn} -> {pred[:50]!r}")

    if not preds:
        print("No predictions collected.")
        return

    cer = cer_metric.compute(predictions=preds, references=refs)
    print(f"CER on {len(preds)} samples: {cer:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark Gemini OCR on the cropped dataset and report CER.

Requirements:
  pip install google-generativeai
Environment:
  export GEMINI_API_KEY=...

Example:
python scripts/benchmark_gemini.py \
  --model gemini-1.5-flash \
  --csv data/dataset/labels.csv \
  --root data/dataset/crops \
  --max-samples 200
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Tuple

import evaluate
import google.generativeai as genai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemini OCR on cropped dataset.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name.")
    parser.add_argument("--csv", type=Path, default=Path("data/dataset/test.csv"), help="CSV with file_name,text.")
    parser.add_argument("--root", type=Path, default=Path("data/dataset/crops"), help="Directory containing crop images.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 for all).")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the CSV.")
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to sleep between calls (for rate limiting).")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key (defaults to GEMINI_API_KEY env).")
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
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set (pass --api-key or set env).")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    rows = load_rows(args.csv)
    if args.offset:
        rows = rows[args.offset :]
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]

    cer_metric = evaluate.load("cer")
    preds, refs = [], []

    for idx, (fn, label) in enumerate(rows, start=1):
        img_path = args.root / fn
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue

        with img_path.open("rb") as f:
            img_bytes = f.read()

        prompt = "Read the Khmer text in this image. Respond with only the raw text."
        pred = ""
        for attempt in range(2):  # simple retry on quota
            try:
                resp = model.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
                )
                pred = (resp.text or "").strip()
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "quota" in msg.lower() or "rate" in msg.lower():
                    sleep_for = max(args.sleep, 10.0)
                    print(f"[quota] {fn}: {exc}. Sleeping {sleep_for}s and retrying...")
                    time.sleep(sleep_for)
                    continue
                print(f"[error] {fn}: {exc}")
                break

        preds.append(pred)
        refs.append(label)
        print(f"[{idx}/{len(rows)}] {fn} -> {pred[:50]!r}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    if not preds:
        print("No predictions collected.")
        return

    cer = cer_metric.compute(predictions=preds, references=refs)
    print(f"CER on {len(preds)} samples: {cer:.4f}")


if __name__ == "__main__":
    main()

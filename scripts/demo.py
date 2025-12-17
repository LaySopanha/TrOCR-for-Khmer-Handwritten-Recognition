#!/usr/bin/env python3
"""
Quick demo script to run inference with the Khmer TrOCR model.

Defaults:
- Model weights: outputs/checkpoint-1900
- Processor/tokenizer: khmer_trocr_model (needed because checkpoints do not carry the tokenizer)

Examples:
- Single image: python scripts/demo.py --inputs data/dataset/crops/example.jpg
- Folder: python scripts/demo.py --inputs data/dataset/crops --max-samples 5
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "outputs" / "checkpoint-1900"
DEFAULT_PROCESSOR_DIR = PROJECT_ROOT / "khmer_trocr_model"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick OCR demo with the trained Khmer TrOCR model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to model weights (checkpoint folder). Defaults to outputs/checkpoint-1900.",
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=DEFAULT_PROCESSOR_DIR,
        help="Path to processor/feature extractor (required for checkpoints). Defaults to khmer_trocr_model.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Optional tokenizer path. Defaults to processor-dir if not set.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Image files or directories containing images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of images to process (0 = no limit).",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional path to save predictions as CSV (columns: file_name,prediction).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu or cuda). Defaults to auto-detect.",
    )
    return parser.parse_args()


def collect_image_paths(paths, max_samples=0):
    images = []
    for p in paths:
        if p.is_dir():
            for ext in ALLOWED_EXTS:
                images.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            images.append(p)
        else:
            print(f"Skipping non-image input: {p}", file=sys.stderr)

    images = sorted(images)
    if max_samples and len(images) > max_samples:
        images = images[:max_samples]
    return images


def load_model_and_processor(model_dir: Path, processor_dir: Path, tokenizer_dir: Path | None, device: torch.device):
    tokenizer_path = tokenizer_dir or processor_dir
    processor = TrOCRProcessor.from_pretrained(processor_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor.tokenizer = tokenizer

    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, processor


def run_inference(model, processor, images, device):
    predictions = []
    with torch.inference_mode():
        for img_path in images:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:  # pragma: no cover - safety for bad files
                print(f"Failed to load {img_path}: {e}", file=sys.stderr)
                continue

            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append((img_path, text))
            print(f"{img_path}: {text}")
    return predictions


def maybe_save_csv(predictions, csv_path: Path):
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "prediction"])
        for img_path, text in predictions:
            writer.writerow([str(img_path), text])
    print(f"Saved predictions to {csv_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    images = collect_image_paths(args.inputs, args.max_samples)
    if not images:
        print("No images found for the provided inputs.", file=sys.stderr)
        sys.exit(1)

    model, processor = load_model_and_processor(args.model_dir, args.processor_dir, args.tokenizer_dir, device)
    predictions = run_inference(model, processor, images, device)
    maybe_save_csv(predictions, args.save_csv)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate a trained TrOCR model on a labeled CSV.

Example:
python scripts/eval_model.py \
  --model-dir ../khmer_trocr_model \
  --csv ../data/dataset/labels.csv \
  --data-root ../data/dataset/crops \
  --max-samples 500
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.dataset import KhmerOCRDataset  # noqa: E402
from src.metrics import build_compute_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TrOCR model on labeled crops.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "khmer_trocr_model")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "data/dataset/labels.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/dataset/crops")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit evaluation to N samples (0 = use all).",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=PROJECT_ROOT / "runs" / "eval_latest.json",
        help="Where to save eval metrics (default saves to runs/eval_latest.json).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing metrics to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["text"].notna()]
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42).reset_index(drop=True)

    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    processor.tokenizer = tokenizer

    dataset = KhmerOCRDataset(args.data_root, df, processor)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = Subset(dataset, range(args.max_samples))

    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.model_dir, "eval_tmp"),
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        do_train=False,
        do_eval=True,
        logging_strategy="no",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        eval_dataset=dataset,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(processor),
    )

    metrics = trainer.evaluate()
    print(metrics)

    if not args.no_save and args.save_metrics:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_dir": str(args.model_dir),
            "csv": str(args.csv),
            "data_root": str(args.data_root),
            "max_samples": args.max_samples,
            "metrics": metrics,
            "num_samples": len(dataset),
        }
        args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
        ts_name = args.save_metrics.parent / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with ts_name.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with args.save_metrics.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {ts_name} and {args.save_metrics}")


if __name__ == "__main__":
    main()

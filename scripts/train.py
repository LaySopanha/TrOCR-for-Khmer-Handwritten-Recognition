import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

# ---------------------------------------------------------------------
# 🔥 H200 GLOBAL OPTIMIZATION
# ---------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import KhmerOCRDataset
from src.metrics import build_compute_metrics

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data" / "dataset" / "crops"
TRAIN_CSV = PROJECT_ROOT / "data" / "dataset" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "dataset" / "val.csv"
MODEL_NAME = "microsoft/trocr-small-handwritten"
TOKENIZER_NAME = "xlm-roberta-large"

# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Khmer TrOCR model (H200 optimized).")

    parser.add_argument("--log-file", type=Path, default=PROJECT_ROOT / "runs/train_latest.json")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_NAME)

    # Tuned defaults (from Optuna best trial)
    parser.add_argument("--learning-rate", type=float, default=1.6625e-4)
    parser.add_argument("--epochs", type=int, default=12)

    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)

    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--warmup-steps", type=int, default=600)

    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
    )

    return parser.parse_args()

# ---------------------------------------------------------------------
def train(args: argparse.Namespace):
    # --------------------------------------------------
    # DATA
    # --------------------------------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # --------------------------------------------------
    # PROCESSOR & TOKENIZER
    # --------------------------------------------------
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    processor.tokenizer = tokenizer

    train_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        ]
    )

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------
    train_dataset = KhmerOCRDataset(DATA_ROOT, train_df, processor, transform=train_transform)
    val_dataset = KhmerOCRDataset(DATA_ROOT, val_df, processor, transform=None)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.decoder.resize_token_embeddings(len(tokenizer))

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = len(tokenizer)

    model.config.max_length = 128
    model.config.num_beams = 4
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    # Disable cache during training to reduce memory when checkpointing
    model.config.use_cache = False

    # 🔥 Critical for H200 memory efficiency
    model.gradient_checkpointing_enable()

    # --------------------------------------------------
    # TRAINING ARGS (H200)
    # --------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,

        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # 🚀 H200 precision
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,

        # 🚀 Performance
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,

        logging_steps=50,
        report_to="none",  # no W&B / cloud logging
    )

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(processor),
    )

    train_output = trainer.train()

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------
    save_dir = PROJECT_ROOT / "khmer_trocr_model"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # --------------------------------------------------
    # LOG SUMMARY (LOCAL)
    # --------------------------------------------------
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "train_samples": len(train_dataset),
        "eval_samples": len(val_dataset),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "train_metrics": train_output.metrics,
        "learning_rate": training_args.learning_rate,
        "epochs": training_args.num_train_epochs,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation": training_args.gradient_accumulation_steps,
        "precision": "bf16",
    }

    runs_dir = args.log_file.parent
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts_name = runs_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with ts_name.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with args.log_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV history
    csv_path = runs_dir / "train_history.csv"
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                ["timestamp", "train_samples", "eval_samples", "best_metric", "best_model_checkpoint"]
            )
        writer.writerow(
            [
                summary["timestamp"],
                summary["train_samples"],
                summary["eval_samples"],
                summary["best_metric"],
                summary["best_model_checkpoint"],
            ]
        )

    print(f"✅ Training complete on {summary['gpu']}")
    return summary

# ---------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)

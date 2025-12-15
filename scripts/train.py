import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

# Resolve project root based on this file location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import our custom modules
sys.path.append(str(PROJECT_ROOT))
from src.dataset import KhmerOCRDataset
from src.metrics import build_compute_metrics

# --- CONFIG ---
DATA_ROOT = PROJECT_ROOT / "data" / "dataset" / "crops"
CSV_PATH = PROJECT_ROOT / "data" / "dataset" / "labels.csv"
MODEL_NAME = "microsoft/trocr-small-handwritten"
TOKENIZER_NAME = "xlm-roberta-base"

def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description="Train Khmer TrOCR model.")
    # Kept for compatibility; defaults log to runs/train_latest.json
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_ROOT / "runs" / "train_latest.json",
        help="Where to store training summary metrics.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace):
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    df = df[df['text'].notna()]
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # 2. Setup Processor & Tokenizer
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    processor.tokenizer = tokenizer

    # 3. Create Datasets
    train_dataset = KhmerOCRDataset(DATA_ROOT, train_df, processor)
    val_dataset = KhmerOCRDataset(DATA_ROOT, val_df, processor)

    # 4. Load Model
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Model Config
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = len(tokenizer)
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # 5. Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir="../outputs",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        eval_strategy="steps",
        num_train_epochs=20,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
    )

    # 6. Trainer
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
    
    # Save
    model.save_pretrained("../khmer_trocr_model")
    processor.save_pretrained("../khmer_trocr_model")

    # Log summary to a timestamped file and latest
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(train_dataset),
        "eval_samples": len(val_dataset),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": training_args.metric_for_best_model,
        "train_metrics": train_output.metrics,
    }
    runs_dir = args.log_file.parent
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts_name = runs_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with ts_name.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with args.log_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved training summary to {ts_name} and {args.log_file}")

if __name__ == "__main__":
    args = parse_args()
    train(args)

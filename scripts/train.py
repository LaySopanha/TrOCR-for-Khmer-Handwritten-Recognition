import os
import sys
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

def train():
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

    trainer.train()
    
    # Save
    model.save_pretrained("../khmer_trocr_model")
    processor.save_pretrained("../khmer_trocr_model")

if __name__ == "__main__":
    train()

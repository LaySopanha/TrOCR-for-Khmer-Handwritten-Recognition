#!/usr/bin/env python3
"""
Train a SentencePiece BPE tokenizer on labels.csv texts.

Example:
python scripts/train_tokenizer.py --csv data/dataset/labels.csv --vocab-size 6000 --output data/tokenizer
"""

import argparse
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on Khmer labels.")
    parser.add_argument("--csv", type=Path, default=Path("data/dataset/labels.csv"))
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--vocab-size", type=int, default=6000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("data/tokenizer"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    texts = df[args.text_column].dropna().astype(str).tolist()
    print(f"Training tokenizer on {len(texts)} texts...")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    fast_tok.save_pretrained(args.output)
    print(f"Saved tokenizer to {args.output}")


if __name__ == "__main__":
    main()

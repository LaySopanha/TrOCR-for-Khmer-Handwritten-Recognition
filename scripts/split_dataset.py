#!/usr/bin/env python3
"""
Create train/val/test splits from labels.csv.

Defaults:
- Input: data/dataset/labels.csv
- Output dir: data/dataset/
- Splits: train 80%, val 10%, test 10%
- Seed: 42

Example:
python scripts/split_dataset.py \
  --input data/dataset/labels.csv \
  --output-dir data/dataset \
  --val-size 0.1 \
  --test-size 0.1 \
  --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split labels.csv into train/val/test.")
    parser.add_argument("--input", type=Path, default=Path("data/dataset/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction for validation.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction for test.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    df = df[df["text"].notna()].reset_index(drop=True)

    train_frac = 1.0 - args.val_size - args.test_size
    if train_frac <= 0:
        raise SystemExit("val-size + test-size must be < 1.0")

    train_df, rest_df = train_test_split(df, test_size=(args.val_size + args.test_size), random_state=args.seed)
    val_rel = args.val_size / (args.val_size + args.test_size) if (args.val_size + args.test_size) > 0 else 0.0
    val_df, test_df = train_test_split(rest_df, test_size=1 - val_rel, random_state=args.seed)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")


if __name__ == "__main__":
    main()

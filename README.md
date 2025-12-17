# Khmer Handwritten OCR (TrOCR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-TrOCR-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A specific-domain OCR project aiming to digitize **Khmer handwritten text lines** using the **Microsoft TrOCR** architecture. This repository contains the complete pipeline for data preprocessing (Label Studio/LabelMe), model training, hyperparameter tuning, and benchmarking against commercial and open-source baselines.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Workflow & Usage](#workflow--usage)
7. [Benchmarking](#benchmarking)
8. [Lessons & Troubleshooting](#lessons--troubleshooting)
9. [Roadmap](#roadmap)

## Overview

**Goal:** Achieve high-accuracy line-level OCR for the complex Khmer script (abugida system) using an encoder-decoder transformer approach.

**Key Features:**
- **End-to-End Pipeline:** From raw annotation JSONs to inference-ready models.
- **Normalization:** Handles Unicode NFC, zero-width characters, and whitespace cleaning.
- **Optimized Training:** Support for `bf16`, TF32, and gradient checkpointing (H200/A100 ready).
- **Benchmarking:** Tools to compare against Tesseract and Gemini 2.0 Flash.

## Dataset

The dataset consists of **~3.9k cropped line images** sourced from Label Studio exports and LabelMe annotations.

| Split | Count | Source File | Description |
| :--- | :--- | :--- | :--- |
| **Train** | 3,132 | `train.csv` | Augmented during training |
| **Val** | 391 | `val.csv` | Used for checkpoint selection |
| **Test** | 392 | `test.csv` | **Fixed seed (42)** for final evaluation |

**Preprocessing Standards:**
- **Cropping:** Polygon/Rectangle masks applied to generate white-background crops.
- **Cleaning:** Removal of `\u200b`, `\u200c`, `\u200d`, `\ufeff`.
- **Storage:** Crops saved to `data/dataset/crops/`; Labels in CSV format.

## Model Architecture

We utilize the **Microsoft TrOCR** (Transformer-based Optical Character Recognition) framework.

- **Base Model:** `microsoft/trocr-small-handwritten`
- **Tokenizer:** `xlm-roberta-large`
    - *Note:* Custom SentencePiece tokenizers (6–8k vocab) were tested but resulted in higher CER due to limited training data. XLM-R provided superior stability.
- **Decoder:** Embeddings resized to match the tokenizer vocabulary.
- **Generation config:** Beam Search (4), Length Penalty (2.0), No Repeat N-Gram (3).

## Project Structure

```text
.
├── data/
│   ├── annotation/          # Raw Label Studio/LabelMe JSONs
│   ├── dataset/             # Processed crops and CSV splits
│   └── tess/                # Tesseract ground truth files
├── scripts/
│   ├── prepare_data.py      # Label Studio -> Dataset
│   ├── prepare_labelme.py   # LabelMe -> Dataset
│   ├── split_dataset.py     # Create Train/Val/Test CSVs
│   ├── train.py             # Main training loop
│   ├── tune.py              # Optuna hyperparameter tuning
│   ├── eval_model.py        # CER calculation
│   ├── benchmark_gemini.py  # Gemini API benchmark
│   └── benchmark_tesseract.py
├── khmer_trocr_model/       # Saved local model/tokenizer artifacts
├── outputs/                 # Training checkpoints
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/yourusername/khmer-trocr.git
cd khmer-trocr
pip install -r requirements.txt
```

## Workflow & Usage

### 1. Data Preparation
Convert raw annotations into the training format.

```bash
# Process Label Studio export
python scripts/prepare_data.py

# Process LabelMe folders
python scripts/prepare_labelme.py

# Generate fixed Train/Val/Test splits (Seed 42)
python scripts/split_dataset.py --input data/dataset/labels.csv
```

### 2. Training
Run the training script (default parameters tuned via Optuna).

```bash
python scripts/train.py
```

- Key params: LR ≈ 1.7e-4, Batch 16, Accumulation 1, Epochs 12.
- Logging: View metrics in `runs/train_history.csv` or `runs/train_latest.json`.
- To run hyperparameter tuning:

```bash
python scripts/tune.py
```

### 3. Evaluation
Evaluate the model using Character Error Rate (CER).

> **Warning:** Checkpoints in `outputs/` do not contain the tokenizer. Point `--processor-dir` and `--tokenizer-dir` to your base model or saved tokenizer folder (`khmer_trocr_model`) when evaluating intermediate checkpoints.

```bash
# Evaluate a specific checkpoint on the test set
python scripts/eval_model.py \
  --model-dir outputs/checkpoint-1800 \
  --processor-dir khmer_trocr_model \
  --tokenizer-dir khmer_trocr_model \
  --csv data/dataset/test.csv \
  --data-root data/dataset/crops \
  --greedy
```

Tip: Use `--greedy` (Greedy Decoding). It often yields better CER than beam search for this dataset.

## Benchmarking

We compare TrOCR against standard baselines:

| Model | Script | Notes |
| :--- | :--- | :--- |
| Tesseract 5 | `benchmark_tesseract.py` | Uses khm language data. Fine-tuning currently WIP (issue with `.lstmf` generation). |
| Gemini 2.0 | `benchmark_gemini.py` | Requires API key. Limited by free-tier quotas (sleeps required). |

Commands:

```bash
# Tesseract Baseline
python scripts/benchmark_tesseract.py --lang khm --psm 6 --max-samples 0

# Gemini Flash (Quota limited)
python scripts/benchmark_gemini.py --model gemini-2.0-flash --max-samples 50 --sleep 5
```

## Lessons & Troubleshooting

- Tokenizer alignment is critical; mismatch between model head and tokenizer can push CER > 90%.
- Always evaluate on the fixed split at `data/dataset/test.csv`; random splits make results incomparable.
- Greedy decoding often beats beam search for this handwriting dataset.
- Tesseract fine-tuning is currently blocked; `scripts/export_tesseract_gt.py` works, but tesstrain expects `.box` files instead of `.gt.txt`.

## Roadmap

- Unblock Tesseract fine-tuning pipeline (adjust export to `.box` or tesstrain inputs).
- Expand dataset with harder samples and additional augmentation.
- Package inference script with simple CLI and Docker image.

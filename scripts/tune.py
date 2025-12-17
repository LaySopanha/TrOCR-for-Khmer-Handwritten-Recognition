#!/usr/bin/env python3
"""
Lightweight Optuna tuning for Khmer TrOCR training.

Example:
python scripts/tune.py --trials 5
"""

import argparse
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure we can import train()
import sys
sys.path.append(str(PROJECT_ROOT / "scripts"))
from train import train  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Khmer TrOCR.")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials.")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help='Optuna storage URL, e.g., "sqlite:///runs/optuna.db". If unset, uses in-memory.',
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="khmer-trocr",
        help="Optuna study name.",
    )
    return parser.parse_args()


def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 2e-4, log=True)
    epochs = trial.suggest_int("epochs", 10, 20)
    train_bs = trial.suggest_categorical("train_batch_size", [8, 16, 32])
    eval_steps = trial.suggest_int("eval_steps", 75, 200, step=25)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 800, step=100)
    grad_accum = trial.suggest_categorical("gradient_accumulation", [1, 2])

    # Derived/output paths
    runs_dir = PROJECT_ROOT / "runs"
    log_file = runs_dir / "optuna" / f"train_trial_{trial.number}.json"
    output_dir = runs_dir / "optuna" / f"trial_{trial.number}_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a dummy argparse.Namespace to reuse train()
    args = argparse.Namespace(
        log_file=log_file,
        tokenizer="xlm-roberta-large",
        learning_rate=lr,
        epochs=epochs,
        train_batch_size=train_bs,
        eval_batch_size=train_bs,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
        gradient_accumulation=grad_accum,
        num_workers=8,
    )

    summary = train(args)
    best_metric = summary.get("best_metric", None)
    if best_metric is None:
        raise optuna.TrialPruned("No best_metric reported.")
    # CER: lower is better
    return best_metric


def main() -> None:
    args = parse_args()
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.trials)
    print("Best trial:")
    best = study.best_trial
    print(f"  value (CER): {best.value}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

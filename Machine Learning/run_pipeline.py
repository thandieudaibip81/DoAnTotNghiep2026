#!/usr/bin/env python3
"""
run_pipeline.py — CLI entry point for the ML pipeline.

Usage examples:
    # Run the full pipeline (tune + train + evaluate) with SMOTE
    python run_pipeline.py --step all --sampling smote

    # Only run baseline (Logistic Regression on raw data)
    python run_pipeline.py --step baseline

    # Tune only Random Forest and Logistic Regression
    python run_pipeline.py --step tune --models rf,lr --sampling smote

    # Train with tuned params (skip tuning)
    python run_pipeline.py --step train --sampling smote

    # Evaluate a previously trained model
    python run_pipeline.py --step evaluate --sampling smote
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_NAMES, SAMPLING_SMOTE, VALID_SAMPLING
from src.evaluator import evaluate_model, export_metrics_csv
from src.models import get_model_display_name
from src.preprocessing import (
    get_sampled_data,
    load_data,
    scale_features,
    split_data,
    subsample_for_tuning,
)
from src.trainer import train_all, train_model, save_model, _load_best_params, load_model
from src.tuner import tune_all

# ──────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-24s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# ──────────────────────────────────────────────────
# Name shortcuts
# ──────────────────────────────────────────────────
_SHORTCUTS = {
    "rf": "random_forest",
    "lr": "logistic_regression",
    "knn": "knn",
    "svm": "svm",
    "nn": "neural_network",
}


def _resolve_model_names(raw: str) -> list[str]:
    """Parse comma-separated model names, expand shortcuts."""
    names = []
    for token in raw.split(","):
        token = token.strip().lower()
        resolved = _SHORTCUTS.get(token, token)
        if resolved not in MODEL_NAMES:
            logger.error("Unknown model name: '%s'", token)
            sys.exit(1)
        names.append(resolved)
    return names


# ──────────────────────────────────────────────────
# Step implementations
# ──────────────────────────────────────────────────


def step_baseline() -> None:
    """Train all models on raw (imbalanced) data to demonstrate imbalance bias.
    
    NOTE: Baseline models are evaluated and their metrics are printed for
    comparison, but they are NOT saved to models/ (only SMOTE models are
    persisted for production use).
    
    SVM/KNN are sub-sampled (10%) and Neural Network (20%) during baseline
    to prevent hanging on the full dataset due to O(n²-n³) complexity.
    """
    import pandas as pd
    from src.evaluator import evaluate_model, export_metrics_csv

    logger.info("=" * 60)
    logger.info("BASELINE: All models on RAW imbalanced data (no save)")
    logger.info("=" * 60)

    df = load_data()
    df = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)

    # Models that are too slow for full baseline data
    SLOW_MODELS_10 = {"svm", "knn"}       # O(n²-n³) → use 10%
    SLOW_MODELS_20 = {"neural_network"}   # iterative → use 20%

    all_results = []
    # Train without any resampling — metrics only, no .pkl saved
    for name in MODEL_NAMES:
        # Sub-sample for slow models to prevent hanging
        if name in SLOW_MODELS_10:
            X_train_bl, y_train_bl = subsample_for_tuning(X_train, y_train, fraction=0.10)
            logger.info("Baseline sub-sampling '%s' → %d rows (10%%)", name, len(X_train_bl))
        elif name in SLOW_MODELS_20:
            X_train_bl, y_train_bl = subsample_for_tuning(X_train, y_train, fraction=0.20)
            logger.info("Baseline sub-sampling '%s' → %d rows (20%%)", name, len(X_train_bl))
        else:
            X_train_bl, y_train_bl = X_train, y_train

        model = train_model(name, X_train_bl, y_train_bl, params={})
        metrics = evaluate_model(model, X_test, y_test, name, "baseline")
        # Baseline models are NOT saved (only SMOTE models are persisted)
        logger.info("Baseline '%s' evaluated — skipping .pkl save", name)

        all_results.append({
            "model": get_model_display_name(name),
            "sampling": "baseline (no sampling)",
            **metrics,
        })

    results_df = pd.DataFrame(all_results)
    export_metrics_csv(results_df, "baseline")

    print("\n⚠️  Baseline shows high Accuracy but LOW Recall — proving the need for resampling!")
    print(results_df.to_string(index=False))


def step_tune(model_names: list[str], sampling: str) -> None:
    """Hyperparameter tuning for selected models."""
    logger.info("=" * 60)
    logger.info("TUNING: models=%s  |  sampling=%s", model_names, sampling)
    logger.info("=" * 60)

    df = load_data()
    df = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, y_train_s = get_sampled_data(X_train, y_train, strategy=sampling)

    tune_all(X_train_s, y_train_s, model_names=model_names)
    logger.info("Tuning complete. Best params saved to reports/best_params_*.json")


def step_train(model_names: list[str], sampling: str) -> None:
    """Train models with best (or default) params."""
    train_all(sampling=sampling, model_names=model_names, use_tuned_params=True)


def step_evaluate(model_names: list[str], sampling: str) -> None:
    """Evaluate previously trained models."""
    import pandas as pd

    df = load_data()
    df = scale_features(df, fit=False)
    _, X_test, _, y_test = split_data(df)

    all_results = []
    for name in model_names:
        model = load_model(name, sampling)
        metrics = evaluate_model(model, X_test, y_test, name, sampling)
        all_results.append({
            "model": get_model_display_name(name),
            "sampling": sampling,
            **metrics,
        })

    results_df = pd.DataFrame(all_results)
    export_metrics_csv(results_df, f"{sampling}_reeval")
    print("\n📊 Re-evaluation Results:")
    print(results_df.to_string(index=False))


# ──────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=["baseline", "tune", "train", "evaluate", "all"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--sampling",
        default=SAMPLING_SMOTE,
        choices=VALID_SAMPLING,
        help="Sampling strategy (default: smote).",
    )
    parser.add_argument(
        "--models",
        default=",".join(["rf", "lr", "knn", "svm", "nn"]),
        help="Comma-separated model names/shortcuts (rf, lr, knn, svm, nn).",
    )

    args = parser.parse_args()
    model_names = _resolve_model_names(args.models)

    if args.step == "baseline":
        step_baseline()
    elif args.step == "tune":
        step_tune(model_names, args.sampling)
    elif args.step == "train":
        step_train(model_names, args.sampling)
    elif args.step == "evaluate":
        step_evaluate(model_names, args.sampling)
    elif args.step == "all":
        logger.info("Running full pipeline: baseline → tune → train")
        step_baseline()
        step_tune(model_names, args.sampling)
        step_train(model_names, args.sampling)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

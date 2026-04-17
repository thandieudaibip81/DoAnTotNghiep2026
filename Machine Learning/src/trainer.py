"""
trainer.py — Train final models with best params, save artefacts.

Workflow:
    1. Load preprocessed data + apply sampling strategy
    2. (Optional) load best params from ``reports/best_params_*.json``
    3. Fit each model on full (resampled) training set
    4. Evaluate on the held-out test set
    5. Save model ``.pkl`` files to ``models/``
    6. (Optional) log to MLflow if ``MLFLOW_TRACKING_URI`` is set
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SAMPLING_SMOTE,
    MODEL_NAMES,
)
from src.evaluator import evaluate_model, export_feature_importance, export_metrics_csv
from src.models import get_model, get_model_display_name
from src.preprocessing import (
    get_sampled_data,
    load_data,
    scale_features,
    split_data,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Helper: load tuned params from JSON
# ──────────────────────────────────────────────────


def _load_best_params(model_name: str) -> Dict[str, Any]:
    """Try to load Optuna best params JSON; return empty dict if missing."""
    path = REPORTS_DIR / f"best_params_{model_name}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded tuned params for '%s' from %s", model_name, path)
        return data.get("params", {})
    logger.warning(
        "No tuned params found for '%s' — using defaults.", model_name,
    )
    return {}


# ──────────────────────────────────────────────────
# Core training function
# ──────────────────────────────────────────────────


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fit a single model on the provided training data.

    Parameters
    ----------
    model_name : str
        Canonical name (e.g. ``"random_forest"``).
    X_train, y_train
        Training features & labels.
    params : dict | None
        Hyperparameters; if None, loads from tuned JSON or defaults.

    Returns
    -------
    Fitted sklearn estimator.
    """
    if params is None:
        params = _load_best_params(model_name)

    model = get_model(model_name, params)
    logger.info("Training '%s' on %d samples…", model_name, len(X_train))

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Training '%s' completed in %.1fs", model_name, elapsed)

    return model


# ──────────────────────────────────────────────────
# Save / load model artefacts
# ──────────────────────────────────────────────────


def save_model(model: Any, model_name: str, sampling: str) -> str:
    """Persist a fitted model to ``models/<name>_<sampling>.pkl``.

    Returns
    -------
    str
        Path where the model was saved.
    """
    filename = f"{model_name}_{sampling}.pkl"
    path = str(MODELS_DIR / filename)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(model_name: str, sampling: str) -> Any:
    """Load a previously saved model from ``models/``."""
    path = MODELS_DIR / f"{model_name}_{sampling}.pkl"
    model = joblib.load(str(path))
    logger.info("Model loaded ← %s", path)
    return model


# ──────────────────────────────────────────────────
# MLflow logging (optional)
# ──────────────────────────────────────────────────


def _log_to_mlflow(
    model_name: str,
    sampling: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model_path: str,
) -> None:
    """Log run to MLflow if MLFLOW_TRACKING_URI is set."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("credit_card_fraud_detection")

        with mlflow.start_run(run_name=f"{model_name}_{sampling}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
            mlflow.sklearn.log_model(
                joblib.load(model_path), artifact_path="model",
            )
        logger.info("Logged to MLflow: %s_%s", model_name, sampling)
    except ImportError:
        logger.warning("mlflow not installed — skipping MLflow logging.")
    except Exception as exc:
        logger.error("MLflow logging failed: %s", exc)


# ──────────────────────────────────────────────────
# Orchestrator: full pipeline
# ──────────────────────────────────────────────────


def train_all(
    sampling: str = SAMPLING_SMOTE,
    model_names: Optional[List[str]] = None,
    use_tuned_params: bool = True,
) -> pd.DataFrame:
    """End-to-end: preprocess → train → evaluate → save for all models.

    Parameters
    ----------
    sampling : str
        ``"none"`` | ``"undersample"`` | ``"smote"``.
    model_names : list[str] | None
        Subset of models; defaults to all four.
    use_tuned_params : bool
        If True, load tuned params from JSON files.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for every trained model.
    """
    names = model_names or MODEL_NAMES

    # 1. Preprocess
    logger.info("=" * 60)
    logger.info("PIPELINE START  |  sampling=%s  |  models=%s", sampling, names)
    logger.info("=" * 60)

    df = load_data()
    df = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_s, y_train_s = get_sampled_data(X_train, y_train, strategy=sampling)

    all_results: List[Dict[str, Any]] = []

    for name in names:
        # 2. Get params
        params = _load_best_params(name) if use_tuned_params else {}

        # 3. Train
        model = train_model(name, X_train_s, y_train_s, params=params)

        # 4. Evaluate
        metrics = evaluate_model(
            model, X_test, y_test,
            model_name=name, sampling=sampling,
        )

        # 5. Save model
        model_path = save_model(model, name, sampling)

        # 6. Feature importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            export_feature_importance(model, X_train.columns.tolist(), name, sampling)

        # 7. MLflow
        _log_to_mlflow(name, sampling, params, metrics, model_path)

        all_results.append(
            {
                "model": get_model_display_name(name),
                "sampling": sampling,
                **metrics,
            }
        )

    # 8. Export comparison CSV
    results_df = pd.DataFrame(all_results)
    export_metrics_csv(results_df, sampling)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))

    return results_df

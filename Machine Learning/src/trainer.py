"""
trainer.py — Train final models with best params, save artefacts.

Workflow:
    1. Load preprocessed data + apply sampling strategy
    2. (Optional) load best params from ``reports/best_params_*.json``
    3. Fit each model on full (resampled) training set
    4. Evaluate on the held-out test set
    5. Save model ``.pkl`` files to ``models/``
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
    """Try to load Optuna best params JSON; return empty dict if missing.
    
    For neural_network, converts Optuna's flat param format
    (n_layers, layer_X_units, lr) into KerasNN constructor format
    (layers=[...], learning_rate).
    """
    path = REPORTS_DIR / f"best_params_{model_name}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded tuned params for '%s' from %s", model_name, path)
        params = data.get("params", {})
        
        # Convert Optuna flat NN params → KerasNN constructor params
        if model_name == "neural_network" and "n_layers" in params:
            n_layers = params.pop("n_layers")
            layers = []
            for i in range(n_layers):
                key = f"layer_{i}_units"
                if key in params:
                    layers.append(params.pop(key))
            params["layers"] = layers
            # Optuna uses 'lr' but KerasNN expects 'learning_rate'
            if "lr" in params:
                params["learning_rate"] = params.pop("lr")
        
        return params
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

    Only saves models trained with SMOTE sampling for production use.
    Baseline and undersample models are NOT persisted.

    Returns
    -------
    str
        Path where the model was saved, or empty string if skipped.
    """
    if sampling not in ("smote", "SMOTE"):
        logger.info("Skipping save for '%s' with sampling='%s' (only SMOTE is saved)", model_name, sampling)
        return ""

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
        Subset of models; defaults to all five.
    use_tuned_params : bool
        If True, load tuned params from JSON files.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for every trained model.
    """
    from src.preprocessing import subsample_for_tuning

    names = model_names or MODEL_NAMES

    # Models with O(n²-n³) complexity that cannot handle full SMOTE data
    SLOW_MODELS = {"svm", "knn"}  # ~455k rows → sub-sample to 20%

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

        # 3. Sub-sample for slow models to prevent hanging
        if name in SLOW_MODELS and sampling == SAMPLING_SMOTE:
            X_fit, y_fit = subsample_for_tuning(X_train_s, y_train_s, fraction=0.20)
            logger.info("Sub-sampling '%s' for training → %d rows (20%%)", name, len(X_fit))
        else:
            X_fit, y_fit = X_train_s, y_train_s

        # 4. Train
        model = train_model(name, X_fit, y_fit, params=params)

        # 5. Evaluate
        metrics = evaluate_model(
            model, X_test, y_test,
            model_name=name, sampling=sampling,
        )

        # 6. Save model
        model_path = save_model(model, name, sampling)

        # 7. Feature importance (tree-based models only)
        if hasattr(model, "feature_importances_"):
            export_feature_importance(model, X_train.columns.tolist(), name, sampling)

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


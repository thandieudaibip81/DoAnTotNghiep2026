"""
tuner.py — Hyperparameter optimisation with Optuna.

Each model has its own search space.  Fast models (RF, LR) are tuned
on the full dataset; slow models (KNN, SVM, Neural Network) are
sub-sampled during tuning for speed.  The final train step always
uses 100% SMOTE data, so tuning quality is preserved.

Objective: maximise **F1-Score** using StratifiedKFold cross-validation
so that Recall is properly rewarded without sacrificing Precision.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import (
    MODEL_NAMES,
    RANDOM_STATE,
    REPORTS_DIR,
    SAMPLE_FRACTION_KNN_SVM,
    SAMPLE_FRACTION_NN,
    TUNER_CV_FOLDS,
    TUNER_N_TRIALS,
)
from src.models import get_model
from src.preprocessing import subsample_for_tuning

logger = logging.getLogger(__name__)

# Silence Optuna's verbose default output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────
# Search-space definitions
# ──────────────────────────────────────────────────


def _rf_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Random Forest search space."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
    }
    return _cv_f1(get_model("random_forest", params), X, y)


def _lr_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Logistic Regression search space."""
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "solver": "saga",
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    return _cv_f1(get_model("logistic_regression", params), X, y)


def _knn_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """KNN search space."""
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
    }
    return _cv_f1(get_model("knn", params), X, y)


def _svm_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """SVM search space — rbf kernel only (poly is too slow for large datasets)."""
    params: Dict[str, Any] = {
        "C": trial.suggest_float("C", 0.1, 50.0, log=True),
        "kernel": "rbf",
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    return _cv_f1(get_model("svm", params), X, y)


def _nn_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Neural Network search space."""
    n_layers = trial.suggest_int("n_layers", 2, 5)
    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(f"layer_{i}_units", 8, 64, step=8)
        layers.append(n_units)

    params = {
        "layers": layers,
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "learning_rate": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 20, 50),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
    }
    return _cv_f1(get_model("neural_network", params), X, y, n_jobs=1)


# Dispatcher
_OBJECTIVES = {
    "random_forest": _rf_objective,
    "logistic_regression": _lr_objective,
    "knn": _knn_objective,
    "svm": _svm_objective,
    "neural_network": _nn_objective,
}

# ──────────────────────────────────────────────────
# Cross-validation helper
# ──────────────────────────────────────────────────


def _cv_f1(model: Any, X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> float:
    """Return mean F1-Score from stratified k-fold CV."""
    skf = StratifiedKFold(
        n_splits=TUNER_CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=n_jobs)
    return float(scores.mean())


# ──────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = TUNER_N_TRIALS,
) -> Dict[str, Any]:
    """Run Optuna study to find the best hyperparameters for *model_name*.

    Parameters
    ----------
    model_name : str
        Canonical model name (e.g. ``"random_forest"``).
    X_train, y_train
        Training features & labels (already resampled if applicable).
    n_trials : int
        Number of Optuna trials.

    Returns
    -------
    dict
        Best hyperparameters found by the study.
    """
    if model_name not in _OBJECTIVES:
        raise ValueError(f"No search space defined for '{model_name}'")

    # --- Selective sub-sampling for slow models ---
    # Tuning runs on a representative subset; final train always uses 100% SMOTE.
    # LR with saga solver on 455k rows takes ~3-5 min per trial — subsample helps.
    SUBSAMPLE_20 = {"knn", "svm"}          # O(n²) / O(n³) complexity
    SUBSAMPLE_30 = {"logistic_regression", "neural_network"}  # iterative / saga

    if model_name in SUBSAMPLE_20:
        frac = SAMPLE_FRACTION_KNN_SVM
        X_tune, y_tune = subsample_for_tuning(X_train, y_train, fraction=frac)
    elif model_name in SUBSAMPLE_30:
        frac = SAMPLE_FRACTION_NN
        X_tune, y_tune = subsample_for_tuning(X_train, y_train, fraction=frac)
    else:
        # Random Forest — fast enough on full data
        X_tune, y_tune = X_train, y_train

    objective_fn = _OBJECTIVES[model_name]

    study = optuna.create_study(
        direction="maximize",
        study_name=f"tune_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    def _objective(trial: optuna.Trial) -> float:
        return objective_fn(trial, X_tune.values, y_tune.values)

    logger.info(
        "Starting Optuna study for '%s' (%d trials)…", model_name, n_trials,
    )
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_f1 = study.best_value
    logger.info(
        "Best F1 for '%s': %.4f  |  Params: %s",
        model_name, best_f1, best_params,
    )

    # Persist best params as JSON
    out_path = REPORTS_DIR / f"best_params_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(
            {"model": model_name, "best_f1": best_f1, "params": best_params},
            f, indent=2,
        )
    logger.info("Saved best params → %s", out_path)

    return best_params


def tune_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: Optional[list[str]] = None,
    n_trials: int = TUNER_N_TRIALS,
) -> Dict[str, Dict[str, Any]]:
    """Tune all (or selected) models and return a dict of best params.

    Parameters
    ----------
    model_names : list[str] | None
        Subset of models to tune; defaults to all four.
    n_trials : int
        Trials per model.

    Returns
    -------
    dict
        ``{model_name: best_params_dict}``
    """
    names = model_names or MODEL_NAMES
    results: Dict[str, Dict[str, Any]] = {}
    for name in names:
        results[name] = tune_model(name, X_train, y_train, n_trials=n_trials)
    return results

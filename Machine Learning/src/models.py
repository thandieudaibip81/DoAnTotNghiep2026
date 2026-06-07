"""
models.py — Model factory for the five classifiers.

Provides a clean factory pattern so callers never need to import
individual sklearn / Keras classes directly.

Supported models:
    - random_forest       → RandomForestClassifier
    - logistic_regression → LogisticRegression
    - knn                 → KNeighborsClassifier
    - svm                 → SVC (with probability=True for ROC/PR curves)
    - neural_network      → KerasNN (TensorFlow/Keras wrapper)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import MODEL_NAMES, RANDOM_STATE
from src.keras_nn import KerasNN

logger = logging.getLogger(__name__)


_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "logistic_regression": {
        "C": 1.0,
        "solver": "saga",      # supports elasticnet/l1_ratio
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "n_jobs": -1,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": True,          # needed for predict_proba → ROC/PR
        "random_state": RANDOM_STATE,
    },
    "neural_network": {
        "layers": [30, 20, 10, 5],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 35,
        "batch_size": 250,
        "activation": "relu",
        "random_state": RANDOM_STATE,
        "verbose": 0,
    },
}

# Map canonical name → sklearn / Keras class
_REGISTRY: Dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "neural_network": KerasNN,
}


# ──────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────


def get_model(
    name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instantiate a classifier by canonical name.

    Parameters
    ----------
    name : str
        One of ``"random_forest"``, ``"logistic_regression"``,
        ``"knn"``, ``"svm"``.
    params : dict | None
        Override default hyperparameters.  Keys not provided fall
        back to ``_DEFAULTS[name]``.

    Returns
    -------
    sklearn estimator
        Un-fitted classifier instance.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {MODEL_NAMES}"
        )

    # Merge: defaults ← overrides
    merged: Dict[str, Any] = {**_DEFAULTS[name], **(params or {})}
    model = _REGISTRY[name](**merged)
    logger.info("Created model '%s' with params: %s", name, merged)
    return model


def get_all_models(
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a dict of all four models with default (or custom) params.

    Parameters
    ----------
    custom_params : dict | None
        ``{model_name: {param: value, ...}, ...}`` to override defaults.

    Returns
    -------
    dict
        ``{model_name: estimator_instance}``
    """
    custom_params = custom_params or {}
    return {
        name: get_model(name, custom_params.get(name))
        for name in MODEL_NAMES
    }


def get_model_display_name(name: str) -> str:
    """Human-readable label for charts and reports."""
    _DISPLAY = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "knn": "K-Nearest Neighbors",
        "svm": "Support Vector Machine",
        "neural_network": "Neural Network (Keras)",
    }
    return _DISPLAY.get(name, name)

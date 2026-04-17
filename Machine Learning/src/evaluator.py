"""
evaluator.py — Model evaluation, visualisation, and CSV exports.

Exports are designed so that Power BI (or any BI tool) can directly
import the CSVs for dashboard creation.

Metrics computed:
    • Accuracy, Precision, Recall, F1-Score
    • AUC-ROC, AUC-PR (Average Precision)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import REPORTS_DIR
from src.models import get_model_display_name

logger = logging.getLogger(__name__)

# Visual defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ──────────────────────────────────────────────────
# 1. Core metric computation
# ──────────────────────────────────────────────────


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    sampling: str = "none",
) -> Dict[str, float]:
    """Compute all metrics and generate plots for a fitted model.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : held-out test data
    model_name, sampling : used for labelling plots / files

    Returns
    -------
    dict
        Metric name → float value.
    """
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    y_proba: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)

    metrics: Dict[str, float] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    if y_proba is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_test, y_proba), 4)
        metrics["auc_pr"] = round(average_precision_score(y_test, y_proba), 4)
    else:
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    logger.info(
        "[%s | %s] Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC-ROC=%.4f",
        model_name, sampling,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1_score"], metrics["auc_roc"],
    )

    # Generate plots
    label = f"{get_model_display_name(model_name)} ({sampling})"
    plot_confusion_matrix(y_test, y_pred, label, model_name, sampling)
    if y_proba is not None:
        plot_precision_recall_curve(y_test, y_proba, label, model_name, sampling)

    # Print classification report to console
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics


# ──────────────────────────────────────────────────
# 2. Confusion Matrix plot
# ──────────────────────────────────────────────────


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    model_name: str = "model",
    sampling: str = "none",
) -> None:
    """Save a confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")

    path = REPORTS_DIR / f"cm_{model_name}_{sampling}.png"
    fig.savefig(str(path))
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", path)


# ──────────────────────────────────────────────────
# 3. Precision-Recall curve
# ──────────────────────────────────────────────────


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    title: str,
    model_name: str = "model",
    sampling: str = "none",
) -> None:
    """Save a Precision-Recall curve as PNG."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, lw=2, label=f"AUC-PR = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {title}")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    path = REPORTS_DIR / f"pr_{model_name}_{sampling}.png"
    fig.savefig(str(path))
    plt.close(fig)
    logger.info("PR curve saved → %s", path)


# ──────────────────────────────────────────────────
# 4. CSV exports (Power BI-friendly)
# ──────────────────────────────────────────────────


def export_metrics_csv(
    results_df: pd.DataFrame,
    sampling: str = "all",
) -> str:
    """Export model comparison metrics to CSV.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    path = str(REPORTS_DIR / f"model_comparison_{sampling}.csv")
    results_df.to_csv(path, index=False)
    logger.info("Metrics CSV saved → %s", path)
    return path


def export_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str = "model",
    sampling: str = "none",
) -> str:
    """Export feature importances for tree-based models to CSV.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model '%s' has no feature_importances_ attribute.", model_name)
        return ""

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    path = str(REPORTS_DIR / f"feature_importance_{model_name}_{sampling}.csv")
    imp_df.to_csv(path, index=False)
    logger.info("Feature importance saved → %s", path)

    # Also plot top-20
    _plot_feature_importance(imp_df.head(20), model_name, sampling)
    return path


def _plot_feature_importance(
    imp_df: pd.DataFrame,
    model_name: str,
    sampling: str,
) -> None:
    """Bar chart of top-N feature importances."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title(
        f"Top-{len(imp_df)} Feature Importance — "
        f"{get_model_display_name(model_name)} ({sampling})"
    )
    ax.set_xlabel("Importance")

    path = REPORTS_DIR / f"feat_imp_{model_name}_{sampling}.png"
    fig.savefig(str(path))
    plt.close(fig)
    logger.info("Feature importance plot saved → %s", path)

"""
preprocessing.py — Data loading, feature scaling, and sampling strategies.

Responsibilities:
    1. Load the raw CSV dataset
    2. Apply RobustScaler to Amount & Time (save scaler for inference)
    3. Stratified train/test split
    4. Three sampling scenarios: none | random under-sampling | SMOTE

The saved scaler is reused in the Ops phase (API inference).
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    SAMPLE_FRACTION,
    SAMPLING_NONE,
    SAMPLING_SMOTE,
    SAMPLING_UNDERSAMPLE,
    SCALE_COLS,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# 1. Data loading
# ──────────────────────────────────────────────────


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the credit-card transaction CSV.

    Parameters
    ----------
    path : str | None
        Override path; defaults to ``config.DATA_FILE``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all 31 columns.
    """
    file_path = path or str(DATA_FILE)
    logger.info("Loading data from %s", file_path)
    df = pd.read_csv(file_path)

    # Basic sanity checks
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found"
    assert df.isnull().sum().sum() == 0, "Dataset contains null values"
    logger.info(
        "Loaded %d rows  |  Fraud: %d (%.3f%%)",
        len(df),
        df[TARGET_COL].sum(),
        df[TARGET_COL].mean() * 100,
    )
    return df


# ──────────────────────────────────────────────────
# 2. Feature scaling
# ──────────────────────────────────────────────────


def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler_path: str | None = None,
) -> pd.DataFrame:
    """Apply RobustScaler to Amount & Time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (modified in-place copy is returned).
    fit : bool
        If True, fit a new scaler and save it.  If False, load an
        existing scaler (for inference).
    scaler_path : str | None
        Override path for the scaler pickle.

    Returns
    -------
    pd.DataFrame
        Dataframe with scaled Amount & Time.
    """
    save_path = scaler_path or str(MODELS_DIR / "robust_scaler.pkl")
    df = df.copy()

    if fit:
        scaler = RobustScaler()
        df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
        joblib.dump(scaler, save_path)
        logger.info("Scaler fitted & saved → %s", save_path)
    else:
        scaler = joblib.load(save_path)
        df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
        logger.info("Scaler loaded from %s", save_path)

    return df


# ──────────────────────────────────────────────────
# 3. Train / Test split
# ──────────────────────────────────────────────────

DataSplit = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def split_data(df: pd.DataFrame) -> DataSplit:
    """Stratified 80/20 train-test split.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Split → Train: %d  |  Test: %d  |  Train fraud ratio: %.3f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────
# 4. Sampling strategies
# ──────────────────────────────────────────────────


def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Random under-sampling: reduce majority class to match minority.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    logger.info(
        "Under-sampling → %d samples  (0: %d | 1: %d)",
        len(y_res),
        (y_res == 0).sum(),
        (y_res == 1).sum(),
    )
    return X_res, y_res


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """SMOTE: generate synthetic fraud samples to balance classes.

    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE → %d samples  (0: %d | 1: %d)",
        len(y_res),
        (y_res == 0).sum(),
        (y_res == 1).sum(),
    )
    return X_res, y_res


def get_sampled_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = SAMPLING_SMOTE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Dispatcher — apply the requested sampling strategy.

    Parameters
    ----------
    strategy : str
        One of ``"none"`` | ``"undersample"`` | ``"smote"``.

    Returns
    -------
    tuple
        (X_train_resampled, y_train_resampled)
    """
    if strategy == SAMPLING_NONE:
        logger.info("No resampling applied.")
        return X_train, y_train
    elif strategy == SAMPLING_UNDERSAMPLE:
        return apply_undersampling(X_train, y_train)
    elif strategy == SAMPLING_SMOTE:
        return apply_smote(X_train, y_train)
    else:
        raise ValueError(
            f"Unknown sampling strategy '{strategy}'. "
            f"Choose from: none, undersample, smote"
        )


# ──────────────────────────────────────────────────
# 5. Convenience: subsample for slow models (KNN/SVM tuning)
# ──────────────────────────────────────────────────


def subsample_for_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    fraction: float = SAMPLE_FRACTION,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Take a stratified subsample for faster tuning of KNN / SVM.

    Parameters
    ----------
    fraction : float
        Proportion of data to keep (default from config).

    Returns
    -------
    tuple
        (X_sub, y_sub)
    """
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=fraction,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Sub-sampled for tuning → %d rows (%.0f%%)", len(X_sub), fraction * 100)
    return X_sub, y_sub

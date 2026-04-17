"""
config.py — Project-wide configuration & constants.

Centralises all paths, hyperparameters, and settings so every other
module imports from here rather than hard-coding values.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Auto-create output directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Data file
# ──────────────────────────────────────────────
DATA_FILE: Path = DATA_DIR / "creditcard.csv"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_STATE: int = 42

# ──────────────────────────────────────────────
# Train / Test split
# ──────────────────────────────────────────────
TEST_SIZE: float = 0.2

# ──────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────
# Fraction of data used for KNN/SVM tuning (speed optimisation)
SAMPLE_FRACTION: float = 0.15

# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
TARGET_COL: str = "Class"
SCALE_COLS: list[str] = ["Amount", "Time"]
# V1–V28 are already PCA-scaled; only Amount & Time need scaling.

# ──────────────────────────────────────────────
# Model registry — canonical names
# ──────────────────────────────────────────────
MODEL_NAMES: list[str] = [
    "random_forest",
    "logistic_regression",
    "knn",
    "svm",
]

# ──────────────────────────────────────────────
# Tuner defaults
# ──────────────────────────────────────────────
TUNER_N_TRIALS: int = 50
TUNER_CV_FOLDS: int = 5

# ──────────────────────────────────────────────
# Sampling strategy names (for CLI / pipeline)
# ──────────────────────────────────────────────
SAMPLING_NONE: str = "none"
SAMPLING_UNDERSAMPLE: str = "undersample"
SAMPLING_SMOTE: str = "smote"
VALID_SAMPLING: list[str] = [SAMPLING_NONE, SAMPLING_UNDERSAMPLE, SAMPLING_SMOTE]

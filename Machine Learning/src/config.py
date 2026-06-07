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
# Tuner sub-sampling (speed optimisation)
# ──────────────────────────────────────────────
# Models with O(n²) or higher complexity get sub-sampled during tuning.
# The final train step always uses 100% of SMOTE data.
SAMPLE_FRACTION_KNN_SVM: float = 0.10   # KNN / SVM use 10%  (~45k rows)
SAMPLE_FRACTION_NN: float = 0.30        # Neural Network uses 30% (~68k rows)

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
    "neural_network",
]

# ──────────────────────────────────────────────
# Tuner defaults
# ──────────────────────────────────────────────
TUNER_N_TRIALS: int = 30
TUNER_CV_FOLDS: int = 5

# ──────────────────────────────────────────────
# Sampling strategy names (for CLI / pipeline)
# ──────────────────────────────────────────────
SAMPLING_NONE: str = "none"
SAMPLING_UNDERSAMPLE: str = "undersample"
SAMPLING_SMOTE: str = "smote"
VALID_SAMPLING: list[str] = [SAMPLING_NONE, SAMPLING_UNDERSAMPLE, SAMPLING_SMOTE]

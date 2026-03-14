"""Project-wide paths and constants.

All other modules should import paths from here rather than
constructing them inline, so that the project works regardless of
the current working directory.
"""

from pathlib import Path
import os

import numpy as np

# ---------------------------------------------------------------------------
# Root directories
# ---------------------------------------------------------------------------

REPO_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = REPO_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"

CONFIG_DIR: Path = REPO_DIR / "config"
SCHEMA_PATH: Path = CONFIG_DIR / "schema.yaml"

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

ID_COLUMN: str = "id"
TARGET_COLUMN: str = "Heart Disease"

CATEGORICAL_COLUMNS: list[str] = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

NUMERICAL_COLUMNS: list[str] = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]

# Raw integer → cleaned string label mappings (applied in dataset.py)
BINARY_MAPPINGS: dict[str, dict[int, str]] = {
    "Sex": {1: "male", 0: "female"},
    "FBS over 120": {1: "true", 0: "false"},
    "Exercise angina": {1: "yes", 0: "no"},
}

NEGATIVE_TARGET_LABEL: str = "Absence"
POSITIVE_TARGET_LABEL: str = "Presence"

TARGET_LABEL_TO_CODE: dict[str, int] = {
    NEGATIVE_TARGET_LABEL: 0,
    POSITIVE_TARGET_LABEL: 1,
}

TARGET_VALUE_TO_LABEL: dict[str | int, str] = {
    NEGATIVE_TARGET_LABEL: NEGATIVE_TARGET_LABEL,
    POSITIVE_TARGET_LABEL: POSITIVE_TARGET_LABEL,
    0: NEGATIVE_TARGET_LABEL,
    1: POSITIVE_TARGET_LABEL,
}

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_SPLITS: int = 10

# ---------------------------------------------------------------------------
# Training pipeline configuration
# ---------------------------------------------------------------------------

# Input data path for training pipeline
INPUT_FILE: Path = DATA_RAW_DIR / "heart_disease.csv" 
PREDICTIONS_DB_PATH: Path = REPO_DIR / "predictions.db"

# MLflow
MLFLOW_TRACKING_DB_PATH: Path = REPO_DIR / "mlflow.db"
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_TRACKING_DB_PATH.resolve().as_posix()}",
)
MLFLOW_EXPERIMENT_NAME: str = "heart_disease"
MLFLOW_MODEL_NAME: str = "heart_disease_model"
MLFLOW_ARTIFACT_PATH: str = "model"
MLFLOW_ACTIVE_ALIAS: str = "active"

# Hyperparameter search space for Logistic Regression
HYPERPARAMETER_GRID = {
    "classifier__C": np.logspace(-4, 2, 7),
    "classifier__class_weight": [None, "balanced"],
    "classifier__solver": ["lbfgs"],
}

# Logistic Regression default settings
LOGISTIC_MAX_ITER: int = 1000

# RandomizedSearchCV settings
DEFAULT_N_ITER: int = 50
DEFAULT_CV_FOLDS: int = 5
SCORING_METRIC: str = "roc_auc"
N_JOBS: int = -1

# ---------------------------------------------------------------------------
# Monitoring and risk thresholds
# ---------------------------------------------------------------------------

RISK_LEVEL_THRESHOLDS_PCT: dict[str, int] = {
    "low": 20,
    "moderate": 40,
    "high": 60,
    "very_high": 80,
}

HIGH_RISK_PROBABILITY_THRESHOLD: float = RISK_LEVEL_THRESHOLDS_PCT["high"] / 100.0

MIN_PREDICTIONS_FOR_DRIFT: int = 20
DRIFT_KS_P_THRESHOLD: float = 0.05
DRIFT_TV_THRESHOLD: float = 0.20
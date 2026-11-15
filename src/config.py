"""Central configuration, data contracts, and hyperparameters."""

from pathlib import Path
from typing import Any

# Directory Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"

# Reproducibility
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20
SYNTHETIC_SAMPLES: int = 1200

# Data Contract Columns
ID_COLUMN: str = "User_ID"
TARGET_COLUMN: str = "Dominant_Emotion"

NUMERIC_FEATURES: list[str] = [
    "Age",
    "Daily_Usage_Time (minutes)",
    "Posts_Per_Day",
    "Likes_Received_Per_Day",
    "Comments_Received_Per_Day",
    "Messages_Sent_Per_Day",
]

CATEGORICAL_FEATURES: list[str] = [
    "Gender",
    "Platform",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid Categorical & Target Domains
VALID_GENDERS: list[str] = ["Female", "Male", "Non-binary"]

VALID_PLATFORMS: list[str] = [
    "Instagram",
    "Twitter",
    "Facebook",
    "LinkedIn",
    "Snapchat",
    "Whatsapp",
    "Telegram",
]

VALID_EMOTIONS: list[str] = [
    "Happiness",
    "Sadness",
    "Anger",
    "Anxiety",
    "Boredom",
    "Neutral",
]

# Model Hyperparameters (ML vs. DL)
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

MLP_PARAMS: dict[str, Any] = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.001,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "max_iter": 300,
    "random_state": RANDOM_SEED,
}

# Evaluation Gate
QUALITY_GATE_MACRO_F1: float = 0.75

# Artifact Paths
RF_MODEL_PATH: Path = MODELS_DIR / "rf_pipeline.joblib"
MLP_MODEL_PATH: Path = MODELS_DIR / "mlp_pipeline.joblib"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

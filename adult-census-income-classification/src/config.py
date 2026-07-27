"""Central configuration: paths, constants, and column definitions."""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "adult.data"
TEST_DATA_PATH = DATA_DIR / "adult.test"
PROCESSED_DATA_PATH = DATA_DIR / "adult_processed.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

# --- Source URLs (UCI Machine Learning Repository) ----------------------
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# --- Column definitions --------------------------------------------------
COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

TARGET_COLUMN = "income"

NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

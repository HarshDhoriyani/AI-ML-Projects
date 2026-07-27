"""Train a classifier on the Adult Census Income dataset and persist it to disk."""

import argparse
import logging

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_train_data
from src.preprocessing import build_preprocessing_pipeline, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=config.RANDOM_STATE, n_jobs=-1
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=config.RANDOM_STATE),
}


def train_model(model_name: str = "random_forest") -> Pipeline:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}"
        )

    logger.info("Loading data...")
    df = load_train_data()
    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    logger.info("Building pipeline with model=%s", model_name)
    preprocessor = build_preprocessing_pipeline()
    classifier = MODEL_REGISTRY[model_name]()

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    logger.info("Fitting pipeline on %d training rows...", len(X_train))
    pipeline.fit(X_train, y_train)

    val_accuracy = pipeline.score(X_val, y_val)
    logger.info("Validation accuracy: %.4f", val_accuracy)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    logger.info("Saved trained pipeline to %s", config.MODEL_PATH)

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train the Adult Census Income model")
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model to train",
    )
    args = parser.parse_args()
    train_model(args.model)


if __name__ == "__main__":
    main()

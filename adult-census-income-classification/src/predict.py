"""Run inference with a trained pipeline on new, unlabeled data."""

import argparse
import logging

import joblib
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def predict(input_csv: str, output_csv: str) -> pd.DataFrame:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. Run `python main.py train` first."
        )

    pipeline = joblib.load(config.MODEL_PATH)

    df = pd.read_csv(input_csv)
    missing = set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    preds = pipeline.predict(df)
    probs = pipeline.predict_proba(df)[:, 1]

    result = df.copy()
    result["predicted_income"] = ["<=50K" if p == 0 else ">50K" for p in preds]
    result["probability_gt_50k"] = probs

    result.to_csv(output_csv, index=False)
    logger.info("Wrote %d predictions to %s", len(result), output_csv)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict income class on new data")
    parser.add_argument("--input", required=True, help="Path to input CSV (14 feature columns)")
    parser.add_argument("--output", default="predictions.csv", help="Path to write predictions CSV")
    args = parser.parse_args()
    predict(args.input, args.output)


if __name__ == "__main__":
    main()

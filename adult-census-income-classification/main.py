"""CLI entry point for the Adult Census Income Classification project.

Usage:
    python main.py train --model random_forest
    python main.py evaluate
    python main.py predict --input new_data.csv --output predictions.csv
"""

import argparse
import sys

from src.evaluate import evaluate_model
from src.predict import predict
from src.train import MODEL_REGISTRY, train_model


def main():
    parser = argparse.ArgumentParser(description="Adult Census Income Classification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", default="random_forest", choices=list(MODEL_REGISTRY.keys())
    )

    subparsers.add_parser("evaluate", help="Evaluate the saved model on the test set")

    predict_parser = subparsers.add_parser("predict", help="Predict on new data")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", default="predictions.csv")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.model)
    elif args.command == "evaluate":
        evaluate_model()
    elif args.command == "predict":
        predict(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

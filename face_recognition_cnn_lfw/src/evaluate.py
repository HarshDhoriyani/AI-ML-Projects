"""
evaluate.py
-----------
Loads a trained model + the saved test split, reports accuracy,
precision/recall/F1 per identity, a confusion matrix, and a sample
prediction gallery.

Usage:
    python src/evaluate.py --model-dir ../models
"""

import os
import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

from utils import plot_confusion_matrix, plot_sample_predictions


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained LFW CNN")
    p.add_argument("--model-dir", type=str, default="../models")
    p.add_argument("--model-file", type=str, default="best_model.keras")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = os.path.join(args.model_dir, args.model_file)
    labels_path = os.path.join(args.model_dir, "labels.json")
    split_path = os.path.join(args.model_dir, "test_split.npz")

    model = load_model(model_path)
    with open(labels_path) as f:
        labels_map = json.load(f)
    class_names = [labels_map[str(i)] for i in range(len(labels_map))]

    split = np.load(split_path)
    X_test, y_test_int = split["X_test"], split["y_test_int"]

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test_int, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test_int, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test_int, y_pred, class_names,
                           save_path=os.path.join(args.model_dir, "confusion_matrix.png"))
    plot_sample_predictions(X_test, y_test_int, y_pred, class_names,
                             save_path=os.path.join(args.model_dir, "sample_predictions.png"))

    print(f"\nPlots saved in: {os.path.abspath(args.model_dir)}")


if __name__ == "__main__":
    main()

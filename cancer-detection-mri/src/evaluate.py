"""
evaluate.py
Evaluate a trained model on the Testing set: prints classification report
and saves a confusion matrix image.

Usage:
    python src/evaluate.py --data_dir data --model_path models/best_model.h5
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from data_loader import get_datasets, CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MRI tumor classifier")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_path", type=str, default="models/best_model.h5")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--assets_dir", type=str, default="assets")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.assets_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} ...")
    model = keras.models.load_model(args.model_path)

    print(f"Loading test data from {args.data_dir} ...")
    _, _, test_ds = get_datasets(args.data_dir, batch_size=args.batch_size, augment_train=False)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(args.assets_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()

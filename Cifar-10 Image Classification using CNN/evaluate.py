"""
Evaluate a trained CIFAR-10 CNN: test accuracy, classification report,
confusion matrix, and a grid of sample predictions.

Usage:
    python evaluate.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from model import CLASS_NAMES, NUM_CLASSES

MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.keras")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
SAMPLE_PREDICTIONS_PATH = os.path.join(MODEL_DIR, "sample_predictions.png")


def load_test_data():
    (_, _), (x_test, y_test_raw) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = to_categorical(y_test_raw, NUM_CLASSES)
    return x_test, y_test, y_test_raw.flatten()


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved confusion matrix to {path}")


def plot_sample_predictions(x_test, y_true, y_pred, class_names, path, n=16):
    idxs = np.random.choice(len(x_test), n, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for ax, idx in zip(axes.flat, idxs):
        ax.imshow(x_test[idx])
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved sample predictions to {path}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train.py first."
        )

    print("Loading test data...")
    x_test, y_test, y_test_labels = load_test_data()

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Evaluating...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\nClassification report:")
    print(classification_report(y_test_labels, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test_labels, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)
    plot_sample_predictions(x_test, y_test_labels, y_pred, CLASS_NAMES, SAMPLE_PREDICTIONS_PATH)


if __name__ == "__main__":
    main()

"""
utils.py
--------
Helper functions: data augmentation pipeline, training curve plots,
confusion matrix plot, and a gallery viewer for sample predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


def get_augmenter():
    """Light augmentation appropriate for faces (no vertical flips!)."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_sample_predictions(X, y_true, y_pred, class_names, n=12, save_path=None):
    n = min(n, len(X))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        ax = axes[i]
        ax.imshow(X[i].squeeze(), cmap="gray")
        true_name = class_names[y_true[i]]
        pred_name = class_names[y_pred[i]]
        color = "green" if true_name == pred_name else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", color=color, fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved sample predictions to {save_path}")
    plt.close(fig)

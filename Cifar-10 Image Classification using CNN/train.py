"""
Train a CNN on CIFAR-10.

Usage:
    python train.py --epochs 25 --batch-size 64
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from model import build_cnn, NUM_CLASSES

MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn.keras")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")


def load_data():
    """Loads and preprocesses CIFAR-10: normalizes pixels, one-hot encodes labels."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test)


def get_augmenter():
    """Light data augmentation to improve generalization."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved training curves to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = load_data()

    # Carve out a validation set from training data
    n_val = int(len(x_train) * args.val_split)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    augmenter = get_augmenter()
    augmenter.fit(x_train)

    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=len(x_train) // args.batch_size,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, HISTORY_PLOT_PATH)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    print(f"\nBest model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()

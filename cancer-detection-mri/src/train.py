"""
train.py
Train the brain tumor MRI classifier.

Usage:
    python src/train.py --data_dir data --epochs 25 --batch_size 32 --model_type cnn
    python src/train.py --data_dir data --epochs 15 --batch_size 32 --model_type transfer
"""

import argparse
import os
import matplotlib.pyplot as plt
from tensorflow import keras

from data_loader import get_datasets, CLASS_NAMES
from model import build_cnn, build_transfer, compile_model


def parse_args():
    p = argparse.ArgumentParser(description="Train MRI tumor classifier")
    p.add_argument("--data_dir", type=str, default="data", help="Path to dataset root (contains Training/Testing)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--model_type", type=str, default="cnn", choices=["cnn", "transfer"])
    p.add_argument("--output_dir", type=str, default="models")
    p.add_argument("--assets_dir", type=str, default="assets")
    return p.parse_args()


def plot_history(history, out_path):
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
    fig.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.assets_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    train_ds, val_ds, test_ds = get_datasets(args.data_dir, batch_size=args.batch_size)

    print(f"Building model ({args.model_type}) ...")
    if args.model_type == "cnn":
        model = build_cnn(num_classes=len(CLASS_NAMES))
    else:
        model = build_transfer(num_classes=len(CLASS_NAMES))

    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    checkpoint_path = os.path.join(args.output_dir, "best_model.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, os.path.join(args.assets_dir, "training_curves.png"))

    print("\nEvaluating on held-out test set ...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f} | precision: {test_prec:.4f} | recall: {test_rec:.4f}")

    print(f"\nBest model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

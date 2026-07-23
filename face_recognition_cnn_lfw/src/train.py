"""
train.py
--------
Trains the CNN on the LFW dataset and saves the best model + label map +
training curve plot.

Usage:
    python src/train.py --min-faces 70 --epochs 40 --batch-size 32
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from data_loader import load_lfw_data
from model import build_cnn
from utils import get_augmenter, plot_training_history


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN face recognizer on LFW")
    p.add_argument("--min-faces", type=int, default=70,
                    help="Minimum images per identity to include as a class")
    p.add_argument("--resize", type=float, default=0.5, help="LFW image resize factor")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", type=str, default="../models")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Data
    data = load_lfw_data(min_faces_per_person=args.min_faces, resize=args.resize, seed=args.seed)

    # 2. Model
    model = build_cnn(data["image_shape"], data["num_classes"])
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Class weights (LFW is imbalanced -- some people have far more photos)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(data["y_train_int"]),
        y=data["y_train_int"],
    )
    class_weight_dict = dict(enumerate(class_weights))

    # 4. Augmented training generator
    augmenter = get_augmenter()
    train_gen = augmenter.flow(data["X_train"], data["y_train"], batch_size=args.batch_size, seed=args.seed)

    # 5. Callbacks
    ckpt_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # 6. Train
    history = model.fit(
        train_gen,
        steps_per_epoch=max(1, len(data["X_train"]) // args.batch_size),
        validation_data=(data["X_val"], data["y_val"]),
        epochs=args.epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    # 7. Save final artifacts
    model.save(os.path.join(args.output_dir, "final_model.keras"))
    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump({str(i): name for i, name in enumerate(data["target_names"])}, f, indent=2)

    plot_training_history(history, save_path=os.path.join(args.output_dir, "training_curves.png"))

    # 8. Quick test-set eval
    test_loss, test_acc = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    # Save test split so evaluate.py can reuse the exact same split
    np.savez(
        os.path.join(args.output_dir, "test_split.npz"),
        X_test=data["X_test"], y_test_int=data["y_test_int"],
    )

    print(f"\nArtifacts saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

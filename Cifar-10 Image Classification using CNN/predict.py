"""
Run inference on a single custom image using the trained CIFAR-10 CNN.

Usage:
    python predict.py --image path/to/image.jpg
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from model import CLASS_NAMES

MODEL_PATH = os.path.join("saved_models", "cifar10_cnn.keras")


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Classify a single image with the CIFAR-10 CNN")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    x = preprocess_image(args.image)

    probs = model.predict(x, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][: args.top_k]

    print(f"\nPredictions for {args.image}:")
    for idx in top_indices:
        print(f"  {CLASS_NAMES[idx]:<12s} {probs[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()

"""
predict.py
Run inference on a single MRI image.

Usage:
    python src/predict.py --image_path path/to/scan.jpg --model_path models/best_model.h5
"""

import argparse

import numpy as np
from PIL import Image
from tensorflow import keras

from data_loader import IMG_SIZE, CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Predict tumor class for a single MRI image")
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--model_path", type=str, default="models/best_model.h5")
    return p.parse_args()


def load_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image_path):
    x = load_and_preprocess(image_path)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), dict(zip(CLASS_NAMES, probs.tolist()))


def main():
    args = parse_args()
    model = keras.models.load_model(args.model_path)
    label, confidence, all_probs = predict(model, args.image_path)

    print(f"\nPrediction: {label}  (confidence: {confidence:.2%})\n")
    print("Full probability breakdown:")
    for cls, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:12s}: {p:.2%}")


if __name__ == "__main__":
    main()

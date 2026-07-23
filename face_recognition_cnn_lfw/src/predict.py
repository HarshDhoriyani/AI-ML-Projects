"""
predict.py
----------
Runs inference on a single face image (already cropped to a face, ideally
grayscale). Uses OpenCV's Haar cascade to auto-detect+crop a face if the
input is a full photo rather than a pre-cropped face.

Usage:
    python src/predict.py --image path/to/photo.jpg --model-dir ../models
"""

import os
import argparse
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model


def detect_and_crop_face(gray_img):
    """Uses OpenCV's built-in Haar cascade to find and crop the largest face."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return gray_img  # fall back to full image
    # pick largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray_img[y:y + h, x:x + w]


def parse_args():
    p = argparse.ArgumentParser(description="Predict identity for a single image")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--model-dir", type=str, default="../models")
    p.add_argument("--model-file", type=str, default="best_model.keras")
    p.add_argument("--top-k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()

    model = load_model(os.path.join(args.model_dir, args.model_file))
    with open(os.path.join(args.model_dir, "labels.json")) as f:
        labels_map = json.load(f)

    input_shape = model.input_shape[1:3]  # (H, W)

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {args.image}")

    face = detect_and_crop_face(img)
    face = cv2.resize(face, (input_shape[1], input_shape[0]))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=(0, -1))  # (1, H, W, 1)

    probs = model.predict(face, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:args.top_k]

    print(f"\nPredictions for {args.image}:")
    for idx in top_idx:
        name = labels_map[str(idx)]
        print(f"  {name}: {probs[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()

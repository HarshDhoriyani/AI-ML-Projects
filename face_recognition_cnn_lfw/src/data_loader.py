"""
data_loader.py
--------------
Loads the "Labeled Faces in the Wild" (LFW) dataset using scikit-learn,
preprocesses images for a CNN (resize, normalize, reshape, one-hot encode
labels) and returns train/val/test splits.

LFW is a "wild" dataset: photos are uncontrolled/unposed, taken from news
and web sources with varying pose, lighting, expression and occlusion --
which is what makes it a good benchmark for real-world face recognition.
"""

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_lfw_data(min_faces_per_person=70, resize=0.5, test_size=0.2, val_size=0.1, seed=42):
    """
    Downloads (and caches) the LFW people dataset and prepares it for training.

    Parameters
    ----------
    min_faces_per_person : int
        Only keep identities (classes) with at least this many photos.
        Higher values -> fewer classes, more images per class, easier task.
        Lower values -> more classes (harder, more "in the wild" realism).
    resize : float
        Scale factor applied to the native LFW image size (125x94 -> resize).
    test_size : float
        Fraction of data held out for the test set.
    val_size : float
        Fraction of the remaining (train) data held out for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                     y_train_int, y_val_int, y_test_int,  # non one-hot labels
                     target_names, num_classes, image_shape
    """
    print("Fetching LFW dataset (this downloads ~200MB the first time)...")
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                            resize=resize,
                            color=False,
                            funneled=True)

    n_samples, h, w = lfw.images.shape
    X = lfw.images.reshape(n_samples, h, w, 1).astype("float32") / 255.0
    y_int = lfw.target
    target_names = lfw.target_names
    num_classes = len(target_names)

    print(f"Dataset loaded: {n_samples} images, {num_classes} identities, "
          f"image size {h}x{w}")
    for i, name in enumerate(target_names):
        count = np.sum(y_int == i)
        print(f"  - {name}: {count} images")

    # Stratified split so every identity is represented in train/val/test
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int, test_size=test_size, stratify=y_int, random_state=seed
    )
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X_train, y_train_int, test_size=val_size, stratify=y_train_int, random_state=seed
    )

    y_train = to_categorical(y_train_int, num_classes)
    y_val = to_categorical(y_val_int, num_classes)
    y_test = to_categorical(y_test_int, num_classes)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "y_train_int": y_train_int, "y_val_int": y_val_int, "y_test_int": y_test_int,
        "target_names": target_names,
        "num_classes": num_classes,
        "image_shape": (h, w, 1),
    }


if __name__ == "__main__":
    data = load_lfw_data()
    print("\nShapes:")
    print("  X_train:", data["X_train"].shape)
    print("  X_val:  ", data["X_val"].shape)
    print("  X_test: ", data["X_test"].shape)
    print("  Classes:", data["num_classes"])

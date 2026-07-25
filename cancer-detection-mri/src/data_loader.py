"""
data_loader.py
Builds tf.data pipelines for the Brain Tumor MRI dataset.

Expected directory layout:
    data_dir/Training/<class_name>/*.jpg
    data_dir/Testing/<class_name>/*.jpg
"""

import os
import tensorflow as tf

IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def _build_dataset(directory, batch_size, shuffle, seed=42):
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Expected directory not found: {directory}\n"
            "Did you download the dataset? See data/README.md for instructions."
        )
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=shuffle,
        seed=seed,
    )
    return ds


def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def _normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def get_datasets(data_dir, batch_size=32, augment_train=True, val_split=0.15, seed=42):
    """
    Returns (train_ds, val_ds, test_ds) as batched, prefetched tf.data.Dataset objects.

    A validation split is carved out of the Training/ folder; Testing/ is
    kept fully held-out for final evaluation.
    """
    train_dir = os.path.join(data_dir, "Training")
    test_dir = os.path.join(data_dir, "Testing")

    full_train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        validation_split=val_split,
        subset="training",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMG_SIZE,
        validation_split=val_split,
        subset="validation",
        seed=seed,
    )
    test_ds = _build_dataset(test_dir, batch_size=batch_size, shuffle=False, seed=seed)

    AUTOTUNE = tf.data.AUTOTUNE

    if augment_train:
        full_train_ds = full_train_ds.map(_augment, num_parallel_calls=AUTOTUNE)

    full_train_ds = full_train_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(_normalize, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return full_train_ds, val_ds, test_ds

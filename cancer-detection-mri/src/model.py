"""
model.py
Defines two model architectures for MRI tumor classification:
  1. build_cnn()      - a lightweight custom CNN trained from scratch
  2. build_transfer()  - EfficientNetB0 pretrained on ImageNet, fine-tuned
"""

from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
NUM_CLASSES = 4


def build_cnn(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """A simple 4-block CNN. Good baseline, trains fast on CPU/GPU."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mri_cnn",
    )
    return model


def build_transfer(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, fine_tune_at=100, weights="imagenet"):
    """
    EfficientNetB0 backbone pretrained on ImageNet + custom classification head.
    Set fine_tune_at=None to freeze the whole backbone (feature extraction only),
    or an integer to unfreeze layers from that index onward for fine-tuning.
    Set weights=None to build with random init (useful for offline unit tests).
    """
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg",
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = keras.Input(shape=input_shape)
    # EfficientNet expects 0-255 range internally; our pipeline normalizes to
    # [0,1], so we rescale back up before the preprocessing layer.
    x = layers.Rescaling(255.0)(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mri_efficientnet")
    return model


def compile_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model

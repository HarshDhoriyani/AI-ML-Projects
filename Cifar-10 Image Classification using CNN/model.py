"""
CNN architecture for CIFAR-10 image classification.
"""

from tensorflow.keras import layers, models, regularizers


NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, weight_decay=1e-4):
    """
    Builds a CNN for CIFAR-10 classification.

    Architecture:
      3 convolutional blocks (each: Conv -> BN -> Conv -> BN -> MaxPool -> Dropout)
      with increasing filter depth (32 -> 64 -> 128), followed by a dense
      classification head.

    Args:
        input_shape: shape of input images, default (32, 32, 3).
        num_classes: number of output classes, default 10.
        weight_decay: L2 regularization factor applied to conv/dense kernels.

    Returns:
        A compiled tf.keras.Model.
    """
    reg = regularizers.l2(weight_decay)

    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Classification head
    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cifar10_cnn")
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()

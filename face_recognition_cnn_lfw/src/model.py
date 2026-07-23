"""
model.py
--------
CNN architecture for face recognition on LFW.

The network is intentionally kept moderate in size since LFW subsets
(after filtering by min_faces_per_person) are small -- a very deep net
would overfit quickly. Batch norm + dropout + light augmentation keep
generalization reasonable.
"""

from tensorflow.keras import layers, models, regularizers


def build_cnn(input_shape, num_classes, l2_reg=1e-4):
    """
    Builds a CNN for multi-class face identification.

    Parameters
    ----------
    input_shape : tuple (H, W, C)
    num_classes : int
    l2_reg : float
        L2 weight regularization strength.

    Returns
    -------
    tf.keras.Model (uncompiled)
    """
    reg = regularizers.l2(l2_reg)

    model = models.Sequential(name="lfw_face_cnn")
    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


if __name__ == "__main__":
    m = build_cnn((62, 47, 1), num_classes=7)
    m.summary()

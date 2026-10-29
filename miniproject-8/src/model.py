"""Convolutional network used for digit recognition."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(num_classes: int = 10) -> keras.Model:
    """A compact CNN: three conv blocks, dropout, softmax output."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="neuralvision_cnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

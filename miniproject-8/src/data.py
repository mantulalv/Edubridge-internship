"""MNIST loading and preprocessing for uploaded digit images."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 28
DIGIT_BOX = 20


def load_mnist():
    """Return ((x_train, y_train), (x_test, y_test)) with float32 inputs in [0, 1]."""
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0
    return (x_train, y_train), (x_test, y_test)


def _crop_to_digit(array: np.ndarray) -> np.ndarray:
    """Crop away empty borders around the drawn digit."""
    mask = array > 30
    if not mask.any():
        return array
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return array[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert an arbitrary image into an MNIST-shaped ``(28, 28, 1)`` array.

    Steps: grayscale -> auto-invert light backgrounds -> crop to the digit ->
    resize the longest side to 20 px -> centre on a black 28x28 canvas -> scale.
    """
    gray = ImageOps.grayscale(image)
    array = np.asarray(gray, dtype="float32")
    if array.size == 0:
        raise ValueError("image has no pixels")

    # MNIST digits are white ink on black; uploaded photos are usually the reverse.
    if float(array.mean()) > 127:
        array = 255.0 - array

    array = _crop_to_digit(array)
    height, width = array.shape
    scale = DIGIT_BOX / max(height, width)
    resized = Image.fromarray(array.astype("uint8")).resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    canvas.paste(
        resized,
        ((IMG_SIZE - resized.width) // 2, (IMG_SIZE - resized.height) // 2),
    )
    return (np.asarray(canvas, dtype="float32") / 255.0)[..., None]

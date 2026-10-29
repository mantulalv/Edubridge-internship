"""Tests for image preprocessing (no TensorFlow needed)."""
import numpy as np
from PIL import Image

from src.data import prepare_image


def test_output_shape_and_range():
    array = np.zeros((120, 90), dtype="uint8")
    array[30:90, 25:65] = 255
    processed = prepare_image(Image.fromarray(array))

    assert processed.shape == (28, 28, 1)
    assert processed.dtype == np.float32
    assert 0.0 <= float(processed.min())
    assert float(processed.max()) <= 1.0
    assert float(processed.mean()) > 0.01


def test_light_background_is_inverted():
    array = np.full((100, 100), 255, dtype="uint8")
    array[25:75, 35:65] = 0
    processed = prepare_image(Image.fromarray(array))

    assert processed.shape == (28, 28, 1)
    assert float(processed.mean()) > 0.05
    assert float(processed.max()) > 0.5


def test_blank_image_is_all_black():
    processed = prepare_image(Image.new("L", (64, 64), color=0))
    assert processed.shape == (28, 28, 1)
    assert float(processed.max()) == 0.0

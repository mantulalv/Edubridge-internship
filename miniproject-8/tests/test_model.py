"""Model tests, skipped automatically when TensorFlow is unavailable."""
import pytest

pytest.importorskip("tensorflow")

from src.model import build_cnn  # noqa: E402


def test_build_cnn_shapes():
    model = build_cnn()
    assert model.input_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, 10)


def test_forward_pass_probabilities():
    import numpy as np

    model = build_cnn()
    batch = np.zeros((2, 28, 28, 1), dtype="float32")
    probabilities = model.predict(batch, verbose=0)
    assert probabilities.shape == (2, 10)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

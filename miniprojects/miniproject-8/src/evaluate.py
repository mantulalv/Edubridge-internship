"""Evaluate the trained CNN and save a confusion matrix."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report  # noqa: E402
from tensorflow import keras  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data import load_mnist  # noqa: E402

ROOT = CURRENT_DIR.parent
MODEL_PATH = ROOT / "models" / "digit_cnn.keras"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"missing {MODEL_PATH}; run `python src/train.py` first"
        )

    model = keras.models.load_model(MODEL_PATH)
    (_, _), (x_test, y_test) = load_mnist()

    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float((y_pred == y_test).mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test_accuracy"] = round(accuracy, 4)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, cmap="Blues", colorbar=False, ax=axis
    )
    axis.set_title(f"MNIST confusion matrix — accuracy {accuracy:.2%}")
    figure.tight_layout()
    figure.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    print(f"test accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test, y_pred, digits=3, zero_division=0
        )
    )


if __name__ == "__main__":
    main()

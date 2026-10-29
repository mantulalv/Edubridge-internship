"""Train the MNIST CNN and write metrics plus training curves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data import load_mnist  # noqa: E402
from model import build_cnn  # noqa: E402

from tensorflow import keras  # noqa: E402

ROOT = CURRENT_DIR.parent
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "digit_cnn.keras"


def plot_history(history: keras.callbacks.History, path: Path) -> None:
    """Save accuracy/loss curves side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0.9, 1.0)
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(epochs: int = 8, batch_size: int = 128) -> None:
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_history(history, OUT_DIR / "training_curves.png")

    metrics = {
        "test_accuracy": round(float(test_accuracy), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs_run": len(history.history["loss"]),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model": MODEL_PATH.name,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()

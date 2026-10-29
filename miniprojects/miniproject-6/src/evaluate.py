"""Evaluate the saved model on the held-out test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import load_bundle  # noqa: E402
from preprocessing import normalize_series  # noqa: E402

ROOT = CURRENT_DIR.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    bundle = load_bundle(MODEL_PATH)
    pipeline = bundle["pipeline"]

    test_path = PROC_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"missing {test_path}; run `python data/data_loader.py` first"
        )

    frame = pd.read_csv(test_path)
    x_test = normalize_series(frame["message"])
    y_test = frame["target"].astype(int)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "model": bundle.get("kind"),
        "rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_test, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0
    )
    payload = {"test": metrics, "report": report}
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["ham", "spam"], cmap="Blues", ax=axis
    )
    figure.tight_layout()
    figure.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    print(json.dumps(metrics, indent=2))
    print(f"wrote {OUT_DIR / 'metrics.json'} and {OUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()

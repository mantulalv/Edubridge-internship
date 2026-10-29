"""Evaluate the premium classifier on the held-out test split."""
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
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_bundle  # noqa: E402
from src.preprocessing import TARGET, feature_frame  # noqa: E402

PROC_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "premium_model.joblib"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    bundle = load_bundle(MODEL_PATH)
    pipeline = bundle["pipeline"]

    test_path = PROC_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"missing {test_path}; run `python data/data_loader.py` first"
        )

    test = pd.read_csv(test_path)
    x_test, y_test = feature_frame(test), test[TARGET].astype(int)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "rows": int(len(y_test)),
        "threshold": round(float(bundle.get("threshold", 0.0)), 0),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_test,
        predictions,
        target_names=["value", "premium"],
        output_dict=True,
        zero_division=0,
    )
    payload = {
        "test": metrics,
        "report": report,
        "selected_C": bundle.get("C"),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["value", "premium"],
        cmap="Greens",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix")

    RocCurveDisplay.from_predictions(
        y_test, probabilities, ax=axes[1]
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[1].set_title("ROC curve")

    figure.tight_layout()
    figure.savefig(OUT_DIR / "evaluation.png", dpi=150)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))
    print(
        classification_report(
            y_test, predictions, target_names=["value", "premium"], zero_division=0
        )
    )
    print(f"wrote {OUT_DIR / 'metrics.json'} and {OUT_DIR / 'evaluation.png'}")


if __name__ == "__main__":
    main()

"""Train the NewsLens classifier and save metrics/plots."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import build_pipeline, save_bundle  # noqa: E402
from preprocessing import clean_series  # noqa: E402

ROOT = CURRENT_DIR.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    train_path = PROC_DIR / "train.csv"
    test_path = PROC_DIR / "test.csv"
    labels_path = PROC_DIR / "labels.json"
    for path in (train_path, test_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `python data/data_loader.py` first"
            )

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    x_train = clean_series(train["text"])
    y_train = train["target"].astype(int)
    x_test = clean_series(test["text"])
    y_test = test["target"].astype(int)

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(
        {"pipeline": pipeline, "labels": labels},
        MODEL_DIR / "news_model.joblib",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "classes": len(labels),
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "report": report}, indent=2),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, y_pred, normalize="true")
    figure, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(matrix, cmap="Purples", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Row-normalised confusion matrix — accuracy {accuracy:.2%}")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))
    print(
        classification_report(
            y_test, y_pred, target_names=labels, digits=2, zero_division=0
        )
    )
    print(f"saved model to {MODEL_DIR / 'news_model.joblib'}")


if __name__ == "__main__":
    main()

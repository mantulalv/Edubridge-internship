"""Train both candidate models, pick the best on validation F1 and save it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import build_pipeline, save_bundle  # noqa: E402
from preprocessing import normalize_series  # noqa: E402

ROOT = CURRENT_DIR.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"


def load_split(name: str) -> tuple[pd.Series, pd.Series]:
    path = PROC_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run `python data/data_loader.py` first"
        )
    frame = pd.read_csv(path)
    return normalize_series(frame["message"]), frame["target"].astype(int)


def score(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def main() -> None:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")

    results: dict[str, dict[str, float]] = {}
    best_kind = ""
    best_pipeline = None
    best_f1 = -1.0

    for kind in ("nb", "lr"):
        pipeline = build_pipeline(kind)
        pipeline.fit(x_train, y_train)
        metrics = score(y_val, pipeline.predict(x_val))
        results[kind] = metrics
        print(f"{kind}: {metrics}")
        if metrics["f1"] > best_f1:
            best_kind, best_pipeline, best_f1 = kind, pipeline, metrics["f1"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(
        {"pipeline": best_pipeline, "kind": best_kind, "validation": results[best_kind]},
        MODEL_DIR / "spam_model.joblib",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected": best_kind,
        "validation": results,
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
    }
    (OUT_DIR / "training_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"selected {best_kind}; saved model to {MODEL_DIR / 'spam_model.joblib'}")


if __name__ == "__main__":
    main()

"""Tune C, train the premium-home classifier and save artefacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CURRENT_DIR = Path(__file__).resolve().parent
ROOT = CURRENT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import build_pipeline, save_bundle  # noqa: E402
from src.preprocessing import FEATURES, PRICE_COLUMN, TARGET, feature_frame  # noqa: E402

ROOT = CURRENT_DIR.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def score(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def resolve_threshold(train: pd.DataFrame, val: pd.DataFrame) -> float:
    summary_path = PROC_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(summary["threshold"])
    return float(pd.concat([train, val])[PRICE_COLUMN].median())


def main() -> None:
    train_path = PROC_DIR / "train.csv"
    val_path = PROC_DIR / "val.csv"
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run `python data/data_loader.py` first"
            )

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    x_train, y_train = feature_frame(train), train[TARGET].astype(int)
    x_val, y_val = feature_frame(val), val[TARGET].astype(int)

    results: list[dict[str, float]] = []
    best_key = (-1.0, -1.0)
    best_C = C_GRID[0]
    best_pipeline = None

    for C in C_GRID:
        pipeline = build_pipeline(C)
        pipeline.fit(x_train, y_train)
        metrics = score(y_val, pipeline.predict(x_val))
        results.append({"C": C, **metrics})
        print(f"C={C:<6} {metrics}")

        key = (metrics["f1"], metrics["accuracy"])
        if key > best_key:
            best_key = key
            best_C = C
            best_pipeline = pipeline

    assert best_pipeline is not None
    threshold = resolve_threshold(train, val)
    validation = next(row for row in results if row["C"] == best_C)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_bundle(
        {
            "pipeline": best_pipeline,
            "features": FEATURES,
            "threshold": threshold,
            "C": best_C,
            "validation": validation,
        },
        MODEL_DIR / "premium_model.joblib",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_C": best_C,
        "threshold": threshold,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "grid": results,
    }
    (OUT_DIR / "training_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    coefficients = best_pipeline.named_steps["clf"].coef_[0]
    table = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "coefficient": [round(float(value), 4) for value in coefficients],
                "odds_ratio": [round(float(np.exp(value)), 4) for value in coefficients],
            }
        )
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(OUT_DIR / "coefficients.csv", index=False)

    print(f"selected C={best_C} on validation; threshold = Rs {threshold:,.0f}")
    print(table.to_string(index=False))
    print(f"saved model to {MODEL_DIR / 'premium_model.joblib'}")


if __name__ == "__main__":
    main()

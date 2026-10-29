"""Logistic-regression pipeline, persistence and local explanations."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import FEATURES


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Standardise the five features, then fit logistic regression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no model at {path}; run `python src/train.py` first"
        )
    return joblib.load(path)


def local_contributions(
    pipeline: Pipeline, features: dict[str, float], k: int = 5
) -> list[tuple[str, float]]:
    """Scaled feature × weight for one house, strongest first.

    Positive values push the prediction towards premium, negative towards value.
    """
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["clf"]

    row = pd.DataFrame([[float(features[name]) for name in FEATURES]], columns=FEATURES)
    scaled = scaler.transform(row)[0]
    contributions = scaled * classifier.coef_[0]

    ranked = sorted(
        zip(FEATURES, contributions), key=lambda item: float(item[1]), reverse=True
    )
    return [(name, round(float(value), 4)) for name, value in ranked[:k]]

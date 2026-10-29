"""Tests for the logistic-regression pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.model import (
    build_pipeline,
    load_bundle,
    local_contributions,
    save_bundle,
)
from src.preprocessing import FEATURES, TARGET, add_premium_target, feature_frame


def make_frame(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    area = rng.integers(500, 4000, rows)
    bedrooms = rng.integers(1, 6, rows)
    bathrooms = rng.integers(1, 4, rows)
    age = rng.integers(0, 40, rows)
    distance = rng.uniform(1, 30, rows).round(1)
    price = (
        50_000
        + area * 120
        + bedrooms * 10_000
        + bathrooms * 8_000
        - age * 1_500
        - distance * 2_000
        + rng.normal(0, 15_000, rows)
    ).round(0)
    return pd.DataFrame(
        {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "distance_to_city_km": distance,
            "price": price,
        }
    )


def fitted():
    labelled, _ = add_premium_target(make_frame())
    pipeline = build_pipeline(1.0)
    pipeline.fit(feature_frame(labelled), labelled[TARGET])
    return labelled, pipeline


def test_pipeline_learns_premium_boundary():
    labelled, pipeline = fitted()
    predictions = pipeline.predict(feature_frame(labelled))
    assert float((predictions == labelled[TARGET]).mean()) > 0.85


def test_probabilities_and_contributions():
    labelled, pipeline = fitted()
    row = {name: float(labelled[name].iloc[0]) for name in FEATURES}
    probability = pipeline.predict_proba(feature_frame(pd.DataFrame([row])))[0][1]
    assert 0.0 <= probability <= 1.0

    contributions = local_contributions(pipeline, row)
    assert len(contributions) == len(FEATURES)
    assert all(isinstance(name, str) for name, _ in contributions)


def test_save_and_load_roundtrip(tmp_path):
    labelled, pipeline = fitted()
    path = tmp_path / "model.joblib"
    save_bundle({"pipeline": pipeline, "features": FEATURES}, path)

    bundle = load_bundle(path)
    assert bundle["features"] == FEATURES
    predictions = bundle["pipeline"].predict(feature_frame(labelled.head(3)))
    assert predictions.shape == (3,)


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")

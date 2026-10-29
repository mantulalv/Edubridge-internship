"""Tests for the feature contract and target construction."""
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURES,
    PRICE_COLUMN,
    TARGET,
    add_premium_target,
    feature_frame,
    validate_house_frame,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area_sqft": [800, 1200, 2000, 3000],
            "bedrooms": [2, 2, 3, 4],
            "bathrooms": [1, 2, 2, 3],
            "age_years": [30, 20, 10, 5],
            "distance_to_city_km": [25.0, 15.0, 8.0, 3.0],
            "price": [100_000, 200_000, 300_000, 400_000],
        }
    )


def test_median_threshold_balances_target():
    labelled, threshold = add_premium_target(sample_frame())
    assert threshold == 250_000
    assert labelled[TARGET].tolist() == [0, 0, 1, 1]


def test_custom_threshold():
    labelled, threshold = add_premium_target(sample_frame(), threshold=300_000)
    assert threshold == 300_000
    assert labelled[TARGET].tolist() == [0, 0, 0, 1]


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_house_frame(sample_frame().drop(columns=["area_sqft"]))


def test_missing_values_raise():
    frame = sample_frame().astype({column: float for column in FEATURES})
    frame.loc[0, "area_sqft"] = float("nan")
    with pytest.raises(ValueError):
        validate_house_frame(frame)


def test_feature_frame_selects_and_casts():
    features = feature_frame(sample_frame())
    assert list(features.columns) == FEATURES
    assert all(str(dtype).startswith("float") for dtype in features.dtypes)


def test_feature_frame_rejects_missing_feature():
    with pytest.raises(ValueError):
        feature_frame(sample_frame().drop(columns=["bathrooms"]))


def test_price_column_name_is_stable():
    assert PRICE_COLUMN == "price"

"""Feature contract and premium-target construction."""
from __future__ import annotations

import pandas as pd

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "distance_to_city_km",
]
PRICE_COLUMN = "price"
TARGET = "premium"


def validate_house_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the raw dataset has every feature and the price column."""
    missing = [
        column for column in FEATURES + [PRICE_COLUMN] if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("dataset is empty")
    if frame[FEATURES].isna().any().any():
        raise ValueError("feature columns contain missing values")
    return frame


def add_premium_target(
    frame: pd.DataFrame, threshold: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Add ``premium`` (price above the median) and return the threshold used."""
    validate_house_frame(frame)
    if threshold is None:
        threshold = float(frame[PRICE_COLUMN].median())
    threshold = float(threshold)

    out = frame.copy()
    out[TARGET] = (out[PRICE_COLUMN] > threshold).astype(int)
    return out, threshold


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and cast the model inputs."""
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return frame[FEATURES].astype(float)

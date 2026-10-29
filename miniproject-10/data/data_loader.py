"""Build the house dataset and split it for premium-home classification.

The default data reproduces ``house_data.csv`` from the Edubridge assignment 5
house-price project (same seed and formula). Pass ``--csv`` to use another file
that has the expected columns.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    FEATURES,
    PRICE_COLUMN,
    TARGET,
    add_premium_target,
)

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RAW_PATH = RAW_DIR / "house_data.csv"
SEED = 42
ROWS = 1000


def generate_dataset(rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    """Recreate the assignment-5 sample data (deterministic for seed 42)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "area_sqft": rng.integers(500, 4000, rows),
            "bedrooms": rng.integers(1, 6, rows),
            "bathrooms": rng.integers(1, 4, rows),
            "age_years": rng.integers(0, 40, rows),
            "distance_to_city_km": rng.uniform(1, 30, rows).round(1),
        }
    )
    frame[PRICE_COLUMN] = (
        50_000
        + frame["area_sqft"] * 120
        + frame["bedrooms"] * 10_000
        + frame["bathrooms"] * 8_000
        - frame["age_years"] * 1_500
        - frame["distance_to_city_km"] * 2_000
        + rng.normal(0, 15_000, rows)
    ).round(0)
    return frame


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load a CSV, or generate and cache the default dataset."""
    if csv_path is not None:
        frame = pd.read_csv(csv_path)
        print(f"loaded {len(frame):,} rows from {csv_path}")
        return frame

    if RAW_PATH.exists():
        frame = pd.read_csv(RAW_PATH)
        print(f"loaded cached data from {RAW_PATH}")
        return frame

    frame = generate_dataset()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_PATH, index=False)
    print(f"generated {len(frame):,} rows -> {RAW_PATH}")
    return frame


def split_and_save(frame: pd.DataFrame) -> tuple[dict[str, Path], dict]:
    """Add the binary target and write stratified 70/15/15 splits."""
    labelled, threshold = add_premium_target(frame)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    train, rest = train_test_split(
        labelled, test_size=0.30, stratify=labelled[TARGET], random_state=SEED
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest[TARGET], random_state=SEED
    )

    paths: dict[str, Path] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = PROC_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        paths[name] = path

    summary = {
        "rows": int(len(labelled)),
        "features": FEATURES,
        "threshold": round(threshold, 0),
        "premium": int(labelled[TARGET].sum()),
        "value": int((labelled[TARGET] == 0).sum()),
        "premium_share": round(float(labelled[TARGET].mean()), 4),
        "price_min": round(float(labelled[PRICE_COLUMN].min()), 0),
        "price_max": round(float(labelled[PRICE_COLUMN].max()), 0),
    }
    summary_path = PROC_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = summary_path
    return paths, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional house CSV with the same columns as house_data.csv",
    )
    args = parser.parse_args()

    frame = load_dataset(args.csv)
    paths, summary = split_and_save(frame)
    print(json.dumps(summary, indent=2))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

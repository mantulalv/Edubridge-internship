"""Download and cache the 20 Newsgroups dataset."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROC_DIR = ROOT / "processed"
SEED = 42

CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def fetch(subset: str) -> tuple[pd.DataFrame, list[str]]:
    """Fetch one split and return it as a text/target frame."""
    data = fetch_20newsgroups(
        subset=subset,
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=SEED,
        data_home=str(RAW_DIR),
    )
    frame = pd.DataFrame({"text": data.data, "target": data.target})
    return frame, list(data.target_names)


def split_and_save() -> dict[str, Path]:
    """Persist train/test CSVs plus the ordered class labels."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    train, labels = fetch("train")
    test, _ = fetch("test")

    paths: dict[str, Path] = {}
    for name, frame in (("train", train), ("test", test)):
        path = PROC_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    labels_path = PROC_DIR / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    paths["labels"] = labels_path
    return paths


def summary(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "classes": int(frame["target"].nunique()),
        "smallest_class": int(frame["target"].value_counts().min()),
        "largest_class": int(frame["target"].value_counts().max()),
    }


if __name__ == "__main__":
    paths = split_and_save()
    train = pd.read_csv(paths["train"])
    print(summary(train))
    for name, path in paths.items():
        print(f"{name}: {path}")

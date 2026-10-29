"""Download and prepare the UCI SMS Spam Collection dataset."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
PROC_DIR = ROOT / "processed"
SEED = 42


def download() -> Path:
    """Return the raw dataset file, downloading and unzipping it once."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "SMSSpamCollection"
    if target.exists():
        return target

    print(f"downloading {RAW_URL} ...")
    try:
        payload = urllib.request.urlopen(RAW_URL, timeout=60).read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the SMS Spam Collection. Download it manually from "
            f"{RAW_URL} and place the extracted file `SMSSpamCollection` in {RAW_DIR}."
        ) from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(RAW_DIR)

    if not target.exists():
        raise FileNotFoundError(f"expected {target} after extracting the archive")
    return target


def load_raw() -> pd.DataFrame:
    """Load, de-duplicate and label the raw messages."""
    frame = pd.read_csv(
        download(),
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    frame = frame.dropna(subset=["message"]).drop_duplicates(subset="message")
    frame = frame.reset_index(drop=True)
    frame["target"] = (frame["label"].str.lower() == "spam").astype(int)
    return frame[["message", "target"]]


def split_and_save() -> dict[str, Path]:
    """Create a stratified 70/15/15 train/validation/test split on disk."""
    frame = load_raw()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    train, rest = train_test_split(
        frame, test_size=0.30, stratify=frame["target"], random_state=SEED
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["target"], random_state=SEED
    )

    paths: dict[str, Path] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = PROC_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        paths[name] = path
    return paths


def summary(frame: pd.DataFrame) -> dict[str, float | int]:
    """Small JSON-friendly overview of a labelled frame."""
    return {
        "rows": int(len(frame)),
        "spam": int(frame["target"].sum()),
        "ham": int((frame["target"] == 0).sum()),
        "spam_rate": round(float(frame["target"].mean()), 4),
    }


if __name__ == "__main__":
    paths = split_and_save()
    print(summary(load_raw()))
    for split_name, split_path in paths.items():
        print(f"{split_name}: {split_path}")

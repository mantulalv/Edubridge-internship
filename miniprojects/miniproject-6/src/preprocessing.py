"""Text normalisation for spam classification."""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
PHONE_RE = re.compile(r"\+?\d[\d\s()-]{6,}\d")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, mask entities, strip punctuation and drop stop/short words."""
    text = text.lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    tokens = TOKEN_RE.findall(text)
    return " ".join(t for t in tokens if len(t) >= 2 and t not in ENGLISH_STOP_WORDS)


def normalize_series(series: pd.Series) -> pd.Series:
    """Vectorised ``normalize`` over a pandas Series."""
    return series.astype(str).map(normalize)

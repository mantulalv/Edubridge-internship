"""Cleaning helpers for newsgroup posts."""
from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

HEADER_RE = re.compile(r"^[A-Za-z][\w-]{1,20}:\s.*$", re.MULTILINE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
TOKEN_RE = re.compile(r"[a-z]+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Remove headers, URLs, e-mails and digits, then drop short/stop words."""
    text = HEADER_RE.sub(" ", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = TOKEN_RE.findall(text)
    return " ".join(token for token in tokens if len(token) > 2 and token not in STOP_WORDS)


def clean_series(series: pd.Series) -> pd.Series:
    """Vectorised ``clean_text`` over a pandas Series."""
    return series.astype(str).map(clean_text)

"""Unit tests for newsgroup cleaning."""
import pandas as pd

from src.preprocessing import clean_series, clean_text


def test_removes_headers_urls_and_emails():
    raw = "From: someone@example.com\nSubject: Re: launch\nSee https://nasa.gov for details"
    cleaned = clean_text(raw)
    assert "example" not in cleaned
    assert "nasa.gov" not in cleaned
    assert "subject" not in cleaned


def test_removes_digits_and_short_tokens():
    cleaned = clean_text("The 2024 mission had 3 rockets and a x y z")
    assert "2024" not in cleaned
    assert "mission" in cleaned
    assert all(len(token) > 2 for token in cleaned.split())


def test_clean_series_returns_strings():
    result = clean_series(pd.Series(["Hello WORLD", "Second post"]))
    assert result.tolist() == ["hello world", "second post"]

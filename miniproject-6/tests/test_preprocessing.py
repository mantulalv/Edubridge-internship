"""Unit tests for text normalisation."""
from src.preprocessing import normalize, normalize_series
import pandas as pd


def test_lowercases_and_strips_punctuation():
    assert normalize("Hello, WORLD!!!") == "hello world"


def test_masks_urls_emails_and_phones():
    result = normalize("Mail me@example.com or visit https://spam.example.com, call 09061701461")
    assert "emailtoken" in result
    assert "urltoken" in result
    assert "phonetoken" in result
    assert "example.com" not in result


def test_drops_single_character_tokens():
    assert normalize("x y z") == ""
    assert normalize("a b c") == ""


def test_normalize_series_returns_series():
    frame = pd.Series(["FREE cash", "hello there"])
    result = normalize_series(frame)
    assert isinstance(result, pd.Series)
    assert result.tolist() == ["free cash", "hello"]

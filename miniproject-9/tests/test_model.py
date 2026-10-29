"""Tests for the NewsLens pipeline on a tiny corpus."""
from src.model import build_pipeline, top_keywords

TEXTS = [
    "hockey puck goal ice",
    "hockey team goal skate",
    "hockey game ice rink",
    "space shuttle orbit nasa",
    "space mission orbit moon",
    "space rocket nasa launch",
    "graphics card pixel image",
    "graphics render image card",
    "graphics pixel render screen",
]
LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2]


def fitted_pipeline():
    pipeline = build_pipeline()
    pipeline.fit(TEXTS, LABELS)
    return pipeline


def test_pipeline_fits_toy_corpus():
    pipeline = fitted_pipeline()
    accuracy = float((pipeline.predict(TEXTS) == LABELS).mean())
    assert accuracy >= 0.8


def test_probabilities_are_valid():
    pipeline = fitted_pipeline()
    probabilities = pipeline.predict_proba(["hockey ice goal"])[0]
    assert len(probabilities) == 3
    assert abs(float(probabilities.sum()) - 1.0) < 1e-9


def test_top_keywords_returns_supporting_tokens():
    pipeline = fitted_pipeline()
    predicted, keywords = top_keywords(pipeline, "hockey ice goal")
    assert predicted in (0, 1, 2)
    assert keywords
    assert all(isinstance(token, str) for token, _ in keywords)

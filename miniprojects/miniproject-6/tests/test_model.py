"""Unit tests for the pipeline factory and explanations."""
from src.model import build_pipeline, top_features

TEXTS = [
    "free cash prize click now",
    "free cash offer click",
    "win cash prize click",
    "meeting at five tomorrow",
    "meeting agenda tomorrow",
    "team lunch tomorrow",
]
LABELS = [1, 1, 1, 0, 0, 0]


def test_pipeline_learns_separable_data():
    pipeline = build_pipeline("nb")
    pipeline.fit(TEXTS, LABELS)

    assert pipeline.predict(["free cash prize"])[0] == 1
    assert pipeline.predict(["team meeting tomorrow"])[0] == 0


def test_pipeline_probabilities_are_valid():
    pipeline = build_pipeline("lr")
    pipeline.fit(TEXTS, LABELS)
    probability = pipeline.predict_proba(["free cash"])[0]
    assert abs(float(probability.sum()) - 1.0) < 1e-9


def test_top_features_returns_both_directions():
    pipeline = build_pipeline("lr")
    pipeline.fit(TEXTS, LABELS)
    contributions = top_features(pipeline, "free cash click")
    assert set(contributions) == {"spam", "ham"}
    assert contributions["spam"]
    assert all(isinstance(token, str) for token, _ in contributions["spam"])

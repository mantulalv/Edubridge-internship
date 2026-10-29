"""Model construction, explanation and persistence."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_pipeline(kind: str) -> Pipeline:
    """Build a TF-IDF pipeline with the requested classifier (``nb`` or ``lr``)."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    if kind == "nb":
        classifier = MultinomialNB()
    elif kind == "lr":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        raise ValueError(f"unknown model kind: {kind!r}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no model at {path}; run `python src/train.py` first"
        )
    return joblib.load(path)


def top_features(pipeline: Pipeline, text: str, k: int = 8) -> dict[str, list[tuple[str, float]]]:
    """Tokens pushing the prediction towards spam / ham (Logistic Regression only)."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        return {"spam": [], "ham": []}

    names = vectorizer.get_feature_names_out()
    matrix = vectorizer.transform([text]).multiply(classifier.coef_[0]).tocoo()
    weighted = sorted(zip(matrix.col, matrix.data), key=lambda item: item[1])

    spam = [
        (str(names[index]), round(float(value), 4))
        for index, value in reversed(weighted)
        if value > 0
    ][:k]
    ham = [
        (str(names[index]), round(float(value), 4))
        for index, value in weighted
        if value < 0
    ][:k]
    return {"spam": spam, "ham": ham}

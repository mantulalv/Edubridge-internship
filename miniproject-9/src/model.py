"""Text-classification pipeline and keyword attribution."""
from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """Word unigrams + bigrams, TF-IDF, multinomial logistic regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, C=4.0),
            ),
        ]
    )


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"no model at {path}; run `python src/train.py` first"
        )
    return joblib.load(path)


def top_keywords(
    pipeline: Pipeline, text: str, k: int = 8
) -> tuple[int, list[tuple[str, float]]]:
    """Return the predicted class index and the tokens that support it most."""
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    predicted = int(pipeline.predict([text])[0])
    names = vectorizer.get_feature_names_out()
    row = vectorizer.transform([text]).tocoo()

    contributions = [
        (
            str(names[column]),
            round(float(value) * float(classifier.coef_[predicted, column]), 4),
        )
        for column, value in zip(row.col, row.data)
    ]
    contributions.sort(key=lambda item: item[1], reverse=True)
    return predicted, contributions[:k]

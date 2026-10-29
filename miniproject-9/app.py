"""Streamlit explorer for NewsLens AI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import load_bundle, top_keywords
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "news_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

SAMPLES = {
    "Space": "The space shuttle will launch from Kennedy Space Center next week. "
    "The crew will orbit the Earth and test a new satellite payload.",
    "Hockey": "The goalie made three saves in the final period and the team scored "
    "on a power play. The puck bounced off the post in overtime.",
    "Graphics": "I need a graphics card that supports a high resolution display and "
    "fast rendering for image processing and animation work.",
}

st.set_page_config(page_title="NewsLens AI", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> dict:
    return load_bundle(Path(path))


st.title("📰 NewsLens AI")
st.caption("20 Newsgroups topic classification with TF-IDF + Logistic Regression.")

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run the setup steps first:\n\n"
        "```\npython data/data_loader.py\npython src/train.py\n```"
    )
    st.stop()

bundle = get_bundle(str(MODEL_PATH))
pipeline = bundle["pipeline"]
labels: list[str] = bundle["labels"]

if "post" not in st.session_state:
    st.session_state.post = ""

with st.sidebar:
    st.header("Try a sample")
    for label, sample in SAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.post = sample
    st.divider()
    st.caption("Samples are short paraphrases, not part of the training set.")

text = st.text_area(
    "Post text",
    key="post",
    height=220,
    placeholder="Paste an e-mail, forum post or article excerpt ...",
)

if st.button("Classify topic", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        cleaned = clean_text(text)
        if not cleaned:
            st.warning("Nothing left after cleaning — try a longer text.")
            st.stop()

        probabilities = pipeline.predict_proba([cleaned])[0]
        order = probabilities.argsort()[::-1]
        predicted = int(order[0])

        st.subheader(f"Prediction: {labels[predicted]}")
        st.metric("Confidence", f"{probabilities[predicted]:.1%}")

        top = pd.DataFrame(
            {
                "topic": [labels[index] for index in order[:5]],
                "probability": [float(probabilities[index]) for index in order[:5]],
            }
        ).set_index("topic")
        st.bar_chart(top, height=280)

        _, keywords = top_keywords(pipeline, cleaned)
        st.markdown("**Keywords supporting the prediction**")
        st.write(", ".join(f"`{token}`" for token, _ in keywords) or "—")

with st.expander("Model card"):
    if METRICS_PATH.exists():
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(payload.get("metrics", {}))
    st.caption(
        "Trained on the standard scikit-learn train split (headers, footers and "
        "quotes removed) and evaluated on the matching test split."
    )

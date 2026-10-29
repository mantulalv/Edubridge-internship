"""Streamlit front end for SpamShield AI."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.model import load_bundle, top_features
from src.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "outputs" / "training_metrics.json"

SAMPLES = {
    "Spam example": "WINNER!! You have been selected to receive a £2000 cash prize. "
    "Call 09061701461 now to claim before midnight!",
    "Ham example": "Hey, are we still meeting at the library around 5? I will bring the notes.",
    "Phishing example": "Your account has been suspended. Verify your details at "
    "http://secure-login.example.com immediately.",
}

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> dict:
    return load_bundle(Path(path))


st.title("🛡️ SpamShield AI")
st.caption(
    "TF-IDF + machine learning, trained on the UCI SMS Spam Collection. "
    "Type a message or load an example."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run the setup steps first:\n\n"
        "```\npython data/data_loader.py\npython src/train.py\n```"
    )
    st.stop()

bundle = get_bundle(str(MODEL_PATH))
pipeline = bundle["pipeline"]

if "message" not in st.session_state:
    st.session_state.message = ""

columns = st.columns(len(SAMPLES))
for column, (label, sample) in zip(columns, SAMPLES.items()):
    if column.button(label, use_container_width=True):
        st.session_state.message = sample

text = st.text_area(
    "Message to classify",
    key="message",
    height=140,
    placeholder="Paste an SMS or e-mail body here ...",
)

if st.button("Classify", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        cleaned = normalize(text)
        probability = float(pipeline.predict_proba([cleaned])[0][1])
        is_spam = probability >= 0.5

        if is_spam:
            st.error(f"🚨 Spam detected — {probability:.1%} spam probability")
        else:
            st.success(f"✅ Looks legitimate — {probability:.1%} spam probability")
        st.progress(min(max(probability, 0.0), 1.0))

        contributions = top_features(pipeline, cleaned)
        if contributions["spam"] or contributions["ham"]:
            st.subheader("Why?")
            left, right = st.columns(2)
            with left:
                st.markdown("**Pushes towards spam**")
                st.write(
                    ", ".join(f"`{token}`" for token, _ in contributions["spam"]) or "—"
                )
            with right:
                st.markdown("**Pushes towards ham**")
                st.write(
                    ", ".join(f"`{token}`" for token, _ in contributions["ham"]) or "—"
                )

with st.expander("Model card"):
    st.write(f"Selected classifier: **{bundle.get('kind', 'unknown')}**")
    validation = bundle.get("validation", {})
    if validation:
        st.json(validation)
    if METRICS_PATH.exists():
        st.caption("Validation comparison from the last training run")
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    st.caption("Prototype only — never rely on a single automated spam decision.")

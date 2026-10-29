"""Streamlit demo for NeuralVision AI."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.data import prepare_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "digit_cnn.keras"

st.set_page_config(page_title="NeuralVision AI", page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Loading the model ...")
def get_model(path: str):
    from tensorflow import keras

    return keras.models.load_model(path)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run:\n\n```\npython src/train.py\n```")
    st.stop()

model = get_model(str(MODEL_PATH))

st.title("🔢 NeuralVision AI")
st.caption("A small CNN trained on MNIST — upload a handwritten digit (0-9).")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Digit image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    st.divider()
    use_sample = st.checkbox("Use an MNIST test image")
    sample_index = st.number_input(
        "Test index", min_value=0, max_value=9999, value=0, step=1,
        disabled=not use_sample,
    )
    st.divider()
    st.caption(
        "Tip: crop tightly, use dark ink on a light background, and keep the digit "
        "centred. Preprocessing handles inversion automatically."
    )

image: Image.Image | None = None
truth: int | None = None

if uploaded is not None:
    try:
        image = Image.open(uploaded)
    except Exception:
        st.error("That file could not be read as an image.")
        st.stop()
elif use_sample:
    from tensorflow import keras

    try:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    except Exception:
        st.error("Could not load MNIST. Check your internet connection once, then retry.")
        st.stop()
    index = int(sample_index)
    image = Image.fromarray(x_test[index])
    truth = int(y_test[index])
    st.info(f"Ground truth for test image #{index}: **{truth}**")

if image is None:
    st.write("Upload an image or enable a test sample to get a prediction.")
    st.stop()

processed = prepare_image(image)

left, right = st.columns([1, 2])
with left:
    st.subheader("Model input")
    st.image(processed[..., 0], width=190, clamp=True, caption="28x28, white on black")

with right:
    probabilities = model.predict(processed[None, ...], verbose=0)[0]
    predicted = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted])

    st.metric("Predicted digit", str(predicted), f"{confidence:.1%} confidence")
    if truth is not None:
        st.caption("Correct ✅" if predicted == truth else f"Wrong ❌ (truth: {truth})")

    chart = pd.DataFrame(
        {"digit": [str(index) for index in range(10)], "probability": probabilities}
    ).set_index("digit")
    st.bar_chart(chart, height=280)

with st.expander("Model card"):
    st.write("Architecture: Conv32 → pool → Conv64 → pool → Conv64 → Dense128 → Dropout(0.4) → Dense10")
    st.write("Training: Adam, sparse categorical cross-entropy, early stopping on validation loss.")
    metrics_path = ROOT / "outputs" / "metrics.json"
    if metrics_path.exists():
        import json

        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))

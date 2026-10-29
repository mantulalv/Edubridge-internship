"""Streamlit demo: is this house above the median price?"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import load_bundle, local_contributions
from src.preprocessing import FEATURES, feature_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "premium_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="HouseValue AI", page_icon="🏡", layout="centered")


@st.cache_resource(show_spinner=False)
def get_bundle(path: str) -> dict:
    return load_bundle(Path(path))


st.title("🏡 HouseValue AI")
st.caption(
    "Logistic regression on the Edubridge assignment-5 house data — "
    "premium (above median) vs value homes."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run the setup steps first:\n\n"
        "```\npython data/data_loader.py\npython src/train.py\n```"
    )
    st.stop()

bundle = get_bundle(str(MODEL_PATH))
pipeline = bundle["pipeline"]
threshold = float(bundle.get("threshold", 0.0))

st.info(
    f"Median price threshold: **₹{threshold:,.0f}**. The model predicts whether a "
    "property sits above or below this line — it does not estimate the exact price."
)

with st.form("property"):
    left, right = st.columns(2)
    with left:
        area = st.slider("Area (sq ft)", 500, 4000, 1500, step=50)
        bedrooms = st.slider("Bedrooms", 1, 5, 3)
        bathrooms = st.slider("Bathrooms", 1, 3, 2)
    with right:
        age = st.slider("Age (years)", 0, 40, 10)
        distance = st.slider(
            "Distance to city centre (km)", 1.0, 30.0, 10.0, step=0.5
        )
    submitted = st.form_submit_button("Classify property", type="primary")

if submitted:
    row = {
        "area_sqft": float(area),
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "age_years": float(age),
        "distance_to_city_km": float(distance),
    }
    probability = float(
        pipeline.predict_proba(feature_frame(pd.DataFrame([row])))[0][1]
    )
    premium = probability >= 0.5

    columns = st.columns(3)
    columns[0].metric("Above-median probability", f"{probability:.1%}")
    columns[1].metric("Segment", "Premium" if premium else "Value")
    columns[2].metric("Median threshold", f"₹{threshold:,.0f}")

    if premium:
        st.success(
            f"Likely a **premium** property — {probability:.1%} chance of being "
            f"priced above ₹{threshold:,.0f}."
        )
    else:
        st.warning(
            f"Likely a **value** property — only {probability:.1%} chance of being "
            f"priced above ₹{threshold:,.0f}."
        )
    st.progress(min(max(probability, 0.0), 1.0))

    contributions = local_contributions(pipeline, row)
    st.subheader("What pushes this prediction?")
    st.caption(
        "Scaled contribution = standardised feature value × logistic-regression "
        "weight. Positive pushes towards premium, negative towards value."
    )
    chart = pd.DataFrame(contributions, columns=["feature", "contribution"]).set_index(
        "feature"
    )
    st.bar_chart(chart)

with st.expander("Model card"):
    st.write(
        f"Selected C: **{bundle.get('C')}** · validation: "
        f"`{bundle.get('validation')}`"
    )
    if METRICS_PATH.exists():
        st.caption("Test-set metrics from the last evaluation run")
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    coefficients = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "coefficient": pipeline.named_steps["clf"].coef_[0],
            }
        )
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(coefficients, use_container_width=True, hide_index=True)
    st.caption(
        "A positive coefficient raises the odds of the premium class; every "
        "feature is standardised before fitting."
    )

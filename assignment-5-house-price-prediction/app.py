import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write("Enter the house details to get an estimated price.")


@st.cache_resource
def load_model():
    if not os.path.exists("house_model.pkl"):
        return None
    return joblib.load("house_model.pkl")


bundle = load_model()
if bundle is None:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model, features = bundle["model"], bundle["features"]

area = st.slider("Area (sq ft)", 500, 4000, 1500, step=50)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
age = st.slider("Age of house (years)", 0, 50, 10)
distance = st.slider("Distance to city centre (km)", 1.0, 30.0, 10.0, step=0.5)

if st.button("Predict Price"):
    row = pd.DataFrame(
        [[area, bedrooms, bathrooms, age, distance]], columns=features
    )
    price = model.predict(row)[0]
    st.success(f"Estimated price: ₹{price:,.0f}")

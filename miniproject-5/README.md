# 🏠 House Price Prediction

A simple machine learning project that predicts house prices using **Linear Regression** and serves the model through a **Streamlit** web app.

The model is trained on synthetic sample data (area, bedrooms, bathrooms, age, distance to city), so it is meant for learning and demonstration.

## Features

- Generates a 1,000-row sample dataset
- Trains a Linear Regression model with scikit-learn
- Reports R² score, MAE and feature coefficients
- Interactive Streamlit app for instant price predictions

## Project Structure

```
house-price-prediction/
├── train_model.py     # Creates data, trains and saves the model
├── app.py             # Streamlit web app
├── requirements.txt   # Python dependencies
├── house_data.csv     # Generated sample data
├── house_model.pkl    # Saved trained model
├── .gitignore
├── LICENSE
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

# 2. Create a virtual environment (Python 3.10 - 3.13 recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**Step 1: Train the model**

```bash
python train_model.py
```

This creates `house_data.csv` and `house_model.pkl` and prints the model's performance.

**Step 2: Run the app**

```bash
streamlit run app.py
```

Open the link shown in the terminal (usually http://localhost:8501), enter the house details and click **Predict Price**.

## Model Details

| Item | Value |
|------|-------|
| Algorithm | Linear Regression (scikit-learn) |
| Features | area_sqft, bedrooms, bathrooms, age_years, distance_to_city_km |
| Target | price |
| Train/Test split | 80% / 20% |

## Using Your Own Data

In `train_model.py`, replace the data generation block with:

```python
df = pd.read_csv("your_data.csv")
```

Then update the `features` list and the `price` target column to match your dataset.

## Tech Stack

Python · pandas · NumPy · scikit-learn · joblib · Streamlit

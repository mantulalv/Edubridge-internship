import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create sample data
rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    "area_sqft": rng.integers(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age_years": rng.integers(0, 40, n),
    "distance_to_city_km": rng.uniform(1, 30, n).round(1),
})

df["price"] = (
    50_000
    + df["area_sqft"] * 120
    + df["bedrooms"] * 10_000
    + df["bathrooms"] * 8_000
    - df["age_years"] * 1_500
    - df["distance_to_city_km"] * 2_000
    + rng.normal(0, 15_000, n)
).round(0)

df.to_csv("house_data.csv", index=False)

# 2. Split data
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_to_city_km"]
X = df[features]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
pred = model.predict(X_test)
print(f"R2 score : {r2_score(y_test, pred):.4f}")
print(f"MAE      : {mean_absolute_error(y_test, pred):,.0f}")
for name, coef in zip(features, model.coef_):
    print(f"{name:22s} coef = {coef:,.2f}")

# 5. Save model
joblib.dump({"model": model, "features": features}, "house_model.pkl")
print("Model saved to house_model.pkl")

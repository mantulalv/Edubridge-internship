# HouseValue AI — Premium Property Classification

A logistic-regression adaptation of the Edubridge **assignment‑5 house price**
project. The original predicts a continuous price with linear regression; this
mini project keeps the same five features and turns the problem into a binary
classification: **is this house priced above or below the dataset median?**

**Developer:** mantulalv · **Repo:** https://github.com/mantulalv/Edubridge-internship/tree/main/miniproject-10

Original regression project:
https://github.com/mantulalv/Edubridge-internship/tree/main/miniproject-5

## Why logistic regression?

The final mini project requires logistic regression, and house price is
continuous. Encoding "premium" as `price > median` gives a balanced, meaningful
binary target while keeping the exact same inputs:

| Feature | Description |
|---|---|
| `area_sqft` | Built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the property |
| `distance_to_city_km` | Distance from the city centre |

Target: `premium = 1` when the price is strictly above the dataset median.
The app therefore reports **above-median probability**, not an exact price.

## Dataset

`data/data_loader.py` recreates the assignment‑5 sample data exactly (same seed
42 and formula), so no download or Kaggle account is needed. You can also point
it at your own CSV with the same columns:

```bash
python data/data_loader.py --csv "D:\path\house_data.csv"
```

The loader writes a stratified 70/15/15 split plus `data/processed/summary.json`
(row count, median threshold, class balance).

## Model

```
StandardScaler -> LogisticRegression(C tuned on the validation split)
```

`C` is chosen from `{0.01, 0.1, 1, 10, 100}` by validation F1 (accuracy breaks
ties). Scaled coefficients are saved to `outputs/coefficients.csv` as both
weights and odds ratios, so the direction of every feature is interpretable.

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
python data/data_loader.py
python src/train.py
python src/evaluate.py
streamlit run app.py
pytest -q
```

## Project layout

```
miniproject-10/
├── app.py                 Streamlit premium-home classifier
├── data/
│   ├── data_loader.py     generate/copy data, add target, split
│   ├── raw/               (git-ignored)
│   └── processed/         (git-ignored)
├── models/                (git-ignored)
├── outputs/               metrics, coefficients, evaluation plots
├── src/
│   ├── preprocessing.py   feature contract + target construction
│   ├── model.py           pipeline, persistence, local explanations
│   ├── train.py           C tuning + training
│   └── evaluate.py        test metrics, confusion matrix, ROC
└── tests/
```

## Results

Paste the real figures from `outputs/metrics.json` here after your first run.

MIT © 2026 mantulalv

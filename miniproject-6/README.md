# SpamShield AI — SMS & Email Spam Classification

TF-IDF + Naive Bayes / Logistic Regression spam classifier with an interactive Streamlit scorer.

**Developer:** mantulalv · **Repo:** https://github.com/mantulalv/Edubridge-internship/tree/main/miniproject-6

## Features

- Downloads and de-duplicates the UCI SMS Spam Collection automatically.
- Stratified 70/15/15 train/validation/test split.
- Cleans text (URL/e-mail/phone masking, stop-word removal) before vectorising.
- Trains Naive Bayes and Logistic Regression, keeps the better validation F1.
- Saves test metrics, a confusion matrix and per-token explanations.

## Project layout

```
miniproject-6/
├── app.py                 Streamlit scorer
├── data/
│   ├── data_loader.py     download + split
│   ├── raw/               (git-ignored)
│   └── processed/         (git-ignored)
├── models/                (git-ignored)
├── outputs/               metrics + plots
├── src/
│   ├── model.py           pipelines + persistence
│   ├── preprocessing.py   text normalisation
│   ├── train.py           training + model selection
│   └── evaluate.py        test-set evaluation
└── tests/
```

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

## Train / Evaluate / App / Test

```bash
python data/data_loader.py
python src/train.py
python src/evaluate.py
streamlit run app.py
pytest -q
```

## Dataset

UCI SMS Spam Collection, downloaded by `data/data_loader.py`, de-duplicated, stratified 70/15/15 split.
If the download is blocked, put the extracted `SMSSpamCollection` file in `data/raw/` and rerun the loader.

## Results

Paste real figures from `outputs/metrics.json` here after your first run.

MIT © 2026 mantulalv

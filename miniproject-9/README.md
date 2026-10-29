# NewsLens AI — 20 Newsgroups Topic Classification

TF-IDF + Logistic Regression over the 20 Newsgroups corpus, with a Streamlit app that
predicts the topic of pasted text and shows the keywords behind the decision.

**Developer:** mantulalv · **Repo:** https://github.com/mantulalv/Edubridge-internship/tree/main/miniproject-9

## Features

- Downloads and caches the 20-class `fetch_20newsgroups` dataset.
- Strips e-mail headers, URLs, e-mail addresses and digits before vectorising.
- Trains a word + bigram TF-IDF pipeline and reports macro F1.
- Streamlit app shows the top-3 topics with probabilities and influencing keywords.

## Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

## Train / App / Test

```bash
python data/data_loader.py
python src/train.py
streamlit run app.py
pytest -q
```

## Dataset

`data/data_loader.py` calls `sklearn.datasets.fetch_20newsgroups` for the standard
`train` and `test` subsets (20 categories, headers/footers/quotes removed). The raw cache
lives in `data/raw/` and the processed CSVs in `data/processed/` — both git-ignored.

If the download is blocked, copy an existing scikit-learn cache into `data/raw/`.

## Project layout

```
miniproject-9/
├── app.py                 Streamlit explorer
├── data/
│   ├── data_loader.py     download + persist splits
│   └── processed/         (git-ignored)
├── src/
│   ├── model.py           pipeline + keyword attribution
│   ├── preprocessing.py   newsgroup text cleaning
│   └── train.py           training + metrics
├── tests/
└── outputs/
```

## Results

Fill in `outputs/metrics.json` figures here after training.

MIT © 2026 mantulalv

# Mini Projects — ML & Data Science Portfolio

| # | Project | Stack | Folder |
|---|---------|-------|--------|
| 6 | SpamShield AI — SMS/email spam classification | scikit-learn, Streamlit | `miniproject-6/` |
| 7 | Student analytics dashboard | pandas, Streamlit, Docker | `miniproject-7/` |
| 8 | NeuralVision AI — handwritten digit recognition | TensorFlow, Streamlit | `miniproject-8/` |
| 9 | NewsLens AI — 20 Newsgroups classification | scikit-learn, Streamlit | `miniproject-9/` |
| 10 | HouseValue AI — premium property classification | scikit-learn, Streamlit | `miniproject-10/` |

Each folder is independent: `cd` in, create a venv, install its `requirements.txt`, follow its README.

## Quick start

```bash
cd miniproject-6
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python data/data_loader.py
python src/train.py
streamlit run app.py
```

```bash
cd miniproject-7
pip install -r requirements.txt
python data/generate_data.py
streamlit run app.py
# or: docker compose up --build
```

```bash
cd miniproject-8
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

```bash
cd miniproject-9
pip install -r requirements.txt
python data/data_loader.py
python src/train.py
streamlit run app.py
```

```bash
cd miniproject-10
pip install -r requirements.txt
python data/data_loader.py
python src/train.py
python src/evaluate.py
streamlit run app.py
```

## Shared conventions

- Every project has `src/`, `tests/`, `outputs/` and a `README.md`.
- Tests run with `pytest -q` from inside the project folder.
- Trained artefacts and raw/processed data are git-ignored; `outputs/` keeps only `.gitkeep`.

MIT © 2026 mantulalv

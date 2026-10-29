# Edubridge Internship — ML & Data Science Projects

Machine Learning / Data Science work completed during the Edubridge internship: ten self-contained mini projects.

| # | Project | Stack | Folder |
|---|---------|-------|--------|
| 1 | Trade data EDA | pandas, Jupyter | [`miniproject-1/`](miniproject-1/) |
| 2 | Loan eligibility (logistic regression) | scikit-learn, Jupyter | [`miniproject-2/`](miniproject-2/) |
| 3 | Crypto dashboard | Streamlit, Plotly | [`miniproject-3/`](miniproject-3/) |
| 4 | Object detection (YOLOv8) | Ultralytics, OpenCV | [`miniproject-4/`](miniproject-4/) |
| 5 | House price prediction | scikit-learn, Streamlit, Docker | [`miniproject-5/`](miniproject-5/) |
| 6 | SpamShield AI — SMS/email spam classification | scikit-learn, Streamlit | [`miniproject-6/`](miniproject-6/) |
| 7 | Student analytics dashboard | pandas, Streamlit, Docker | [`miniproject-7/`](miniproject-7/) |
| 8 | NeuralVision AI — handwritten digit recognition | TensorFlow, Streamlit | [`miniproject-8/`](miniproject-8/) |
| 9 | NewsLens AI — 20 Newsgroups classification | scikit-learn, Streamlit | [`miniproject-9/`](miniproject-9/) |
| 10 | HouseValue AI — premium property classification | scikit-learn, Streamlit | [`miniproject-10/`](miniproject-10/) |

Each project is self-contained: its own `README.md`, `requirements.txt` and source folder.

```bash
cd miniproject-6
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
# then follow that folder's README for data, training and app commands
```

## Notes

- Raw/processed data, trained models and generated plots are git-ignored. Run each project's data/training scripts to regenerate them.
- Tests (where present): `pytest -q` from inside the project folder.

MIT © 2026 mantulalv

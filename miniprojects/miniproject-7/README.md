# Student Analytics Dashboard

Interactive cohort analytics for student performance: KPIs, department comparisons,
semester trends, attendance bands and at-risk detection.

**Developer:** mantulalv · **Repo:** https://github.com/mantulalv/Edubridge-internship/tree/main/miniprojects/miniproject-7

## Features

- Synthetic but realistic dataset (`data/generate_data.py`, seed 42).
- Pass/fail, grade bands and a transparent at-risk rule:
  attendance < 75% **or** score < 50 **or** two or more backlogs.
- Sidebar filters (department, semester, grade, student ID, at-risk only).
- Plotly visuals plus CSV export of the filtered table.
- Docker image that generates data at build time.

## Local run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
python data/generate_data.py
streamlit run app.py
pytest -q
```

## Docker

```bash
docker compose up --build
# open http://localhost:8501
```

## Project layout

```
miniproject-7/
├── app.py                 Streamlit dashboard
├── data/
│   ├── generate_data.py   synthetic dataset builder
│   └── students.csv       (git-ignored)
├── src/
│   ├── analytics.py       pure pandas logic (tested)
│   └── charts.py          Plotly figures
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Notes

`src/analytics.py` contains no Streamlit imports, so all business rules are unit-testable.
Swap `data/generate_data.py` for a real CSV reader while keeping the same column names.

MIT © 2026 mantulalv

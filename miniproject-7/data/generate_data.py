"""Create a reproducible synthetic student performance dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "students.csv"
SEED = 42
DEPARTMENTS = ["CSE", "IT", "ECE", "MECH", "CIVIL"]
SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8]


def build(rows: int = 1200, seed: int = SEED) -> pd.DataFrame:
    """Return ``rows`` synthetic students with correlated features."""
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(82, 11, rows), 35, 100)
    study_hours = np.clip(rng.normal(14, 5.5, rows), 0, 40)
    prior_score = np.clip(rng.normal(64, 13, rows), 10, 100)

    base = 0.38 * attendance + 1.55 * study_hours + 0.22 * prior_score
    assignment = np.clip(base + rng.normal(4, 6, rows), 0, 100)
    exam = np.clip(base + rng.normal(0, 8, rows), 0, 100)
    total = 0.4 * assignment + 0.6 * exam
    risk = np.clip((55 - total) / 30, 0, 1)
    backlogs = rng.binomial(4, risk * 0.45)

    return pd.DataFrame(
        {
            "student_id": [f"STU{index:04d}" for index in range(1, rows + 1)],
            "department": rng.choice(DEPARTMENTS, rows),
            "semester": rng.choice(SEMESTERS, rows),
            "attendance_pct": attendance.round(1),
            "study_hours": study_hours.round(1),
            "prior_score": prior_score.round(1),
            "assignment_score": assignment.round(1),
            "exam_score": exam.round(1),
            "backlogs": backlogs,
        }
    )


def main() -> None:
    frame = build()
    frame.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(frame):,} rows to {OUT_PATH}")
    print(frame.head().to_string(index=False))
    print(frame[["attendance_pct", "study_hours", "assignment_score", "exam_score"]].describe().round(2))


if __name__ == "__main__":
    main()

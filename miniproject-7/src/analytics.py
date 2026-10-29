"""Pure pandas analytics shared by the dashboard and the tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GRADE_ORDER = ["A+", "A", "B", "C", "D", "E", "F"]
PASS_MARK = 40
REQUIRED_COLUMNS = {
    "student_id",
    "department",
    "semester",
    "attendance_pct",
    "study_hours",
    "prior_score",
    "assignment_score",
    "exam_score",
    "backlogs",
}
DEPARTMENT_COLUMNS = [
    "department",
    "students",
    "avg_score",
    "pass_rate",
    "avg_attendance",
    "at_risk_rate",
]
SEMESTER_COLUMNS = ["semester", "students", "avg_score", "pass_rate"]
BAND_COLUMNS = ["band", "students", "pass_rate", "avg_score"]


def grade_for(score: float) -> str:
    """Map a 0-100 total score to a letter grade."""
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    if score >= PASS_MARK:
        return "E"
    return "F"


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add total score, grade, pass flag and the at-risk flag."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = frame.copy()
    out["total_score"] = (
        0.4 * out["assignment_score"] + 0.6 * out["exam_score"]
    ).round(2)
    out["grade"] = out["total_score"].map(grade_for)
    out["passed"] = out["total_score"] >= PASS_MARK
    out["at_risk"] = (
        (out["attendance_pct"] < 75)
        | (out["total_score"] < 50)
        | (out["backlogs"] >= 2)
    )
    return out


def load_students(path: str | Path) -> pd.DataFrame:
    """Read a CSV and attach derived columns."""
    return add_derived(pd.read_csv(path))


def apply_filters(
    frame: pd.DataFrame,
    departments: list[str] | None = None,
    semesters: list[int] | None = None,
    grades: list[str] | None = None,
    search: str = "",
    only_at_risk: bool = False,
) -> pd.DataFrame:
    """Filter the cohort; empty selections mean 'no filter'."""
    out = frame
    if departments:
        out = out[out["department"].isin(departments)]
    if semesters:
        out = out[out["semester"].isin(semesters)]
    if grades:
        out = out[out["grade"].isin(grades)]
    if search.strip():
        out = out[out["student_id"].str.contains(search.strip(), case=False, na=False)]
    if only_at_risk:
        out = out[out["at_risk"]]
    return out.reset_index(drop=True)


def kpis(frame: pd.DataFrame) -> dict[str, float | int]:
    """Headline numbers for the selected cohort."""
    if frame.empty:
        return {
            "students": 0,
            "avg_score": 0.0,
            "pass_rate": 0.0,
            "avg_attendance": 0.0,
            "at_risk": 0,
            "at_risk_rate": 0.0,
        }
    return {
        "students": int(len(frame)),
        "avg_score": round(float(frame["total_score"].mean()), 1),
        "pass_rate": round(float(frame["passed"].mean()) * 100, 1),
        "avg_attendance": round(float(frame["attendance_pct"].mean()), 1),
        "at_risk": int(frame["at_risk"].sum()),
        "at_risk_rate": round(float(frame["at_risk"].mean()) * 100, 1),
    }


def department_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-department averages, sorted by average score."""
    if frame.empty:
        return pd.DataFrame(columns=DEPARTMENT_COLUMNS)

    summary = (
        frame.groupby("department")
        .agg(
            students=("student_id", "count"),
            avg_score=("total_score", "mean"),
            pass_rate=("passed", "mean"),
            avg_attendance=("attendance_pct", "mean"),
            at_risk_rate=("at_risk", "mean"),
        )
        .reset_index()
    )
    summary["avg_score"] = summary["avg_score"].round(1)
    summary["avg_attendance"] = summary["avg_attendance"].round(1)
    summary["pass_rate"] = (summary["pass_rate"] * 100).round(1)
    summary["at_risk_rate"] = (summary["at_risk_rate"] * 100).round(1)
    return summary.sort_values("avg_score", ascending=False).reset_index(drop=True)


def semester_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """Average score and pass rate per semester."""
    if frame.empty:
        return pd.DataFrame(columns=SEMESTER_COLUMNS)

    trend = (
        frame.groupby("semester")
        .agg(
            students=("student_id", "count"),
            avg_score=("total_score", "mean"),
            pass_rate=("passed", "mean"),
        )
        .reset_index()
    )
    trend["avg_score"] = trend["avg_score"].round(1)
    trend["pass_rate"] = (trend["pass_rate"] * 100).round(1)
    return trend.sort_values("semester").reset_index(drop=True)


def attendance_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Pass rate grouped into attendance bands."""
    if frame.empty:
        return pd.DataFrame(columns=BAND_COLUMNS)

    bands = pd.cut(
        frame["attendance_pct"],
        bins=[0, 60, 75, 85, 95, 100.01],
        labels=["<60", "60-74", "75-84", "85-94", "95-100"],
        right=False,
        include_lowest=True,
    )
    summary = (
        frame.assign(band=bands)
        .groupby("band", observed=False)
        .agg(
            students=("student_id", "count"),
            pass_rate=("passed", "mean"),
            avg_score=("total_score", "mean"),
        )
        .reset_index()
    )
    summary["students"] = summary["students"].astype(int)
    summary["pass_rate"] = (summary["pass_rate"].fillna(0) * 100).round(1)
    summary["avg_score"] = summary["avg_score"].fillna(0).round(1)
    return summary


def correlation_with_score(frame: pd.DataFrame) -> pd.Series:
    """Pearson correlation of each numeric feature with ``total_score``."""
    columns = [
        "attendance_pct",
        "study_hours",
        "prior_score",
        "assignment_score",
        "exam_score",
        "backlogs",
        "total_score",
    ]
    if len(frame) < 2:
        return pd.Series(dtype=float)
    correlation = frame[columns].corr(numeric_only=True)["total_score"]
    return correlation.drop("total_score").sort_values(ascending=False).round(3)

"""Unit tests for the analytics layer (no Streamlit required)."""
import pandas as pd
import pytest

from src.analytics import (
    add_derived,
    apply_filters,
    attendance_bands,
    correlation_with_score,
    department_summary,
    grade_for,
    kpis,
    semester_trend,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": ["STU0001", "STU0002", "STU0003", "STU0004"],
            "department": ["CSE", "CSE", "ECE", "ECE"],
            "semester": [1, 1, 2, 2],
            "attendance_pct": [95.0, 60.0, 85.0, 70.0],
            "study_hours": [20.0, 5.0, 15.0, 8.0],
            "prior_score": [80.0, 45.0, 70.0, 50.0],
            "assignment_score": [90.0, 30.0, 75.0, 55.0],
            "exam_score": [88.0, 20.0, 70.0, 45.0],
            "backlogs": [0, 3, 0, 1],
        }
    )


def test_grade_bands():
    assert grade_for(95) == "A+"
    assert grade_for(80) == "A"
    assert grade_for(49) == "E"
    assert grade_for(39.9) == "F"


def test_add_derived_columns(frame):
    result = add_derived(frame)
    assert result.loc[0, "total_score"] == pytest.approx(88.8)
    assert result["grade"].tolist() == ["A", "F", "B", "E"]
    assert result["passed"].tolist() == [True, False, True, True]
    assert result["at_risk"].tolist() == [False, True, False, True]


def test_kpis(frame):
    stats = kpis(add_derived(frame))
    assert stats["students"] == 4
    assert stats["pass_rate"] == 75.0
    assert stats["avg_score"] == 58.5
    assert stats["at_risk"] == 2
    assert stats["at_risk_rate"] == 50.0


def test_filters_compose(frame):
    enriched = add_derived(frame)
    assert len(apply_filters(enriched, departments=["CSE"])) == 2
    assert len(apply_filters(enriched, only_at_risk=True)) == 2
    assert apply_filters(enriched, search="0003")["student_id"].tolist() == ["STU0003"]
    assert apply_filters(enriched, grades=["A"]).empty is False


def test_department_summary_sorted(frame):
    summary = department_summary(add_derived(frame))
    assert list(summary.columns) == [
        "department",
        "students",
        "avg_score",
        "pass_rate",
        "avg_attendance",
        "at_risk_rate",
    ]
    assert summary.loc[0, "department"] == "ECE"
    assert summary.loc[0, "pass_rate"] == 100.0


def test_semester_trend_and_bands(frame):
    enriched = add_derived(frame)
    trend = semester_trend(enriched)
    assert trend["semester"].tolist() == [1, 2]
    bands = attendance_bands(enriched)
    assert int(bands["students"].sum()) == len(enriched)


def test_correlation_returns_series(frame):
    correlation = correlation_with_score(add_derived(frame))
    assert "exam_score" in correlation.index
    assert "total_score" not in correlation.index


def test_missing_columns_raise(frame):
    with pytest.raises(ValueError):
        add_derived(frame.drop(columns=["exam_score"]))

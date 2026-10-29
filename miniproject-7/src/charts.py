"""Plotly figure builders for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

ACCENT = "#e4572e"
PRIMARY = "#2a9d8f"

CORRELATION_COLUMNS = [
    "attendance_pct",
    "study_hours",
    "prior_score",
    "assignment_score",
    "exam_score",
    "backlogs",
    "total_score",
]


def score_histogram(frame: pd.DataFrame):
    figure = px.histogram(
        frame,
        x="total_score",
        nbins=30,
        color="passed",
        color_discrete_map={True: PRIMARY, False: ACCENT},
        labels={"total_score": "Total score", "passed": "Passed"},
        title="Score distribution",
    )
    figure.update_layout(
        bargap=0.05,
        legend_title_text="Passed",
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return figure


def department_bar(summary: pd.DataFrame):
    figure = px.bar(
        summary.sort_values("avg_score"),
        x="avg_score",
        y="department",
        orientation="h",
        text="avg_score",
        color="pass_rate",
        color_continuous_scale="Teal",
        labels={
            "avg_score": "Average score",
            "department": "",
            "pass_rate": "Pass rate (%)",
        },
        title="Average score by department",
    )
    figure.update_traces(textposition="outside")
    figure.update_layout(
        xaxis_range=[0, 100],
        coloraxis_colorbar_title="Pass %",
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return figure


def trend_line(trend: pd.DataFrame):
    figure = px.line(
        trend,
        x="semester",
        y=["avg_score", "pass_rate"],
        markers=True,
        labels={
            "semester": "Semester",
            "value": "Score / pass rate",
            "variable": "Metric",
        },
        title="Semester trend",
    )
    figure.update_layout(
        yaxis_range=[0, 100],
        legend_title_text="",
        margin=dict(t=50, l=10, r=10, b=10),
    )
    return figure


def attendance_scatter(frame: pd.DataFrame):
    figure = px.scatter(
        frame,
        x="attendance_pct",
        y="total_score",
        color="grade",
        size="study_hours",
        hover_data=["student_id", "department", "semester"],
        labels={
            "attendance_pct": "Attendance (%)",
            "total_score": "Total score",
            "grade": "Grade",
            "study_hours": "Study hours",
        },
        title="Attendance vs total score",
        category_orders={"grade": ["A+", "A", "B", "C", "D", "E", "F"]},
    )
    figure.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return figure


def correlation_heatmap(frame: pd.DataFrame):
    correlation = frame[CORRELATION_COLUMNS].corr(numeric_only=True).round(2)
    figure = px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation matrix",
        aspect="auto",
    )
    figure.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    return figure

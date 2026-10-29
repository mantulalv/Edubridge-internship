"""Student performance analytics dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import (
    GRADE_ORDER,
    apply_filters,
    attendance_bands,
    correlation_with_score,
    department_summary,
    kpis,
    load_students,
    semester_trend,
)
from src.charts import (
    attendance_scatter,
    correlation_heatmap,
    department_bar,
    score_histogram,
    trend_line,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "students.csv"

st.set_page_config(page_title="Student Analytics", page_icon="🎓", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(path: str, modified: float) -> pd.DataFrame:
    return load_students(path)


if not DATA_PATH.exists():
    st.error(
        "Missing `data/students.csv`. Generate it with:\n\n"
        "```\npython data/generate_data.py\n```"
    )
    st.stop()

frame = get_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

st.title("🎓 Student Performance Dashboard")
st.caption(
    f"{len(frame):,} synthetic student records · filters apply to every chart and table"
)

with st.sidebar:
    st.header("Filters")
    departments = st.multiselect(
        "Department",
        sorted(frame["department"].unique()),
        default=sorted(frame["department"].unique()),
    )
    semesters = st.multiselect(
        "Semester",
        sorted(frame["semester"].unique().tolist()),
        default=sorted(frame["semester"].unique().tolist()),
    )
    grades = st.multiselect("Grade", GRADE_ORDER, default=GRADE_ORDER)
    search = st.text_input("Student ID contains")
    only_risk = st.toggle("At-risk students only", value=False)

filtered = apply_filters(
    frame,
    departments=departments,
    semesters=semesters,
    grades=grades,
    search=search,
    only_at_risk=only_risk,
)

if filtered.empty:
    st.warning("No students match the current filters.")
    st.stop()

stats = kpis(filtered)
metrics = st.columns(5)
metrics[0].metric("Students", f"{stats['students']:,}")
metrics[1].metric("Average score", f"{stats['avg_score']:.1f}")
metrics[2].metric("Pass rate", f"{stats['pass_rate']:.1f}%")
metrics[3].metric("Average attendance", f"{stats['avg_attendance']:.1f}%")
metrics[4].metric(
    "At risk",
    f"{stats['at_risk']:,}",
    f"{stats['at_risk_rate']:.1f}%",
    delta_color="inverse",
)

overview, departments_tab, students_tab, insights_tab = st.tabs(
    ["Overview", "Departments", "Students", "Insights"]
)

with overview:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(score_histogram(filtered), use_container_width=True)
    with right:
        st.plotly_chart(trend_line(semester_trend(filtered)), use_container_width=True)
    st.plotly_chart(attendance_scatter(filtered), use_container_width=True)

with departments_tab:
    summary = department_summary(filtered)
    st.plotly_chart(department_bar(summary), use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

with students_tab:
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        "students_filtered.csv",
        "text/csv",
    )
    st.dataframe(
        filtered.sort_values("total_score", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=460,
    )

with insights_tab:
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(correlation_heatmap(filtered), use_container_width=True)
    with right:
        st.subheader("What correlates with score?")
        correlation = correlation_with_score(filtered)
        st.dataframe(
            correlation.rename("correlation").to_frame(),
            use_container_width=True,
        )
        st.subheader("Attendance bands")
        st.dataframe(attendance_bands(filtered), use_container_width=True, hide_index=True)

"""Chart grid rendering for the AI dashboard (BI-style 2-column layout)."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

try:
    from dashboard.charts import build_figure_for_chart
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.charts import build_figure_for_chart


def _render_table(chart: dict[str, Any]) -> None:
    """Render a tabular chart payload as a data frame."""
    data = chart.get("data")
    if not isinstance(data, list) or not data:
        st.warning("No rows to display.")
        return
    frame = pd.DataFrame(data)
    st.dataframe(frame, use_container_width=True, height=320, hide_index=True)


def _render_tile(chart: dict[str, Any]) -> None:
    chart_type = str(chart.get("chart_type", "")).strip().lower()
    title = html.escape(str(chart.get("title", "Chart")))
    subtitle = html.escape(str(chart.get("subtitle", "") or ""))
    st.markdown('<div class="ada-bi-chart-tile">', unsafe_allow_html=True)
    st.markdown(f'<div class="ada-bi-chart-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="ada-bi-chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)

    if chart_type == "table":
        _render_table(chart)
    else:
        figure = build_figure_for_chart(chart)
        if figure is None:
            st.markdown(
                '<div class="ada-bi-empty">Chart unavailable for the current data shape.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
            )
    data_note = str(chart.get("data_note", "") or "")
    if data_note:
        st.caption(data_note)
    st.markdown("</div>", unsafe_allow_html=True)


def render_chart_section_header(title: str) -> None:
    """Render a BI-style section header."""
    safe_title = html.escape(title)
    st.markdown(f'<div class="ada-bi-section">{safe_title}</div>', unsafe_allow_html=True)


def render_chart_grid(charts: list[dict[str, Any]]) -> None:
    """Render charts in a responsive 2-column grid respecting width hints."""
    useful = [c for c in charts if isinstance(c, dict)]
    if not useful:
        st.markdown(
            '<div class="ada-bi-empty">No charts could be built for the current filter '
            "selection. Adjust filters or regenerate the dashboard.</div>",
            unsafe_allow_html=True,
        )
        return

    row: list[dict[str, Any]] = []
    for chart in useful:
        row.append(chart)
        width = sum(int(c.get("width_span", 6)) for c in row)
        if width >= 12:
            _render_row(row)
            row = []
    if row:
        _render_row(row)


def _render_row(row: list[dict[str, Any]]) -> None:
    if len(row) == 1:
        _render_tile(row[0])
        return
    cols = st.columns(2, gap="large")
    for col, chart in zip(cols, row):
        with col:
            _render_tile(chart)
"""Main Streamlit renderer for the AI-planned dynamic dashboard.

Orchestrates the BI workspace: header, filter bar, KPI grid, chart grid, AI
insights and exports. Only the engine recomputes on filter changes; the
configuration and LLM are untouched.
"""

from __future__ import annotations

import csv
import html
import io
import json
from typing import Any

import pandas as pd
import streamlit as st

try:
    from dashboard.engine import compute_dashboard, compute_insight_facts
    from dashboard.filters import build_filter_options
    from dashboard.insights import build_dashboard_insights, refine_insights_with_llm
    from dashboard.render.charts import render_chart_grid
    from dashboard.render.filters import render_filter_bar
    from dashboard.render.insights import render_insight_cards
    from dashboard.render.kpis import render_kpi_grid
    from dashboard.render.theme import inject_dashboard_styles
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.engine import compute_dashboard, compute_insight_facts
    from project.dashboard.filters import build_filter_options
    from project.dashboard.insights import build_dashboard_insights, refine_insights_with_llm
    from project.dashboard.render import charts as _render_charts
    from project.dashboard.render import filters as _render_filters
    from project.dashboard.render import insights as _render_insights
    from project.dashboard.render import kpis as _render_kpis
    from project.dashboard.render import theme as _render_theme

    render_chart_grid = _render_charts.render_chart_grid
    render_filter_bar = _render_filters.render_filter_bar
    render_insight_cards = _render_insights.render_insight_cards
    render_kpi_grid = _render_kpis.render_kpi_grid
    inject_dashboard_styles = _render_theme.inject_dashboard_styles


@st.cache_data(show_spinner=False, max_entries=16)
def _compute_payload(config: dict[str, Any], df: pd.DataFrame, active: dict[str, Any]):
    """Compute the runtime payload + insight facts for the current view (cached)."""
    runtime = compute_dashboard(config, df, active)
    facts = compute_insight_facts(runtime.get("filtered_df", df), config, runtime.get("kpis", []))
    return runtime, facts


@st.cache_data(show_spinner="Generating AI insights...", max_entries=8)
def _llm_insights(config: dict[str, Any], active: dict[str, Any], facts: dict[str, Any]):
    """Compute LLM-narrated insights for the *current* filtered view (cached)."""
    deterministic = build_dashboard_insights(facts, config)
    return refine_insights_with_llm(facts, config, deterministic, active)


def _config_signature(config: dict[str, Any]) -> str:
    """Stable fingerprint used to detect a new dashboard (new dataset/session)."""
    keys = (
        "generated_at",
        "title",
        "data_source",
        "row_count",
    )
    return "|".join(f"{key}={config.get(key, '')}" for key in keys)
def _render_header(config: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Render the dashboard header (title, badges, active filter chips)."""
    title = html.escape(str(config.get("title") or "AI Analytics Dashboard"))
    subtitle = html.escape(str(config.get("subtitle") or "") or "")
    source = html.escape(str(config.get("data_source") or "").upper() or "DATASET")
    rows = int(runtime.get("row_count") or 0)
    meta = []
    if rows:
        meta.append(f"{rows:,} rows")
    time_dim = config.get("time_dimension")
    if time_dim:
        meta.append(f"time: {html.escape(str(time_dim))}")

    st.markdown(
        '<div class="ada-bi-header">'
        f'<span class="ada-bi-title">{title}</span>'
        f'<span class="ada-bi-badge">● {source}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    subtitle_bits = [subtitle] + meta
    st.markdown(
        f'<div class="ada-bi-subtitle">{" · ".join(bit for bit in subtitle_bits if bit)}</div>',
        unsafe_allow_html=True,
    )

    active_filters = st.session_state.get("dashboard_filters") or {}
    if active_filters:
        chips: list[str] = []
        for filt in config.get("filters") or []:
            filter_id = filt.get("id")
            if filter_id not in active_filters:
                continue
            value = active_filters[filter_id]
            label = html.escape(str(filt.get("label") or filt.get("column") or filter_id))
            rendered = _filter_value_text(filt, value)
            chips.append(
                f'<span class="ada-bi-filter-chip"><strong>{label}</strong> '
                f"{html.escape(str(rendered))}</span>"
            )
        if chips:
            st.markdown(f'<div>{"".join(chips)}</div>', unsafe_allow_html=True)


def _filter_value_text(filt: dict[str, Any], value: Any) -> str:
    """Human-readable text for an active filter value."""
    ftype = str(filt.get("type", ""))
    if value is None:
        return "All"
    if ftype == "numeric_range":
        if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value
        ):
            return f"{value[0]:,.2f} – {value[1]:,.2f}"
        return str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _render_export_bar(config: dict[str, Any], runtime: dict[str, Any]) -> None:
    """Render download buttons for the current view (KPI CSV + config JSON)."""
    kpi_csv = _build_kpi_csv(runtime.get("kpis", []))
    config_json = json.dumps(config, indent=2, ensure_ascii=False, default=str)
    col1, col2 = st.columns(2)
    col1.download_button(
        "↓ KPI data (CSV)",
        data=kpi_csv,
        file_name="dashboard_kpis.csv",
        mime="text/csv",
        use_container_width=True,
    )
    col2.download_button(
        "↓ Dashboard spec (JSON)",
        data=config_json,
        file_name="dashboard_spec.json",
        mime="application/json",
        use_container_width=True,
    )


def _build_kpi_csv(kpis: list[dict[str, Any]]) -> str:
    """Serialize the KPI row into a small CSV string."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["label", "value", "delta", "sub"])
    for kpi in kpis:
        writer.writerow(
            [
                str(kpi.get("label", "")),
                str(kpi.get("value", "")),
                str(kpi.get("delta") or ""),
                str(kpi.get("sub", "")),
            ]
        )
    return buffer.getvalue()


def render_ask_ai_box() -> str:
    """Render the 'Ask AI about this view' field; returns the submitted query or ''."""
    st.markdown('<div class="ada-bi-section">💬 Ask AI about this view</div>', unsafe_allow_html=True)
    with st.form("ada_dashboard_ask_form"):
        question = st.text_input(
            "Ask a question about the current filters (e.g. 'Why is profit low?' or "
            "'Which segment drives sales?')",
            placeholder="Your question...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask AI", use_container_width=False)
    if submitted and question.strip():
        return question.strip()
    return ""
def render_ai_dashboard(config: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Render the full AI analytics dashboard and return the current view state.

    Returns ``{"active_filters": dict, "runtime": dict, "ask_query": str}`` so the
    main app can build the dashboard chat context and trigger follow-up analysis.
    """
    inject_dashboard_styles()

    if not isinstance(config, dict) or not config.get("kpis"):
        st.markdown(
            '<div class="ada-bi-empty">No dashboard configuration is available. '
            "Upload a dataset and build a dashboard to continue.</div>",
            unsafe_allow_html=True,
        )
        return {"active_filters": {}, "runtime": {}, "ask_query": ""}

    if "dashboard_filters" not in st.session_state:
        st.session_state["dashboard_filters"] = {}
    signature = _config_signature(config)
    if st.session_state.get("ada_dashboard_signature") != signature:
        st.session_state["ada_dashboard_signature"] = signature
        st.session_state["dashboard_filters"] = {}

    filter_options = build_filter_options(df, config.get("filters") or [])
    active_filters = render_filter_bar(config.get("filters") or [], filter_options)
    st.session_state["dashboard_filters"] = active_filters

    runtime, facts = _compute_payload(config, df, active_filters)
    _render_header(config, runtime)
    _render_export_bar(config, runtime)
    ask_query = render_ask_ai_box()

    st.markdown('<div class="ada-bi-section">📊 KPI Summary</div>', unsafe_allow_html=True)
    render_kpi_grid(runtime.get("kpis", []))

    render_chart_grid(runtime.get("charts", []))

    _render_insights(config, active_filters, facts)

    return {"active_filters": active_filters, "runtime": runtime, "ask_query": ask_query}


def _render_insights(
    config: dict[str, Any], active_filters: dict[str, Any], facts: dict[str, Any]
) -> None:
    """Render deterministic insights plus an optional LLM-narrated pass."""
    deterministic = build_dashboard_insights(facts, config)
    if not deterministic:
        return

    signature = json.dumps(active_filters, sort_keys=True, default=str)
    if st.session_state.get("ada_insights_filters") != signature:
        st.session_state["ada_insights_filters"] = signature
        st.session_state["ada_insights_list"] = None

    regenerate = st.button("✨ Generate AI narrative insights", key="ada_ai_insights_btn")
    if regenerate:
        with st.spinner("Analyzing the current view..."):
            llm_list = _llm_insights(config, active_filters, facts)
        st.session_state["ada_insights_filters"] = signature
        st.session_state["ada_insights_list"] = llm_list

    insights = st.session_state.get("ada_insights_list") or deterministic
    render_insight_cards(insights)
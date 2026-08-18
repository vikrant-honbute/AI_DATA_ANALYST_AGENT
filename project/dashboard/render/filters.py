"""Filter bar rendering for the AI dashboard.

Renders the appropriate Streamlit widget for each filter definition and returns
the active filter selections as a dict (filter id -> value). Selections persist in
``st.session_state["dashboard_filters"]``; changing a filter triggers a recompute
of only the runtime payload (the configuration and LLM are untouched).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

_ALL = "All"


def _default_value(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    """Return the widget default (no-filter) value for a filter."""
    ftype = str(filt.get("type", "categorical_multi"))
    if ftype == "categorical_single":
        return _ALL
    if ftype == "numeric_range":
        return (options.get("min"), options.get("max"))
    if ftype == "date_range":
        return (options.get("min"), options.get("max"))
    return []  # multiselect: empty = no filter


def _render_date_year(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    years = sorted({int(v) for v in options.get("options", []) if v is not None})
    if not years:
        return None
    selected = st.multiselect(
        "Year",
        options=years,
        default=[],
        key=f"ada_filt_{filt.get('id')}",
        placeholder="All years",
    )
    # Selecting all years is equivalent to no filter.
    if set(selected) >= set(years):
        return None
    return selected or None


def _render_categorical_single(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    values = [str(v) for v in options.get("options", [])]
    selected = st.selectbox(
        str(filt.get("label", "Category")),
        [_ALL] + values,
        key=f"ada_filt_{filt.get('id')}",
    )
    return None if selected == _ALL else selected


def _render_categorical_multi(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    values = [str(v) for v in options.get("options", [])]
    selected = st.multiselect(
        str(filt.get("label", "Category")),
        options=values,
        default=[],
        key=f"ada_filt_{filt.get('id')}",
        placeholder=f"All {str(filt.get('label', '')).lower() or 'values'}",
    )
    if selected and set(selected) >= set(values):
        return None
    return selected or None


def _render_numeric_range(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    minimum = options.get("min")
    maximum = options.get("max")
    if minimum is None or maximum is None:
        return None
    minimum, maximum = float(minimum), float(maximum)
    lo, hi = st.slider(
        str(filt.get("label", "Range")),
        min_value=float(minimum),
        max_value=float(maximum),
        value=(float(minimum), float(maximum)),
        key=f"ada_filt_{filt.get('id')}",
    )
    # Full range selection is equivalent to no filter.
    if (lo, hi) == (round(float(minimum), 2), round(float(maximum), 2)):
        return None
    return (lo, hi)


def _render_date_range(filt: dict[str, Any], options: dict[str, Any]) -> Any:
    minimum = options.get("min")
    maximum = options.get("max")
    if not minimum or not maximum:
        return None
    try:
        start = pd.to_datetime(str(minimum)).date()
        end = pd.to_datetime(str(maximum)).date()
    except Exception:
        return None
    selected = st.date_input(
        str(filt.get("label", "Date range")),
        value=(start, end),
        min_value=start,
        max_value=end,
        key=f"ada_filt_{filt.get('id')}",
    )
    if isinstance(selected, (tuple, list)) and len(selected) == 2:
        lo, hi = selected
        if (lo, hi) == (start, end):
            return None
        return (str(lo), str(hi))
    if selected is not None:
        # Single-date selection → apply that exact day.
        value_str = str(selected)
        return (value_str, value_str)
    return None


_RENDERERS = {
    "date_year": _render_date_year,
    "categorical_single": _render_categorical_single,
    "categorical_multi": _render_categorical_multi,
    "numeric_range": _render_numeric_range,
    "date_range": _render_date_range,
}


def render_filter_bar(
    filter_specs: list[dict[str, Any]], filter_options: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Render the filter widgets and return the active filter selections."""
    specs = [f for f in filter_specs if isinstance(f, dict) and f.get("column")]
    if not specs:
        return {}

    st.markdown('<div class="ada-bi-filters">', unsafe_allow_html=True)

    reset_col, chip_col = st.columns([1, 3])
    reset_clicked = reset_col.button("↺ Reset filters", key="ada_filters_reset", use_container_width=True)
    chip_col.markdown(
        '<div style="font-size:12px;color:#9CA3AF;padding-top:6px">FILTERS · apply to all KPIs, '
        "charts and insights</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        for filt in specs:
            st.session_state[f"ada_filt_{filt.get('id')}"] = _default_value(
                filt, filter_options.get(filt.get("id"), {})
            )
        st.session_state["dashboard_filters"] = {}

    active: dict[str, Any] = {}
    chunks = [specs[i : i + 4] for i in range(0, len(specs), 4)]
    for chunk in chunks:
        cols = st.columns(len(chunk))
        for col, filt in zip(cols, chunk):
            with col:
                renderer = _RENDERERS.get(str(filt.get("type", "")).lower())
                if renderer is None:
                    continue
                options = filter_options.get(filt.get("id"), {})
                value = renderer(filt, options)
                if value is not None:
                    active[filt.get("id")] = value

    if active:
        st.session_state["dashboard_filters"] = active
    return active
"""KPI card rendering for the AI dashboard (BI-style cards)."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

_SUCCESS_CACHE = {"success", "danger", "info", "warning", "neutral"}


def _delta_parts(value: Any) -> tuple[str, str]:
    """Return (css_class, text) for a KPI delta string like '+3.2%' or '▼ 5%'."""
    text = str(value or "").strip()
    if not text or text.lower() == "false":
        return "", ""
    if text.startswith(("▲", "+")):
        return "delta-up", text
    if text.startswith(("▼", "-")):
        return "delta-down", text
    return "delta-flat", text


def render_kpi_grid(kpis: list[dict[str, Any]]) -> None:
    """Render the KPI card row."""
    useful = [kpi for kpi in kpis if isinstance(kpi, dict) and kpi.get("value")]
    if not useful:
        return

    cards: list[str] = []
    for kpi in useful:
        tone = str(kpi.get("tone", "neutral")).lower()
        if tone not in _SUCCESS_CACHE:
            tone = "neutral"
        label = html.escape(str(kpi.get("label", "")))
        value = html.escape(str(kpi.get("value", "")))
        sub = html.escape(str(kpi.get("sub", "") or ""))
        icon = html.escape(str(kpi.get("icon", "") or ""))
        delta_class, delta_text = _delta_parts(kpi.get("delta"))
        delta_html = (
            f'<div class="ada-bi-kpi-delta {delta_class}">{html.escape(delta_text)}</div>'
            if delta_text
            else ""
        )
        sub_html = f'<div class="ada-bi-kpi-label" style="font-size:11px">{sub}</div>' if sub else ""
        cards.append(
            f'<div class="ada-bi-kpi-card">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            f'<span style="font-size:18px">{icon}</span>'
            f'<span class="ada-bi-kpi-label">{label}</span>'
            f"</div>"
            f'<div class="ada-bi-kpi-value tone-{tone}">{value}</div>'
            f"{delta_html}{sub_html}"
            f"</div>"
        )

    st.markdown(f'<div class="ada-bi-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
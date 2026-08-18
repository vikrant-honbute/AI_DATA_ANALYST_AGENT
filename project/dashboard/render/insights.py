"""AI Insights section rendering for the dashboard."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def render_insight_cards(insights: list[dict[str, Any]]) -> None:
    """Render insight cards with a small header."""
    useful = [item for item in insights if isinstance(item, dict) and item.get("body")]
    if not useful:
        return

    st.markdown('<div class="ada-bi-section">💡 AI Insights</div>', unsafe_allow_html=True)
    for item in useful:
        title = html.escape(str(item.get("title", "Insight")))
        body = html.escape(str(item.get("body", "")))
        st.markdown(
            '<div class="ada-bi-insight">'
            f'<div class="ada-bi-insight-title">{title}</div>'
            f'<div class="ada-bi-insight-body">{body}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="font-size:12px;color:#6B7280;margin-top:-4px;margin-bottom:16px">'
        "Insights are computed from the current filtered data.</div>",
        unsafe_allow_html=True,
    )
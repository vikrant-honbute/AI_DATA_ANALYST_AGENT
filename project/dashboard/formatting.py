"""Shared formatting and theme helpers for the dashboard engine.

These helpers keep the deterministic dashboard engine and the Plotly figure
builders visually consistent with the Streamlit app's premium dark theme.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------- Theme constants (mirrors the Streamlit app) ----------

PLOTLY_CARD_BG = "#111827"
PLOTLY_GRID = "#1F2937"
PLOTLY_TEXT = "#E5E7EB"
PLOTLY_MUTED = "#9CA3AF"
PLOTLY_BLUE = "#3B82F6"
PLOTLY_BLUE_LIGHT = "#60A5FA"
PLOTLY_BLUE_PALE = "#93C5FD"
PLOTLY_GREEN = "#22C55E"
PLOTLY_RED = "#EF4444"
PLOTLY_BLUE_PALETTE = (PLOTLY_BLUE, PLOTLY_BLUE_LIGHT, PLOTLY_BLUE_PALE, "#7DD3FC", "#2563EB")
PLOTLY_FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
PLOTLY_CHART_HEIGHT = 380


def prettify_name(name: Any) -> str:
    """Convert a column/step name into a readable title ('total_profit' -> 'Total Profit')."""
    text = re.sub(r"[\-_]+", " ", str(name))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def looks_like_money(name: Any) -> bool:
    """Return True when a metric name suggests a currency value."""
    lowered = str(name).lower()
    tokens = [
        "revenue", "sales", "profit", "amount", "income", "price",
        "cost", "spend", "budget", "salary", "fee", "value",
    ]
    return any(token in lowered for token in tokens)


def format_compact(value: Any) -> str:
    """Format a number compactly: 108000 -> '108K', 1200000 -> '1.2M'."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    if abs_value >= 100:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def format_value(value: Any, money: bool = False) -> str:
    """Format a value compactly for chart labels: 108000 -> '$108K'."""
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    prefix = "$" if money else ""
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{prefix}{value / 1_000:.0f}K"
    if float(value).is_integer():
        return f"{prefix}{int(value):,}"
    return f"{prefix}{value:,.2f}"


def format_kpi_value(value: Any, money: bool = False) -> str:
    """Format a KPI value for card display with an optional currency prefix."""
    prefix = "$" if money else ""
    return f"{prefix}{format_compact(value)}"


def highlight_colors(values: list[float]) -> list[str]:
    """Green for the best value, red for the worst, blue palette for the rest."""
    if not values:
        return []
    best_index = max(range(len(values)), key=lambda i: values[i])
    worst_index = min(range(len(values)), key=lambda i: values[i])
    colors: list[str] = []
    for index in range(len(values)):
        if index == best_index:
            colors.append(PLOTLY_GREEN)
        elif index == worst_index:
            colors.append(PLOTLY_RED)
        else:
            colors.append(PLOTLY_BLUE_PALETTE[index % len(PLOTLY_BLUE_PALETTE)])
    return colors


def classify_trend(series: pd.Series) -> str:
    """Classify a numeric series trend using monotonic agreement, not just endpoints."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return "stable"
    if float(values.std()) == 0:
        return "stable"

    import numpy as np

    x = np.arange(len(values))
    correlation = float(np.corrcoef(x, values)[0, 1])
    first = float(values.iloc[0])
    relative_change = (float(values.iloc[-1]) - first) / max(abs(first), 1e-9)

    if correlation >= 0.5 and relative_change >= 0.05:
        return "rising"
    if correlation <= -0.5 and relative_change <= -0.05:
        return "declining"
    return "mixed"

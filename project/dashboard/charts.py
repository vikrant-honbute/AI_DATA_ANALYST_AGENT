"""Plotly figure builders for dashboard chart specs.

Each function takes a normalized chart spec (chart_type + embedded data
records) and returns a dark-theme Plotly figure matching the Streamlit app's
visual language.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

try:
    from dashboard.formatting import (
        PLOTLY_BLUE,
        PLOTLY_CARD_BG,
        PLOTLY_CHART_HEIGHT,
        PLOTLY_FONT,
        PLOTLY_GRID,
        PLOTLY_GREEN,
        PLOTLY_MUTED,
        PLOTLY_RED,
        PLOTLY_TEXT,
        format_value,
        highlight_colors,
        looks_like_money,
        prettify_name,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.formatting import (
        PLOTLY_BLUE,
        PLOTLY_CARD_BG,
        PLOTLY_CHART_HEIGHT,
        PLOTLY_FONT,
        PLOTLY_GRID,
        PLOTLY_GREEN,
        PLOTLY_MUTED,
        PLOTLY_RED,
        PLOTLY_TEXT,
        format_value,
        highlight_colors,
        looks_like_money,
        prettify_name,
    )


def base_layout(x_title: str = "", y_title: str = "") -> dict[str, Any]:
    """Dark-theme layout dict matching the Streamlit app charts."""
    return dict(
        height=PLOTLY_CHART_HEIGHT,
        margin=dict(l=64, r=20, t=20, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOTLY_CARD_BG,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=12),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=PLOTLY_MUTED, size=12)),
            tickfont=dict(color=PLOTLY_MUTED, size=11),
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            linecolor=PLOTLY_GRID,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=PLOTLY_MUTED, size=12)),
            tickfont=dict(color=PLOTLY_MUTED, size=11),
            gridcolor=PLOTLY_GRID,
            zeroline=False,
            linecolor=PLOTLY_GRID,
        ),
        hoverlabel=dict(
            bgcolor="#1F2937",
            bordercolor="#374151",
            font=dict(color="#FFFFFF", size=12),
        ),
        showlegend=False,
    )


def _records_to_frame(chart: dict[str, Any]) -> pd.DataFrame | None:
    """Convert embedded chart records into a DataFrame."""
    data = chart.get("data")
    if not isinstance(data, list) or not data:
        return None
    try:
        return pd.DataFrame(data)
    except Exception:
        return None


def _figure_line(df: pd.DataFrame, chart: dict[str, Any]) -> go.Figure | None:
    """Trend line with highlighted peak and trough markers."""
    x_key, y_key = chart.get("x") or "x", chart.get("y") or "value"
    if x_key not in df.columns or y_key not in df.columns:
        return None
    frame = df[[x_key, y_key]].copy()
    frame[y_key] = pd.to_numeric(frame[y_key], errors="coerce")
    frame = frame.dropna().sort_values(x_key)
    if frame.empty:
        return None

    x_values = [str(value) for value in frame[x_key].tolist()]
    y_values = [float(value) for value in frame[y_key].tolist()]
    metric_label = prettify_name(y_key)
    money = looks_like_money(metric_label)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(color=PLOTLY_BLUE, width=3),
            marker=dict(size=6, color=PLOTLY_BLUE),
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    if len(y_values) >= 2:
        best_index = max(range(len(y_values)), key=lambda i: y_values[i])
        worst_index = min(range(len(y_values)), key=lambda i: y_values[i])
        if best_index != worst_index:
            for index, color, label in (
                (best_index, PLOTLY_GREEN, "Peak"),
                (worst_index, PLOTLY_RED, "Trough"),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=[x_values[index]],
                        y=[y_values[index]],
                        mode="markers",
                        name=label,
                        marker=dict(
                            size=11, symbol="diamond", color=color,
                            line=dict(color=PLOTLY_TEXT, width=1),
                        ),
                        hovertemplate=f"{label}: %{y:,.2f}<extra></extra>",
                    )
                )
        if len(y_values) <= 24:
            show_text = [None] * len(y_values)
            show_text[best_index] = format_value(y_values[best_index], money)
            show_text[worst_index] = format_value(y_values[worst_index], money)
            figure.data[0].text = show_text
            figure.data[0].textposition = "top center"
            figure.data[0].textfont = dict(size=10, color=PLOTLY_MUTED)

    x_title = {"period": "Period", "date": "Date", "time": "Time"}.get(
        str(x_key).lower(), prettify_name(x_key)
    )
    figure.update_layout(**base_layout(x_title=x_title, y_title=metric_label))
    figure.update_layout(legend=dict(font=dict(color=PLOTLY_MUTED), bgcolor="rgba(0,0,0,0)"))
    return figure


def _figure_bar(df: pd.DataFrame, chart: dict[str, Any]) -> go.Figure | None:
    """Modern bar chart with value labels and best/worst highlighting."""
    x_key, y_key = chart.get("x") or "x", chart.get("y") or "value"
    if x_key not in df.columns or y_key not in df.columns:
        return None
    frame = df[[x_key, y_key]].copy()
    frame[y_key] = pd.to_numeric(frame[y_key], errors="coerce")
    frame = frame.dropna().sort_values(y_key, ascending=False)
    if frame.empty:
        return None

    labels = [str(value) for value in frame[x_key].tolist()]
    values = [float(value) for value in frame[y_key].tolist()]
    metric_label = prettify_name(y_key)
    money = looks_like_money(metric_label)

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=highlight_colors(values),
            text=[format_value(value, money) for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(**base_layout(x_title=prettify_name(x_key), y_title=metric_label))
    if len(labels) > 12:
        figure.update_xaxes(tickangle=45, tickfont=dict(size=10))
    return figure


def _figure_donut(df: pd.DataFrame, chart: dict[str, Any]) -> go.Figure | None:
    """Donut chart for categorical share."""
    x_key, y_key = chart.get("x") or "x", chart.get("y") or "value"
    if x_key not in df.columns or y_key not in df.columns:
        return None
    frame = df[[x_key, y_key]].copy()
    frame[y_key] = pd.to_numeric(frame[y_key], errors="coerce")
    frame = frame.dropna().sort_values(y_key, ascending=False)
    if frame.empty:
        return None

    labels = [str(value) for value in frame[x_key].tolist()]
    values = [float(value) for value in frame[y_key].tolist()]
    total = sum(values)
    metric_label = prettify_name(y_key)
    money = looks_like_money(metric_label)

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=highlight_colors(values), line=dict(color=PLOTLY_CARD_BG, width=2)),
            textinfo="percent",
            textfont=dict(color=PLOTLY_TEXT, size=11),
            hovertemplate="%{label}: %{y:,.2f} (%{percent})<extra></extra>",
            sort=False,
        )
    )
    figure.update_layout(
        height=PLOTLY_CHART_HEIGHT,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOTLY_CARD_BG,
        font=dict(family=PLOTLY_FONT, color=PLOTLY_TEXT, size=12),
        hoverlabel=dict(
            bgcolor="#1F2937", bordercolor="#374151", font=dict(color="#FFFFFF", size=12)
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02, xanchor="center", x=0.5,
            font=dict(color=PLOTLY_MUTED, size=11),
        ),
    )
    if total:
        figure.add_annotation(
            text=f"Total<br><b>{format_value(total, money)}</b>",
            showarrow=False,
            font=dict(color=PLOTLY_TEXT, size=14, family=PLOTLY_FONT),
        )
    return figure


def _figure_hist(df: pd.DataFrame, chart: dict[str, Any]) -> go.Figure | None:
    """Histogram for a single numeric metric."""
    x_key = chart.get("x") or "value"
    if x_key not in df.columns:
        return None
    values = pd.to_numeric(df[x_key], errors="coerce").dropna()
    if values.empty:
        return None

    metric_label = prettify_name(x_key)
    figure = go.Figure(
        go.Histogram(
            x=[float(v) for v in values.tolist()],
            nbinsx=min(24, max(8, int(len(values) ** 0.5))),
            marker_color=PLOTLY_BLUE,
            marker_line_color=PLOTLY_BLUE,
            opacity=0.85,
            hovertemplate="bin: %{x}<br>count: %{y}<extra></extra>",
        )
    )
    figure.update_layout(**base_layout(x_title=metric_label, y_title="Records"))
    return figure


def _figure_heatmap(df: pd.DataFrame, chart: dict[str, Any]) -> go.Figure | None:
    """Correlation heatmap built from {x, y, z} cell records."""
    if "z" not in df.columns or "x" not in df.columns or "y" not in df.columns:
        return None
    frame = df[["x", "y", "z"]].dropna(subset=["z"])
    if frame.empty:
        return None

    rows = list(dict.fromkeys(str(v) for v in frame["x"].tolist()))
    cols = list(dict.fromkeys(str(v) for v in frame["y"].tolist()))
    lookup = {
        (str(r), str(c)): float(z)
        for r, c, z in zip(frame["x"], frame["y"], frame["z"])
    }
    matrix = [[lookup.get((row, col)) for col in cols] for row in rows]

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=cols,
            y=rows,
            text=[[None if z is None else f"{z:.2f}" for z in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=10, color=PLOTLY_TEXT),
            colorscale=[[0.0, "#0B1220"], [0.5, "#1E3A8A"], [1.0, PLOTLY_BLUE]],
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
            colorbar=dict(thickness=12, tickfont=dict(color=PLOTLY_MUTED, size=10), outlinewidth=0),
        )
    )
    figure.update_layout(**base_layout(x_title="", y_title=""))
    if len(rows) > 8:
        figure.update_xaxes(tickangle=45, tickfont=dict(size=9))
        figure.update_yaxes(tickfont=dict(size=9))
    return figure


def build_figure_for_chart(chart: dict[str, Any]) -> go.Figure | None:
    """Build a Plotly figure from one dashboard chart spec entry."""
    if not isinstance(chart, dict):
        return None
    chart_type = str(chart.get("chart_type", "")).strip().lower()
    frame = _records_to_frame(chart)
    if frame is None or frame.empty:
        return None

    builders = {
        "line": _figure_line,
        "bar": _figure_bar,
        "donut": _figure_donut,
        "hist": _figure_hist,
        "heatmap": _figure_heatmap,
    }
    builder = builders.get(chart_type)
    if builder is None:
        return None
    try:
        return builder(frame, chart)
    except Exception:
        return None
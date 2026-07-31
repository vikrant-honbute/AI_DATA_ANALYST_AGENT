"""Streamlit UI for running the AI Data Analysis Agent graph."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import get_settings
from graph import build_workflow
from graph.state import AgentState

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback to static chart images.
    _PLOTLY_AVAILABLE = False

_PLOTLY_CARD_BG = "#111827"
_PLOTLY_GRID = "#1F2937"
_PLOTLY_TEXT = "#E5E7EB"
_PLOTLY_MUTED = "#9CA3AF"
_PLOTLY_BLUE = "#3B82F6"
_PLOTLY_BLUE_LIGHT = "#60A5FA"
_PLOTLY_BLUE_PALE = "#93C5FD"
_PLOTLY_GREEN = "#22C55E"
_PLOTLY_RED = "#EF4444"
_PLOTLY_BLUE_PALETTE = (_PLOTLY_BLUE, _PLOTLY_BLUE_LIGHT, _PLOTLY_BLUE_PALE, "#7DD3FC", "#2563EB")
_PLOTLY_CHART_HEIGHT = 420
_PLOTLY_FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _build_initial_state(query: str, uploaded_df: pd.DataFrame | None) -> AgentState:
    """Build the graph state from UI inputs."""
    state: AgentState = {
        "query": query,
        "plan": [],
        "data_source": "csv",
        "intermediate_results": [],
        "final_result": "",
        "insights": "",
        "memory": [],
        "retry_count": 0,
    }

    if uploaded_df is not None:
        state["uploaded_dataframe"] = uploaded_df.copy(deep=True)

    return state


def _extract_result(final_state: AgentState) -> str:
    """Return final result text with a deterministic fallback."""
    raw_final_result = final_state.get("final_result", "")
    if isinstance(raw_final_result, str) and raw_final_result.strip():
        return raw_final_result.strip()
    if raw_final_result:
        return str(raw_final_result)

    intermediate = final_state.get("intermediate_results", [])
    if isinstance(intermediate, list) and intermediate:
        return str(intermediate[-1])

    return ""


def _extract_insights(final_state: AgentState) -> str:
    """Return insight text from final state."""
    raw_insights = final_state.get("insights", "")
    if isinstance(raw_insights, str):
        return _dedupe_bullet_lines(raw_insights)
    return _dedupe_bullet_lines(str(raw_insights))


def _dedupe_bullet_lines(text: str) -> str:
    """Remove repeated bullet lines so insight sections stay concise."""
    if not text.strip():
        return ""

    cleaned_lines: list[str] = []
    seen_bullets: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("-"):
            normalized = re.sub(r"\s+", " ", stripped.lower())
            if normalized in seen_bullets:
                continue
            seen_bullets.add(normalized)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _extract_step_results(final_state: AgentState) -> list[dict[str, Any]]:
    """Return structured intermediate step results from final state."""
    raw_results = final_state.get("intermediate_results", [])
    if not isinstance(raw_results, list):
        return []

    return [item for item in raw_results if isinstance(item, dict)]


def _count_step_errors(step_results: list[dict[str, Any]]) -> int:
    """Count execution steps that contain an error payload."""
    return sum(
        1
        for step in step_results
        if isinstance(step.get("error"), str) and step.get("error", "").strip()
    )


def _render_step_payload(result: Any) -> None:
    """Render one step payload in a readable visual format."""
    if isinstance(result, pd.DataFrame):
        st.caption(f"DataFrame: {len(result)} rows x {len(result.columns)} columns")
        st.dataframe(result, use_container_width=True)
        return

    if isinstance(result, pd.Series):
        st.caption(f"Series: {len(result)} items")
        st.dataframe(result.to_frame(name=result.name or "value"), use_container_width=True)
        return

    if isinstance(result, dict):
        if result.get("type") == "plot":
            st.json(_json_safe(result), expanded=False)
            raw_path = result.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = Path.cwd() / image_path
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
            return

        st.json(_json_safe(result), expanded=False)
        return

    if isinstance(result, (list, tuple)):
        st.json(_json_safe(result), expanded=False)
        return

    rendered = str(result).strip()
    if rendered:
        st.code(rendered, language="text")
    else:
        st.info("No result payload for this step.")


def _collect_chart_paths(intermediate_results: Any) -> list[dict[str, Any]]:
    """Collect plot files and metadata from intermediate step results."""
    if not isinstance(intermediate_results, list):
        return []

    charts: list[dict[str, Any]] = []
    for step_number, item in enumerate(intermediate_results, start=1):
        if not isinstance(item, dict):
            continue

        result = item.get("result")
        if not isinstance(result, dict):
            continue

        if result.get("type") != "plot":
            continue

        raw_path = result.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        chart_path = Path(raw_path)
        if not chart_path.is_absolute():
            chart_path = Path.cwd() / chart_path

        if not chart_path.exists():
            continue

        raw_chart_type = result.get("chart_type")
        chart_type = (
            raw_chart_type.strip().lower()
            if isinstance(raw_chart_type, str) and raw_chart_type.strip()
            else "unknown"
        )

        raw_rows = result.get("rows")
        row_count = int(raw_rows) if isinstance(raw_rows, int) else 0

        raw_columns = result.get("columns")
        columns = [str(col) for col in raw_columns] if isinstance(raw_columns, list) else []

        charts.append(
            {
                "step_number": step_number,
                "step": str(item.get("step", "")),
                "tool": str(item.get("tool", "")),
                "action": str(item.get("action", "")).strip(),
                "path": str(chart_path),
                "chart_type": chart_type,
                "rows": row_count,
                "columns": columns,
            }
        )

    return charts


def _result_to_dataframe(result: Any) -> pd.DataFrame | None:
    """Convert step result payload into a DataFrame when possible."""
    if isinstance(result, pd.DataFrame):
        return result.copy(deep=True)

    if isinstance(result, pd.Series):
        return result.to_frame().T.reset_index(drop=True)

    return None


def _find_chart_source_dataframe(
    step_results: list[dict[str, Any]],
    chart_step_number: int,
) -> pd.DataFrame | None:
    """Find the most recent tabular result before the chart step."""
    if chart_step_number <= 1:
        return None

    for index in range(chart_step_number - 2, -1, -1):
        item = step_results[index]
        if not isinstance(item, dict):
            continue

        if isinstance(item.get("error"), str):
            continue

        dataframe = _result_to_dataframe(item.get("result"))
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            return dataframe

    return None


def _prettify_name(name: str) -> str:
    """Convert a column/step name into a readable title ('total_profit' -> 'Total Profit')."""
    text = re.sub(r"[\-_]+", " ", str(name))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def _looks_like_money(name: str) -> bool:
    """Return True when a metric name suggests a currency value."""
    lowered = str(name).lower()
    tokens = [
        "revenue", "sales", "profit", "amount", "income", "price",
        "cost", "spend", "budget", "salary", "fee", "value",
    ]
    return any(token in lowered for token in tokens)


def _format_compact(value: float) -> str:
    """Format a number compactly: 108000 -> '108K', 1200000 -> '1.2M'."""
    if value is None or pd.isna(value):
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


def _format_number(value: float) -> str:
    """Format numeric values for concise UI display."""
    if pd.isna(value):
        return "n/a"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{float(value):,.2f}"


def _html_escape(text: str) -> str:
    """Escape text for safe embedding in rendered HTML."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _analyze_chart_data(chart: dict[str, Any], source_df: pd.DataFrame | None) -> dict[str, Any]:
    """Compute KPI cards and numeric facts for a chart from its source data."""
    empty = {"kpis": [], "mode": "none", "metric_label": "", "group_label": ""}

    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return empty

    numeric_cols = list(source_df.select_dtypes(include="number").columns)
    categorical_cols = [c for c in source_df.columns if c not in numeric_cols]
    chart_type = str(chart.get("chart_type", "")).lower()
    record_kpi = {
        "label": "Records",
        "icon": "📊",
        "value": f"{len(source_df):,}",
        "sub": "Total records",
        "tone": "neutral",
    }

    if chart_type == "scatter" and len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        x_series = pd.to_numeric(source_df[x_col], errors="coerce")
        y_series = pd.to_numeric(source_df[y_col], errors="coerce")
        corr = x_series.corr(y_series)
        return {
            "kpis": [
                {
                    "label": "Correlation",
                    "icon": "🔗",
                    "value": f"{corr:.2f}",
                    "sub": f"{_prettify_name(x_col)} vs {_prettify_name(y_col)}",
                    "tone": "info",
                },
                {
                    "label": "Highest Value",
                    "icon": "🏆",
                    "value": _format_compact(y_series.max()),
                    "sub": _prettify_name(y_col),
                    "tone": "success",
                },
                {
                    "label": "Average Value",
                    "icon": "📈",
                    "value": _format_compact(y_series.mean()),
                    "sub": "Mean",
                    "tone": "neutral",
                },
                record_kpi,
            ],
            "mode": "scatter",
            "metric_label": _prettify_name(y_col),
            "group_label": _prettify_name(x_col),
            "best_name": _prettify_name(x_col),
            "best_value": _format_compact(y_series.max()),
            "worst_name": _prettify_name(y_col),
            "worst_value": _format_compact(y_series.min()),
            "avg_value": _format_compact(y_series.mean()),
            "trend": "rising" if y_series.iloc[-1] > y_series.iloc[0] else "declining",
        }

    if chart_type in {"line", "area"} and numeric_cols:
        metric = numeric_cols[0]
        series = pd.to_numeric(source_df[metric], errors="coerce").dropna()
        if not series.empty:
            best_value = float(series.max())
            worst_value = float(series.min())
            avg_value = float(series.mean())
            metric_label = _prettify_name(metric)
            money_prefix = "$" if _looks_like_money(metric_label) else ""
            return {
                "kpis": [
                    {
                        "label": "Highest Value",
                        "icon": "🏆",
                        "value": f"{money_prefix}{_format_compact(best_value)}",
                        "sub": "Peak",
                        "tone": "success",
                    },
                    {
                        "label": "Lowest Value",
                        "icon": "⚠️",
                        "value": f"{money_prefix}{_format_compact(worst_value)}",
                        "sub": "Trough",
                        "tone": "danger",
                    },
                    {
                        "label": "Average",
                        "icon": "📈",
                        "value": f"{money_prefix}{_format_compact(avg_value)}",
                        "sub": "Mean",
                        "tone": "neutral",
                    },
                    record_kpi,
                ],
                "mode": "series",
                "metric_label": metric_label,
                "group_label": _prettify_name(categorical_cols[0]) if categorical_cols else "time",
                "best_name": "Peak",
                "best_value": best_value,
                "worst_name": "Trough",
                "worst_value": worst_value,
                "avg_value": avg_value,
                "trend": "rising" if series.iloc[-1] > series.iloc[0] else "declining",
            }

    if categorical_cols and numeric_cols:
        group_col = categorical_cols[0]
        metric = numeric_cols[0]
        grouped = (
            pd.to_numeric(source_df[metric], errors="coerce")
            .groupby(source_df[group_col].astype(str), sort=False)
            .sum()
            .dropna()
        )
        if not grouped.empty:
            best = grouped.idxmax()
            worst = grouped.idxmin()
            best_value = float(grouped.max())
            worst_value = float(grouped.min())
            total_value = float(grouped.sum())
            avg_value = float(grouped.mean())
            best_share = best_value / total_value * 100 if total_value else 0.0
            metric_label = _prettify_name(metric)
            money_prefix = "$" if _looks_like_money(metric_label) else ""

            return {
                "kpis": [
                    {
                        "label": "Highest Value",
                        "icon": "🏆",
                        "value": f"{money_prefix}{_format_compact(best_value)}",
                        "sub": best,
                        "tone": "success",
                    },
                    {
                        "label": "Lowest Value",
                        "icon": "⚠️",
                        "value": f"{money_prefix}{_format_compact(worst_value)}",
                        "sub": worst,
                        "tone": "danger",
                    },
                    {
                        "label": "Average",
                        "icon": "📈",
                        "value": f"{money_prefix}{_format_compact(avg_value)}",
                        "sub": f"Across {len(grouped)} segments",
                        "tone": "neutral",
                    },
                    record_kpi,
                ],
                "mode": "categorical",
                "metric_label": metric_label,
                "group_label": _prettify_name(group_col),
                "best_name": best,
                "best_value": best_value,
                "best_share": best_share,
                "worst_name": worst,
                "worst_value": worst_value,
                "avg_value": avg_value,
                "total_value": total_value,
                "n_segments": len(grouped),
            }

    if numeric_cols:
        metric = numeric_cols[0]
        series = pd.to_numeric(source_df[metric], errors="coerce").dropna()
        if not series.empty:
            best_value = float(series.max())
            worst_value = float(series.min())
            avg_value = float(series.mean())
            metric_label = _prettify_name(metric)
            money_prefix = "$" if _looks_like_money(metric_label) else ""
            return {
                "kpis": [
                    {
                        "label": "Highest Value",
                        "icon": "🏆",
                        "value": f"{money_prefix}{_format_compact(best_value)}",
                        "sub": "Peak",
                        "tone": "success",
                    },
                    {
                        "label": "Lowest Value",
                        "icon": "⚠️",
                        "value": f"{money_prefix}{_format_compact(worst_value)}",
                        "sub": "Trough",
                        "tone": "danger",
                    },
                    {
                        "label": "Average",
                        "icon": "📈",
                        "value": f"{money_prefix}{_format_compact(avg_value)}",
                        "sub": "Mean",
                        "tone": "neutral",
                    },
                    record_kpi,
                ],
                "mode": "series",
                "metric_label": metric_label,
                "group_label": _prettify_name(categorical_cols[0]) if categorical_cols else "time",
                "best_name": "Peak",
                "best_value": best_value,
                "worst_name": "Trough",
                "worst_value": worst_value,
                "avg_value": avg_value,
                "trend": "rising" if series.iloc[-1] > series.iloc[0] else "declining",
            }

    if categorical_cols:
        counts = source_df[categorical_cols[0]].astype(str).value_counts()
        if not counts.empty:
            best = counts.idxmax()
            worst = counts.idxmin()
            best_value = int(counts.max())
            worst_value = int(counts.min())
            return {
                "kpis": [
                    {
                        "label": "Most Frequent",
                        "icon": "🏆",
                        "value": f"{best_value:,}",
                        "sub": best,
                        "tone": "success",
                    },
                    {
                        "label": "Least Frequent",
                        "icon": "⚠️",
                        "value": f"{worst_value:,}",
                        "sub": worst,
                        "tone": "danger",
                    },
                    {
                        "label": "Average Frequency",
                        "icon": "📈",
                        "value": f"{counts.mean():,.1f}",
                        "sub": "Per category",
                        "tone": "neutral",
                    },
                    record_kpi,
                ],
                "mode": "counts",
                "metric_label": _prettify_name(categorical_cols[0]),
                "group_label": _prettify_name(categorical_cols[0]),
                "best_name": best,
                "best_value": best_value,
                "worst_name": worst,
                "worst_value": worst_value,
                "avg_value": float(counts.mean()),
            }

    return empty


def _chart_title_and_subtitle(chart: dict[str, Any], analysis: dict[str, Any]) -> tuple[str, str]:
    """Derive a user-facing title and subtitle for a chart."""
    chart_type = str(chart.get("chart_type", "")).lower()
    mode = analysis.get("mode", "none")
    metric = analysis.get("metric_label", "")
    group = analysis.get("group_label", "")

    if mode == "categorical" and metric and group:
        title = f"{metric} by {group}"
        subtitle = f"Comparison of {metric.lower()} across all {group.lower()} segments."
    elif mode == "scatter" and metric and group:
        title = f"{group} vs {metric}"
        subtitle = f"Relationship between {group.lower()} and {metric.lower()}."
    elif mode == "series" and metric:
        title = metric
        subtitle = f"Trend of {metric.lower()} over the analyzed period."
    elif mode == "counts" and metric:
        title = f"Distribution of {metric}"
        subtitle = f"Frequency of each {metric.lower()} category in the dataset."
    elif metric:
        title = metric
        subtitle = "Visual summary of the analyzed metric."
    else:
        title = "Data Overview"
        subtitle = "Visual overview of the analyzed dataset."

    if chart_type in {"pie", "donut"} and mode == "categorical" and metric and group:
        subtitle = f"Share of {metric.lower()} contributed by each {group.lower()}."
    if chart_type in {"hist", "kde", "box", "boxplot", "violin"} and metric:
        subtitle = f"Distribution of {metric.lower()} across the analyzed data."

    return title, subtitle


def _build_business_insight(chart: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Compose a senior-business-analyst style narrative from computed facts."""
    mode = analysis.get("mode", "none")
    metric = analysis.get("metric_label", "")
    money_prefix = "$" if _looks_like_money(metric) else ""

    if mode == "categorical":
        best = analysis["best_name"]
        worst = analysis["worst_name"]
        share = analysis.get("best_share", 0.0)
        n = analysis.get("n_segments", 0)
        best_v = f"{money_prefix}{_format_compact(analysis['best_value'])}"
        worst_v = f"{money_prefix}{_format_compact(analysis['worst_value'])}"
        avg_v = f"{money_prefix}{_format_compact(analysis['avg_value'])}"
        paragraphs = [
            f"{best} is the strongest performer, reaching {best_v} — approximately {share:.0f}% of total "
            f"{metric.lower()}. This makes it the clear growth engine of the portfolio.",
            f"{worst} trails significantly at {worst_v}, pointing to an opportunity to investigate the drivers "
            f"behind its underperformance and unlock untapped potential.",
            f"Across the {n} segments, the average is {avg_v}, providing a solid baseline for setting realistic "
            f"targets and tracking performance over time.",
        ]
        return "\n\n".join(paragraphs)

    if mode == "series":
        best_v = f"{money_prefix}{_format_compact(analysis['best_value'])}"
        worst_v = f"{money_prefix}{_format_compact(analysis['worst_value'])}"
        avg_v = f"{money_prefix}{_format_compact(analysis['avg_value'])}"
        trend = analysis.get("trend", "stable")
        paragraphs = [
            f"The metric peaked at {best_v} and dipped to {worst_v} over the analyzed period.",
            f"The overall trend shows {'an upward trajectory' if trend == 'rising' else 'a downward trajectory'}, "
            f"with an average of {avg_v} across all observations.",
        ]
        if trend == "rising":
            paragraphs.append(
                "Sustaining this momentum should remain a top priority for the coming periods."
            )
        else:
            paragraphs.append(
                "Reverse engineering the periods of decline will be essential to stabilize performance."
            )
        return "\n\n".join(paragraphs)

    if mode == "counts":
        best = analysis["best_name"]
        worst = analysis["worst_name"]
        best_v = f"{analysis['best_value']:,}"
        worst_v = f"{analysis['worst_value']:,}"
        avg_v = f"{analysis['avg_value']:,.1f}"
        paragraphs = [
            f"'{best}' dominates the dataset with {best_v} occurrences, making it the clear focus of the data.",
            f"'{worst}' appears only {worst_v} times, suggesting a segment that may be underserved or underexplored.",
            f"On average, each category appears {avg_v} times, which helps calibrate expectations for future sampling.",
        ]
        return "\n\n".join(paragraphs)

    return (
        "The visualization provides a clear overview of the dataset, highlighting the distribution and "
        "key patterns across the analyzed fields. No single metric dominates, so decisions should weigh "
        "multiple dimensions before committing resources."
    )


def _build_recommendations(chart: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    """Generate concise, actionable recommendations from the analysis."""
    mode = analysis.get("mode", "none")

    if mode == "categorical":
        best = analysis["best_name"]
        worst = analysis["worst_name"]
        return [
            f"Increase investment in {best}, the highest-performing segment.",
            f"Investigate why {worst} underperforms and build a corrective plan.",
            f"Replicate the success factors of {best} across other segments.",
            "Revisit pricing and operations to close the gap between top and bottom performers.",
        ]

    if mode == "series":
        if analysis.get("trend") == "rising":
            return [
                "Capitalize on the upward trend by scaling the initiatives behind this metric.",
                "Set stretch targets to lock in the current growth momentum.",
                "Monitor the metric for early signs of slowdown in the coming periods.",
            ]
        return [
            "Develop a mitigation plan to reverse the downward trend.",
            "Identify the periods driving the decline and investigate their root causes.",
            "Introduce a closer monitoring cadence until the metric stabilizes.",
        ]

    if mode == "counts":
        best = analysis["best_name"]
        worst = analysis["worst_name"]
        return [
            f"Deep-dive into what drives the popularity of '{best}'.",
            f"Explore ways to lift engagement with '{worst}'.",
            "Use segment-level feedback to guide the next iteration of the strategy.",
        ]

    return [
        "Focus on the metrics with the highest business impact first.",
        "Establish a monitoring cadence to track changes over time.",
    ]

def _render_kpi_cards(kpis: list[dict[str, str]]) -> None:
    """Render responsive KPI cards with icons and tone-based colors."""
    if not kpis:
        return

    cards: list[str] = []
    for kpi in kpis:
        tone = str(kpi.get("tone", "neutral"))
        icon = str(kpi.get("icon", "📊"))
        label = _html_escape(kpi.get("label", ""))
        value = _html_escape(kpi.get("value", ""))
        sub = _html_escape(kpi.get("sub", ""))
        sub_html = f'<div class="ada-kpi-sub">{sub}</div>' if sub else ""
        cards.append(
            f'<div class="ada-kpi-card tone-{tone}">'
            f'<div class="ada-kpi-icon">{icon}</div>'
            f'<div class="ada-kpi-body">'
            f'<div class="ada-kpi-label">{label}</div>'
            f'<div class="ada-kpi-value tone-{tone}">{value}</div>'
            f"{sub_html}"
            f"</div></div>"
        )

    st.markdown(f'<div class="ada-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_insight_card(text: str) -> None:
    """Render the Key Insight card with a business-analyst narrative."""
    if not text.strip():
        return

    paragraphs = "".join(
        f"<p>{_html_escape(paragraph)}</p>" for paragraph in text.split("\n\n") if paragraph.strip()
    )
    st.markdown(
        '<div class="ada-card ada-insight">'
        '<div class="ada-card-head">'
        '<span class="ada-card-icon">💡</span>'
        '<span class="ada-card-title">Key Insight</span>'
        "</div>"
        f'<div class="ada-card-body">{paragraphs}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_recommendations_card(items: list[str]) -> None:
    """Render the Recommendations card with concise action items."""
    if not items:
        return

    bullets = "".join(f"<li>{_html_escape(item)}</li>" for item in items)
    st.markdown(
        '<div class="ada-card ada-reco">'
        '<div class="ada-card-head">'
        '<span class="ada-card-icon">🎯</span>'
        '<span class="ada-card-title">Recommendations</span>'
        "</div>"
        f'<div class="ada-card-body"><ul>{bullets}</ul></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _plotly_base_layout(
    x_title: str = "",
    y_title: str = "",
    height: int = _PLOTLY_CHART_HEIGHT,
) -> dict:
    """Shared dark-theme layout for all Plotly charts."""
    return dict(
        height=height,
        margin=dict(l=64, r=20, t=20, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=_PLOTLY_CARD_BG,
        font=dict(family=_PLOTLY_FONT, color=_PLOTLY_TEXT, size=12),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=_PLOTLY_MUTED, size=12)),
            tickfont=dict(color=_PLOTLY_MUTED, size=11),
            gridcolor=_PLOTLY_GRID,
            zeroline=False,
            linecolor=_PLOTLY_GRID,
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=_PLOTLY_MUTED, size=12)),
            tickfont=dict(color=_PLOTLY_MUTED, size=11),
            gridcolor=_PLOTLY_GRID,
            zeroline=False,
            linecolor=_PLOTLY_GRID,
        ),
        hoverlabel=dict(
            bgcolor="#1F2937",
            bordercolor="#374151",
            font=dict(color="#FFFFFF", size=12),
        ),
        showlegend=False,
    )


def _plotly_format_value(value: float, money: bool = False) -> str:
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


def _highlight_colors(values: list[float]) -> list[str]:
    """Green for the best value, red for the worst, blue palette for the rest."""
    if not values:
        return []
    best_index = max(range(len(values)), key=lambda i: values[i])
    worst_index = min(range(len(values)), key=lambda i: values[i])
    colors: list[str] = []
    for index in range(len(values)):
        if index == best_index:
            colors.append(_PLOTLY_GREEN)
        elif index == worst_index:
            colors.append(_PLOTLY_RED)
        else:
            colors.append(_PLOTLY_BLUE_PALETTE[index % len(_PLOTLY_BLUE_PALETTE)])
    return colors


def _kde_density(series: pd.Series, n_points: int = 200) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate a gaussian KDE over a numeric series using numpy only."""
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty or values.nunique() < 2:
        return None
    std = float(values.std())
    bandwidth = 1.06 * std * (len(values) ** -0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max(float(values.max() - values.min()) / 4.0, 1e-6)
    grid = np.linspace(
        float(values.min()) - 3 * bandwidth,
        float(values.max()) + 3 * bandwidth,
        n_points,
    )
    differences = grid[:, None] - values.to_numpy()[None, :]
    density = np.mean(
        np.exp(-0.5 * (differences / bandwidth) ** 2), axis=1
    ) / (bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def _plotly_bar(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Modern bar chart with value labels and best/worst highlighting."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    metric_label = _prettify_name(metric)
    money = _looks_like_money(metric_label)

    if categorical_cols:
        category = categorical_cols[0]
        grouped = (
            pd.to_numeric(df[metric], errors="coerce")
            .groupby(df[category].astype(str), sort=False)
            .sum()
            .dropna()
        )
        if grouped.empty:
            return None
        labels = [str(label) for label in grouped.index]
        values = [float(value) for value in grouped.values]
        x_title = _prettify_name(category)
    else:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            return None
        labels = [str(index) for index in range(len(series))]
        values = [float(value) for value in series.values]
        x_title = ""

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=_highlight_colors(values),
            text=[_plotly_format_value(value, money) for value in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(**_plotly_base_layout(x_title=x_title, y_title=metric_label))
    if len(labels) > 12:
        figure.update_xaxes(tickangle=45, tickfont=dict(size=10))
    return figure


def _plotly_line(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Trend line with highlighted peak and trough markers."""
    if not numeric_cols:
        return None
    metric_label = _prettify_name(numeric_cols[0])
    money = _looks_like_money(metric_label)

    if len(numeric_cols) >= 2 and not categorical_cols:
        figure = go.Figure()
        for index, metric in enumerate(numeric_cols[:3]):
            series = pd.to_numeric(df[metric], errors="coerce").dropna()
            figure.add_trace(
                go.Scatter(
                    x=[str(i) for i in range(len(series))],
                    y=[float(value) for value in series.values],
                    mode="lines+markers",
                    name=_prettify_name(metric),
                    line=dict(
                        color=_PLOTLY_BLUE_PALETTE[index % len(_PLOTLY_BLUE_PALETTE)],
                        width=3,
                    ),
                    marker=dict(size=5),
                    hovertemplate=f"{_prettify_name(metric)}: %{{y:,.2f}}<extra></extra>",
                )
            )
        figure.update_layout(**_plotly_base_layout(x_title="Index", y_title=metric_label))
        figure.update_layout(showlegend=True, legend=dict(font=dict(color=_PLOTLY_MUTED)))
        return figure

    metric = numeric_cols[0]
    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    if series.empty:
        return None

    if categorical_cols:
        x_values = [
            str(value) for value in df.loc[series.index, categorical_cols[0]].values
        ]
        x_title = _prettify_name(categorical_cols[0])
    else:
        x_values = [str(index) for index in range(len(series))]
        x_title = "Index"

    y_values = [float(value) for value in series.values]
    show_text = len(y_values) <= 24
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line=dict(color=_PLOTLY_BLUE, width=3),
            marker=dict(size=6, color=_PLOTLY_BLUE),
            text=[_plotly_format_value(value, money) for value in y_values] if show_text else None,
            textposition="top center",
            textfont=dict(color=_PLOTLY_MUTED, size=10),
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )

    best_index = int(np.argmax(y_values))
    worst_index = int(np.argmin(y_values))
    for marker_index, color in ((best_index, _PLOTLY_GREEN), (worst_index, _PLOTLY_RED)):
        figure.add_trace(
            go.Scatter(
                x=[x_values[marker_index]],
                y=[y_values[marker_index]],
                mode="markers",
                marker=dict(
                    color=color,
                    size=14,
                    symbol="diamond",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
            )
        )

    figure.update_layout(**_plotly_base_layout(x_title=x_title, y_title=metric_label))
    return figure


def _plotly_area(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Filled area chart reusing the line logic with an area fill."""
    figure = _plotly_line(df, numeric_cols, categorical_cols)
    if figure is None:
        return None
    for trace in figure.data:
        trace.fill = "tozeroy"
        trace.fillcolor = "rgba(59, 130, 246, 0.15)"
    figure.update_layout(**_plotly_base_layout(height=_PLOTLY_CHART_HEIGHT))
    return figure


def _plotly_scatter(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> go.Figure | None:
    """Scatter plot highlighting the top value in green."""
    if len(numeric_cols) < 2:
        return None
    x_col, y_col = numeric_cols[0], numeric_cols[1]
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    mask = x_values.notna() & y_values.notna()
    x_values, y_values = x_values[mask], y_values[mask]
    if x_values.empty:
        return None

    colors = [_PLOTLY_BLUE_LIGHT] * len(x_values)
    best_index = int(np.argmax(y_values.values))
    colors[best_index] = _PLOTLY_GREEN

    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                size=9,
                color=colors,
                opacity=0.85,
                line=dict(color=_PLOTLY_CARD_BG, width=0.5),
            ),
            hovertemplate="%{x:,.2f}<br>%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        **_plotly_base_layout(x_title=_prettify_name(x_col), y_title=_prettify_name(y_col))
    )
    return figure


def _plotly_bubble(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> go.Figure | None:
    """Bubble chart with a size dimension when three numeric columns exist."""
    if len(numeric_cols) < 3:
        return _plotly_scatter(df, numeric_cols)
    x_col, y_col, size_col = numeric_cols[:3]
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    size_values = pd.to_numeric(df[size_col], errors="coerce")
    mask = x_values.notna() & y_values.notna() & size_values.notna()
    x_values, y_values, size_values = x_values[mask], y_values[mask], size_values[mask]
    if x_values.empty:
        return None

    size_min, size_max = float(size_values.min()), float(size_values.max())
    size_range = size_max - size_min if size_max > size_min else 1.0
    marker_sizes = 12 + 48 * (size_values - size_min) / size_range

    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=_PLOTLY_BLUE_LIGHT,
                opacity=0.75,
                line=dict(color=_PLOTLY_CARD_BG, width=0.5),
            ),
            text=[_plotly_format_value(value) for value in size_values.values],
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=9),
            hovertemplate=f"{_prettify_name(size_col)}: %{{text}}<extra></extra>",
        )
    )
    figure.update_layout(
        **_plotly_base_layout(x_title=_prettify_name(x_col), y_title=_prettify_name(y_col))
    )
    return figure


def _plotly_pie(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Donut chart with best/worst slice highlighting."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    if categorical_cols:
        grouped = (
            pd.to_numeric(df[metric], errors="coerce")
            .groupby(df[categorical_cols[0]].astype(str), sort=False)
            .sum()
            .dropna()
        )
    else:
        grouped = pd.to_numeric(df[metric], errors="coerce").dropna()
    if grouped.empty:
        return None

    labels = [str(label) for label in grouped.index]
    values = [float(value) for value in grouped.values]

    figure = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(
                colors=_highlight_colors(values),
                line=dict(color=_PLOTLY_CARD_BG, width=2),
            ),
            textinfo="label+percent",
            textfont=dict(color="#FFFFFF", size=12),
            hovertemplate="%{label}<br>%{value:,.2f} (%{percent})<extra></extra>",
        )
    )
    figure.update_layout(**_plotly_base_layout())
    return figure


def _plotly_hist(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> go.Figure | None:
    """Clean histogram with 20 bins."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    if series.empty:
        return None
    figure = go.Figure(
        go.Histogram(
            x=series,
            nbinsx=20,
            marker_color=_PLOTLY_BLUE,
            opacity=0.85,
            marker_line=dict(color=_PLOTLY_CARD_BG, width=1),
            hovertemplate="%{x:,.2f}<br>Count: %{y}<extra></extra>",
        )
    )
    figure.update_layout(**_plotly_base_layout(x_title=_prettify_name(metric), y_title="Count"))
    return figure


def _plotly_kde(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> go.Figure | None:
    """Smoothed density curve rendered from a numpy KDE estimate."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    estimate = _kde_density(df[metric])
    if estimate is None:
        return None
    grid, density = estimate
    figure = go.Figure(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            line=dict(color=_PLOTLY_BLUE, width=3),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.15)",
            hovertemplate="%{x:,.2f}<br>Density: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(**_plotly_base_layout(x_title=_prettify_name(metric), y_title="Density"))
    return figure


def _plotly_box(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Box plot with mean markers, grouped by category when present."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    metric_label = _prettify_name(metric)
    figure = go.Figure()
    if categorical_cols:
        category = categorical_cols[0]
        for label, subset in df.groupby(df[category].astype(str), sort=False):
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            figure.add_trace(
                go.Box(
                    y=values,
                    name=str(label),
                    boxmean=True,
                    marker_color=_PLOTLY_BLUE,
                    line=dict(color=_PLOTLY_BLUE_LIGHT, width=2),
                    fillcolor="rgba(59, 130, 246, 0.18)",
                )
            )
        x_title = _prettify_name(category)
    else:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        figure.add_trace(
            go.Box(
                y=values,
                boxmean=True,
                marker_color=_PLOTLY_BLUE,
                line=dict(color=_PLOTLY_BLUE_LIGHT, width=2),
                fillcolor="rgba(59, 130, 246, 0.18)",
            )
        )
        x_title = ""
    figure.update_layout(**_plotly_base_layout(x_title=x_title, y_title=metric_label))
    return figure


def _plotly_violin(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> go.Figure | None:
    """Violin plot with an inner box, grouped by category when present."""
    if not numeric_cols:
        return None
    metric = numeric_cols[0]
    metric_label = _prettify_name(metric)
    figure = go.Figure()
    if categorical_cols:
        category = categorical_cols[0]
        for label, subset in df.groupby(df[category].astype(str), sort=False):
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            figure.add_trace(
                go.Violin(
                    y=values,
                    name=str(label),
                    box_visible=True,
                    meanline_visible=True,
                    line=dict(color=_PLOTLY_BLUE_LIGHT, width=2),
                    fillcolor="rgba(59, 130, 246, 0.25)",
                    opacity=0.9,
                )
            )
        x_title = _prettify_name(category)
    else:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        figure.add_trace(
            go.Violin(
                y=values,
                box_visible=True,
                meanline_visible=True,
                line=dict(color=_PLOTLY_BLUE_LIGHT, width=2),
                fillcolor="rgba(59, 130, 246, 0.25)",
                opacity=0.9,
            )
        )
        x_title = ""
    figure.update_layout(**_plotly_base_layout(x_title=x_title, y_title=metric_label))
    return figure


def _plotly_heatmap(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> go.Figure | None:
    """Correlation heatmap with annotated values."""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr()
    figure = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=[str(col) for col in corr.columns],
            y=[str(col) for col in corr.index],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#EF4444"],
                [0.5, "#1E293B"],
                [1.0, "#3B82F6"],
            ],
            colorbar=dict(
                title=dict(text="Corr", font=dict(color=_PLOTLY_MUTED, size=11)),
                tickfont=dict(color=_PLOTLY_MUTED, size=10),
            ),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(**_plotly_base_layout(height=400))
    figure.update_xaxes(tickangle=45)
    return figure


def _build_plotly_figure(chart: dict[str, Any], source_df: pd.DataFrame | None) -> go.Figure | None:
    """Build an interactive Plotly figure mirroring the backend chart intent."""
    if not _PLOTLY_AVAILABLE:
        return None
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return None

    chart_type = str(chart.get("chart_type", "")).lower()
    numeric_cols = list(source_df.select_dtypes(include="number").columns)
    categorical_cols = [col for col in source_df.columns if col not in numeric_cols]

    if chart_type == "bar":
        return _plotly_bar(source_df, numeric_cols, categorical_cols)
    if chart_type == "line":
        return _plotly_line(source_df, numeric_cols, categorical_cols)
    if chart_type == "area":
        return _plotly_area(source_df, numeric_cols, categorical_cols)
    if chart_type == "scatter":
        return _plotly_scatter(source_df, numeric_cols)
    if chart_type == "bubble":
        return _plotly_bubble(source_df, numeric_cols)
    if chart_type == "pie":
        return _plotly_pie(source_df, numeric_cols, categorical_cols)
    if chart_type == "hist":
        return _plotly_hist(source_df, numeric_cols)
    if chart_type == "kde":
        return _plotly_kde(source_df, numeric_cols)
    if chart_type in {"box", "boxplot"}:
        return _plotly_box(source_df, numeric_cols, categorical_cols)
    if chart_type == "violin":
        return _plotly_violin(source_df, numeric_cols, categorical_cols)
    if chart_type == "heatmap":
        return _plotly_heatmap(source_df, numeric_cols)

    return None


def _render_plotly_chart(chart: dict[str, Any], source_df: pd.DataFrame | None) -> bool:
    """Render an interactive Plotly chart; return False when falling back to the image."""
    try:
        figure = _build_plotly_figure(chart, source_df)
        if figure is None:
            logger.warning(
                "Plotly render skipped for chart '%s': unsupported type or missing source data",
                chart.get("chart_type"),
            )
            return False
        st.plotly_chart(
            figure,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
        )
        return True
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Plotly render failed for chart '%s': %s",
            chart.get("chart_type"),
            exc,
        )
        return False


def _render_chart_section(chart: dict[str, Any], step_results: list[dict[str, Any]]) -> None:
    """Render one complete chart section: title, KPIs, chart, insights, recommendations."""
    chart_step_number = int(chart.get("step_number", 0))
    source_df = _find_chart_source_dataframe(step_results, chart_step_number)
    analysis = _analyze_chart_data(chart, source_df)
    title, subtitle = _chart_title_and_subtitle(chart, analysis)

    st.markdown(
        f'<div class="ada-chart-title">{_html_escape(title)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="ada-chart-subtitle">{_html_escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )

    _render_kpi_cards(analysis.get("kpis", []))

    st.markdown('<div class="ada-chart-frame">', unsafe_allow_html=True)
    if not _render_plotly_chart(chart, source_df):
        st.image(str(chart.get("path", "")), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    insight_text = _build_business_insight(chart, analysis)
    _render_insight_card(insight_text)
    _render_recommendations_card(_build_recommendations(chart, analysis))

    st.markdown('<hr class="ada-divider"/>', unsafe_allow_html=True)


_CSV_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1")


def _read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV, retrying with common encodings when UTF-8 fails."""
    last_error: Exception | None = None

    for encoding in _CSV_ENCODING_FALLBACKS:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return pd.read_csv(uploaded_file)


_PREMIUM_CSS = """
<style>
/* ============================================================
   AI DATA ANALYST — PREMIUM DARK THEME
   ============================================================ */

.stApp {
    background: #0B1220;
    background: linear-gradient(180deg, #0B1220 0%, #0D1428 100%);
    color: #FFFFFF;
}

/* ---------- Typography ---------- */

.ada-chart-title {
    font-size: 26px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.25;
    margin: 0 0 4px 0;
}

.ada-chart-subtitle {
    font-size: 14px;
    font-weight: 400;
    color: #9CA3AF;
    line-height: 1.5;
    margin: 0 0 24px 0;
}

/* ---------- KPI cards ---------- */

.ada-kpi-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 24px;
}

.ada-kpi-card {
    flex: 1 1 220px;
    min-width: 200px;
    display: flex;
    align-items: center;
    gap: 14px;
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 18px 20px;
    animation: adaFadeUp 0.5s ease both;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

.ada-kpi-card:hover {
    transform: translateY(-3px);
    border-color: #374151;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.ada-kpi-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.tone-success .ada-kpi-icon { background: rgba(34, 197, 94, 0.12); }
.tone-danger  .ada-kpi-icon { background: rgba(239, 68, 68, 0.12); }
.tone-info    .ada-kpi-icon { background: rgba(59, 130, 246, 0.12); }
.tone-warning .ada-kpi-icon { background: rgba(245, 158, 11, 0.12); }
.tone-neutral .ada-kpi-icon { background: rgba(156, 163, 175, 0.12); }

.ada-kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.ada-kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
}

.ada-kpi-value.tone-success { color: #22C55E; }
.ada-kpi-value.tone-danger  { color: #EF4444; }
.ada-kpi-value.tone-info    { color: #3B82F6; }
.ada-kpi-value.tone-warning { color: #F59E0B; }
.ada-kpi-value.tone-neutral { color: #E5E7EB; }

.ada-kpi-sub {
    font-size: 13px;
    color: #9CA3AF;
    margin-top: 2px;
}

/* ---------- Chart frame ---------- */

.ada-chart-frame {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    animation: adaFadeUp 0.5s ease both;
}

.ada-chart-frame img {
    width: 100%;
    border-radius: 10px;
}

/* ---------- Insight / Recommendation cards ---------- */

.ada-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 20px;
    animation: adaFadeUp 0.5s ease both;
}

.ada-card-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.ada-card-icon {
    font-size: 18px;
}

.ada-card-title {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.01em;
}

.ada-card-body {
    font-size: 15px;
    color: #D1D5DB;
    line-height: 1.65;
}

.ada-card-body p {
    margin: 0 0 10px 0;
}

.ada-card-body p:last-child {
    margin-bottom: 0;
}

.ada-card-body ul {
    margin: 0;
    padding-left: 20px;
}

.ada-card-body li {
    margin-bottom: 8px;
}

.ada-card-body li:last-child {
    margin-bottom: 0;
}

.ada-insight .ada-card-title { color: #3B82F6; }
.ada-reco   .ada-card-title { color: #F59E0B; }

/* ---------- Divider ---------- */

hr.ada-divider {
    border: none;
    height: 1px;
    background: #1F2937;
    margin: 32px 0;
}

/* ---------- Streamlit chrome polish ---------- */

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 12px;
    padding: 14px 16px;
}

.stButton > button[kind="primary"] {
    background: #3B82F6;
    border: none;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background: #2563EB;
}

.stTextArea textarea, .stTextInput input {
    background: #111827;
    border: 1px solid #1F2937;
    color: #FFFFFF;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 500;
}

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-thumb {
    background: #1F2937;
    border-radius: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

/* ---------- Animation ---------- */

@keyframes adaFadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


def _inject_premium_theme() -> None:
    """Inject the premium dark theme stylesheet into the app."""
    st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)


def _json_safe(value: Any) -> Any:
    """Convert runtime objects into Streamlit JSON-safe objects."""
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(col) for col in value.columns],
            "preview": value.head(10).to_dict(orient="records"),
        }

    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "name": str(value.name),
            "preview": value.head(10).to_dict(),
        }

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


@st.cache_resource(show_spinner=False)
def _get_compiled_graph():
    """Build and cache the compiled LangGraph application."""
    _ = get_settings()
    return build_workflow()


def main() -> None:
    """Render Streamlit controls and run the analysis workflow."""
    st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
    _inject_premium_theme()
    st.title("AI Data Analysis Agent")

    explain_mode = st.toggle("Explain mode", value=False)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    uploaded_df: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            uploaded_df = _read_uploaded_csv(uploaded_file)
            st.success(
                f"Loaded {uploaded_file.name}: {len(uploaded_df)} rows x {len(uploaded_df.columns)} columns"
            )
            with st.expander("CSV Preview", expanded=False):
                st.dataframe(uploaded_df.head(25), use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to read uploaded CSV: {exc}")

    query = st.text_area(
        "Query",
        placeholder="Example: Compare this month revenue with last month and show a chart.",
        height=120,
    )

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    if "final_state" not in st.session_state:
        st.session_state["final_state"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if run_clicked:
        if not query.strip():
            st.warning("Please enter a query before running analysis.")
        else:
            app = _get_compiled_graph()
            initial_state = _build_initial_state(query.strip(), uploaded_df)

            with st.spinner("Running workflow..."):
                try:
                    final_state: AgentState = app.invoke(
                        initial_state,
                        config={"recursion_limit": 25},
                    )
                except Exception as exc:
                    st.error(f"Workflow execution failed: {exc}")
                    return

            st.session_state["final_state"] = final_state
            st.session_state["last_query"] = query.strip()

    final_state = st.session_state.get("final_state")
    if not isinstance(final_state, dict):
        st.info("Run a query to see results.")
        return

    result = _extract_result(final_state)
    insights = _extract_insights(final_state)
    step_results = _extract_step_results(final_state)
    chart_entries = _collect_chart_paths(step_results)

    raw_retry_count = final_state.get("retry_count", 0)
    retry_count = raw_retry_count if isinstance(raw_retry_count, int) else 0
    data_source = str(final_state.get("data_source", "unknown")).upper()

    st.subheader("Run Summary")
    metric_col_source, metric_col_steps, metric_col_errors, metric_col_retries = st.columns(4)
    metric_col_source.metric("Data Source", data_source)
    metric_col_steps.metric("Steps", len(step_results))
    metric_col_errors.metric("Errors", _count_step_errors(step_results))
    metric_col_retries.metric("Retries", retry_count)

    tab_result, tab_steps, tab_insights, tab_charts = st.tabs(
        ["Final Output", "Execution Steps", "Insights", "Charts"]
    )

    with tab_result:
        st.markdown("### Result")
        if result:
            st.code(result, language="text")
        else:
            st.info("No result produced.")

    with tab_steps:
        if not step_results:
            st.info("No execution steps were captured.")
        else:
            for index, item in enumerate(step_results, start=1):
                step_name = str(item.get("step", "Unnamed step")).strip() or "Unnamed step"
                tool_name = str(item.get("tool", "unknown")).strip() or "unknown"
                error_text = item.get("error")
                has_error = isinstance(error_text, str) and bool(error_text.strip())
                status = "Failed" if has_error else "Completed"

                with st.expander(
                    f"Step {index}: {step_name} | Tool: {tool_name} | Status: {status}",
                    expanded=index == 1,
                ):
                    action = str(item.get("action", "")).strip()
                    if action:
                        st.caption(f"Action: {action}")

                    if has_error:
                        st.error(error_text)
                    else:
                        _render_step_payload(item.get("result"))

    with tab_insights:
        st.markdown("### Insights")
        if insights:
            st.markdown(insights)
        else:
            st.info("No insights produced.")

    with tab_charts:
        if chart_entries:
            for chart in chart_entries:
                _render_chart_section(chart, step_results)
        else:
            st.info("No charts generated for this run.")

    if explain_mode:
        st.subheader("Execution Trace")
        trace_payload = {
            "query": st.session_state.get("last_query", ""),
            "data_source": final_state.get("data_source"),
            "plan": final_state.get("plan", []),
            "retry": final_state.get("retry", False),
            "retry_count": retry_count,
            "intermediate_results": step_results,
        }
        st.json(_json_safe(trace_payload), expanded=False)


if __name__ == "__main__":
    main()
"""Deterministic chart-type recommendation and fallback dashboard suggestions.

The AI dashboard planner proposes chart configurations; this module validates the
choices against the data shape AND provides a fully deterministic fallback that
runs without any LLM. All rules are based on data type, cardinality and
analytical objective (time -> trend, few categories -> donut, many -> top-N bar,
two measures -> scatter, etc.).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.models import MAX_BAR_CATEGORIES, MAX_DONUT_CATEGORIES, MAX_FILTERS, MAX_KPIS
    from dashboard.profiler import DataProfile
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.models import MAX_BAR_CATEGORIES, MAX_DONUT_CATEGORIES, MAX_FILTERS, MAX_KPIS
    from project.dashboard.profiler import DataProfile


def recommend_chart_type(
    df: pd.DataFrame, dimension: str | None, measures: list[dict[str, Any]], split_by: str | None
) -> str:
    """Pick the most appropriate chart type for a dimension/measure combination."""
    if split_by and dimension is not None:
        return "bar"  # grouped/stacked comparison across a category

    if dimension is not None and dimension in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[dimension]):
            return "line"
        if pd.api.types.is_numeric_dtype(df[dimension]):
            return "scatter"
        cardinality = int(df[dimension].nunique(dropna=True))
        if cardinality <= MAX_DONUT_CATEGORIES:
            return "donut"
        if cardinality <= MAX_BAR_CATEGORIES:
            return "bar"
        return "hbar"

    # No categorical dimension: numeric-only comparisons.
    numeric_measures = [m for m in measures if m.get("column") in df.columns]
    if len(numeric_measures) >= 2:
        return "scatter"
    if len(numeric_measures) == 1:
        column = numeric_measures[0]["column"]
        if pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique(dropna=True) >= 6:
            return "hist"
        return "bar"
    return "table"


def suggest_kpi_configs(df: pd.DataFrame, profile: DataProfile) -> list[dict[str, Any]]:
    """Deterministic fallback KPI list derived from the profile."""
    kpis: list[dict[str, Any]] = []
    candidates = [c for c in profile.metric_candidates if c in df.columns]
    primary = profile.primary_metric if profile.primary_metric in df.columns else None

    if primary is None and candidates:
        primary = candidates[0]
    if primary is None:
        primary = next(iter(df.select_dtypes(include="number").columns), None)

    if primary is not None:
        kpis.append(
            {
                "id": f"kpi_{primary}",
                "label": f"Total {pretty(primary)}",
                "column": primary,
                "aggregation": "sum",
                "format": "number",
                "tone": "info",
                "icon": "📈",
                "delta": True,
            }
        )

    if profile.time_column and profile.time_column in df.columns:
        kpis.append(
            {
                "id": "kpi_records",
                "label": "Records",
                "column": profile.time_column,
                "aggregation": "count",
                "format": "int",
                "tone": "neutral",
                "icon": "📊",
                "delta": False,
            }
        )

    secondary = [c for c in candidates if c != primary][:3]
    for column in secondary:
        if len(kpis) >= MAX_KPIS:
            break
        kpis.append(
            {
                "id": f"kpi_{column}",
                "label": f"Total {pretty(column)}",
                "column": column,
                "aggregation": "sum",
                "format": "number",
                "tone": "neutral",
                "icon": "📊",
                "delta": False,
            }
        )
    return kpis[:MAX_KPIS]


def suggest_filter_configs(df: pd.DataFrame, profile: DataProfile) -> list[dict[str, Any]]:
    """Deterministic fallback filter list: date year + up to 3 categoricals."""
    filters: list[dict[str, Any]] = []

    if profile.time_column and profile.time_column in df.columns:
        filters.append(
            {
                "id": "f_year",
                "label": "Year",
                "column": profile.time_column,
                "type": "date_year",
            }
        )

    for column in profile.categorical_columns:
        if column not in df.columns:
            continue
        if len(filters) >= MAX_FILTERS:
            break
        unique_count = int(profile.categorical_unique_counts.get(column, df[column].nunique()))
        if unique_count <= 1 or unique_count > 200:
            continue
        filters.append(
            {
                "id": f"f_{column}",
                "label": pretty(column),
                "column": column,
                "type": "categorical_multi",
            }
        )
    return filters[:MAX_FILTERS]

def suggest_chart_configs(df: pd.DataFrame, profile: DataProfile) -> list[dict[str, Any]]:
    """Deterministic fallback chart list covering trend, segments, distribution."""
    charts: list[dict[str, Any]] = []
    candidates = [c for c in profile.metric_candidates if c in df.columns]
    primary = profile.primary_metric if profile.primary_metric in df.columns else None
    if primary is None and candidates:
        primary = candidates[0]
    if primary is None:
        primary = next(iter(df.select_dtypes(include="number").columns), None)
    if primary is None:
        return charts

    if profile.time_column and profile.time_column in df.columns:
        charts.append(
            {
                "id": "chart_trend",
                "chart_type": "line",
                "title": f"{pretty(primary)} over time",
                "dimension": profile.time_column,
                "measures": [{"column": primary, "aggregation": "sum"}],
                "max_points": 60,
                "width_span": 6,
            }
        )

    category = None
    for column in profile.categorical_columns:
        unique_count = int(profile.categorical_unique_counts.get(column, 0))
        if 2 <= unique_count <= 20:
            category = column
            break
    if category is not None:
        charts.append(
            {
                "id": "chart_segments",
                "chart_type": recommend_chart_type(
                    df, category, [{"column": primary, "aggregation": "sum"}], None
                ),
                "title": f"{pretty(primary)} by {pretty(category)}",
                "dimension": category,
                "measures": [{"column": primary, "aggregation": "sum"}],
                "max_points": 12,
                "width_span": 6,
            }
        )

    remaining = [c for c in candidates if c != primary][:1]
    if remaining and len(candidates) >= 2:
        charts.append(
            {
                "id": "chart_compare",
                "chart_type": "scatter",
                "title": f"{pretty(primary)} vs {pretty(remaining[0])}",
                "dimension": None,
                "measures": [
                    {"column": primary, "aggregation": "sum"},
                    {"column": remaining[0], "aggregation": "sum"},
                ],
                "max_points": 200,
                "width_span": 6,
            }
        )

    numeric = [c for c in profile.numeric_columns if c in df.columns][:6]
    if len(numeric) >= 3:
        charts.append(
            {
                "id": "chart_correlation",
                "chart_type": "heatmap",
                "title": "Metric correlation heatmap",
                "dimension": None,
                "measures": [{"column": column, "aggregation": "sum"} for column in numeric[:4]],
                "max_points": 50,
                "width_span": 6,
            }
        )
    return charts


def pretty(name: Any) -> str:
    """Readable title for a column name."""
    try:
        from dashboard.formatting import prettify_name
    except ModuleNotFoundError:  # pragma: no cover - package-style execution.
        from project.dashboard.formatting import prettify_name
    return prettify_name(name)


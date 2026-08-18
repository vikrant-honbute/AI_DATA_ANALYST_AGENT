"""Validation and safety for AI-generated dashboard configurations.

The AI dashboard planner may only reference columns and operations that exist in
the data profile. This module deterministically checks and repairs a raw
configuration before it is stored in agent state or rendered by the engine,
so an LLM mistake can never crash the dashboard or produce invalid charts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.models import (
        MAX_BAR_CATEGORIES,
        MAX_CHARTS,
        MAX_DONUT_CATEGORIES,
        MAX_FILTERS,
        MAX_KPIS,
        VALID_AGGREGATIONS,
        VALID_CHART_TYPES,
        VALID_FILTER_TYPES,
        VALID_FORMATS,
        VALID_TONES,
    )
    from dashboard.profiler import DataProfile
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.models import (
        MAX_BAR_CATEGORIES,
        MAX_CHARTS,
        MAX_DONUT_CATEGORIES,
        MAX_FILTERS,
        MAX_KPIS,
        VALID_AGGREGATIONS,
        VALID_CHART_TYPES,
        VALID_FILTER_TYPES,
        VALID_FORMATS,
        VALID_TONES,
    )
    from project.dashboard.profiler import DataProfile


class ConfigIssue:
    """A single validation finding with a deterministic repair."""

    __slots__ = ("field", "message", "repaired")

    def __init__(self, field: str, message: str, repaired: bool = False) -> None:
        self.field = field
        self.message = message
        self.repaired = repaired


def _existing_columns(df: pd.DataFrame) -> set[str]:
    """Return the set of columns the configuration may reference."""
    return {str(column) for column in df.columns}


def _agg_matches_column(aggregation: str, column: str, df: pd.DataFrame) -> bool:
    """Return True when an aggregation is type-compatible with a column."""
    if column not in df.columns:
        return False
    if aggregation in {"sum", "mean", "median", "min", "max"}:
        return pd.api.types.is_numeric_dtype(df[column])
    return True  # count / nunique / ratio / margin tolerate text columns


def _df_profile(df: pd.DataFrame) -> DataProfile:
    """Profile a DataFrame for validation checks."""
    try:
        from dashboard.profiler import profile_dataframe
    except ModuleNotFoundError:  # pragma: no cover - package-style execution.
        from project.dashboard.profiler import profile_dataframe
    try:
        return profile_dataframe(df)
    except Exception:  # pragma: no cover - profiler should not fail on valid frames.
        return DataProfile(rows=int(len(df)), columns=[str(c) for c in df.columns])


def validate_kpi(kpi: dict[str, Any], df: pd.DataFrame) -> list[ConfigIssue]:
    """Validate and repair a single KPI definition."""
    issues: list[ConfigIssue] = []
    columns = _existing_columns(df)

    aggregation = str(kpi.get("aggregation", "sum")).lower()
    if aggregation not in VALID_AGGREGATIONS:
        issues.append(
            ConfigIssue(
                "aggregation",
                f"Invalid aggregation '{aggregation}' for KPI '{kpi.get('label', '')}'.",
            )
        )
        kpi["aggregation"] = "sum"

    column = kpi.get("column")
    if column is not None and str(column) not in columns:
        issues.append(
            ConfigIssue("column", f"KPI references missing column '{column}'.", repaired=True)
        )
        kpi["column"] = None

    if kpi["column"] is not None and not _agg_matches_column(kpi["aggregation"], str(kpi["column"]), df):
        issues.append(
            ConfigIssue(
                "aggregation",
                f"Aggregation '{kpi['aggregation']}' is not valid for column "
                f"'{kpi['column']}'; falling back to 'count'.",
                repaired=True,
            )
        )
        kpi["aggregation"] = "count"

    denominator = kpi.get("denominator")
    if kpi["aggregation"] in {"ratio", "margin"}:
        if denominator is None or str(denominator) not in columns:
            issues.append(
                ConfigIssue(
                    "denominator",
                    f"KPI '{kpi.get('label', '')}' needs a valid denominator column.",
                    repaired=True,
                )
            )
            kpi["aggregation"] = "sum"

    fmt = str(kpi.get("format", "number")).lower()
    if fmt not in VALID_FORMATS:
        kpi["format"] = "number"

    tone = str(kpi.get("tone", "neutral")).lower()
    if tone not in VALID_TONES:
        kpi["tone"] = "neutral"

    return issues


def validate_filter(filt: dict[str, Any], df: pd.DataFrame) -> list[ConfigIssue]:
    """Validate and repair a single filter definition."""
    issues: list[ConfigIssue] = []
    columns = _existing_columns(df)

    column = str(filt.get("column", "")).strip()
    if not column or column not in columns:
        issues.append(
            ConfigIssue("column", f"Filter references missing column '{column}'.", repaired=True)
        )
        filt["column"] = ""

    filter_type = str(filt.get("type", "categorical_multi")).lower()
    if filter_type not in VALID_FILTER_TYPES:
        issues.append(
            ConfigIssue(
                "type", f"Invalid filter type '{filter_type}'; using multi-select.", repaired=True
            )
        )
        filt["type"] = "categorical_multi"

    if column in columns:
        if filter_type in {"date_year", "date_range"} and not pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            issues.append(
                ConfigIssue(
                    "type",
                    f"Filter '{column}' is not a datetime; using categorical multi-select.",
                    repaired=True,
                )
            )
            filt["type"] = "categorical_multi"
        if filter_type == "numeric_range" and not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(
                ConfigIssue(
                    "type",
                    f"Filter '{column}' is not numeric; using categorical multi-select.",
                    repaired=True,
                )
            )
            filt["type"] = "categorical_multi"

    return issues


def validate_chart(chart: dict[str, Any], df: pd.DataFrame) -> list[ConfigIssue]:
    """Validate and repair a single chart definition."""
    issues: list[ConfigIssue] = []
    columns = _existing_columns(df)

    chart_type = str(chart.get("chart_type", "bar")).lower()
    if chart_type not in VALID_CHART_TYPES:
        issues.append(
            ConfigIssue(
                "chart_type",
                f"Invalid chart type '{chart_type}'; falling back to 'bar'.",
                repaired=True,
            )
        )
        chart["chart_type"] = "bar"

    dimension = chart.get("dimension")
    if dimension is not None and str(dimension) not in columns:
        issues.append(
            ConfigIssue(
                "dimension",
                f"Chart references missing dimension column '{dimension}'.",
                repaired=True,
            )
        )
        chart["dimension"] = None

    measures = chart.get("measures") or []
    if not isinstance(measures, list) or not measures:
        issues.append(
            ConfigIssue("measures", f"Chart '{chart.get('id', '')}' has no measures.", repaired=True)
        )
        chart["measures"] = []

    valid_measures: list[dict[str, Any]] = []
    for measure in measures:
        if not isinstance(measure, dict):
            continue
        column = measure.get("column")
        aggregation = str(measure.get("aggregation", "sum")).lower()
        if column is None or str(column) not in columns:
            issues.append(
                ConfigIssue(
                    "measures",
                    f"Chart references missing measure column '{column}'.",
                    repaired=True,
                )
            )
            continue
        if aggregation not in VALID_AGGREGATIONS:
            aggregation = "sum"
        if not _agg_matches_column(aggregation, str(column), df):
            aggregation = "count"
        valid_measures.append({"column": str(column), "aggregation": aggregation})
    chart["measures"] = valid_measures

    split_by = chart.get("split_by")
    if split_by is not None and str(split_by) not in columns:
        chart["split_by"] = None

    # Chart type / data-shape compatibility (deterministic chart selector rules).
    if chart_type in {"donut", "pie"} and dimension is not None and dimension in columns:
        if df[dimension].nunique() > MAX_DONUT_CATEGORIES:
            issues.append(
                ConfigIssue(
                    "chart_type",
                    f"'{dimension}' has too many values for a {chart_type}; using 'bar'.",
                    repaired=True,
                )
            )
            chart["chart_type"] = "bar"
    if chart_type == "bar" and dimension is not None and dimension in columns:
        if df[dimension].nunique() > MAX_BAR_CATEGORIES * 2:
            issues.append(
                ConfigIssue(
                    "chart_type",
                    f"'{dimension}' is high-cardinality; using horizontal top-N 'hbar'.",
                    repaired=True,
                )
            )
            chart["chart_type"] = "hbar"
    if chart_type in {"line", "area"} and dimension is not None and dimension in columns:
        if not pd.api.types.is_datetime64_any_dtype(df[dimension]):
            issues.append(
                ConfigIssue(
                    "chart_type",
                    f"'{dimension}' is not a time column; using 'bar' for trend-free data.",
                    repaired=True,
                )
            )
            chart["chart_type"] = "bar"
    if chart_type == "scatter" and len(valid_measures) < 2:
        issues.append(
            ConfigIssue(
                "chart_type",
                "Scatter charts require at least two measures; using 'bar'.",
                repaired=True,
            )
        )
        chart["chart_type"] = "bar"

    return issues


def validate_config(raw: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Validate and repair a full dashboard configuration.

    Returns a cleaned, JSON-safe config dict. Unknown or unusable entries are
    dropped, and incompatible definitions are downgraded deterministically.
    """
    if not isinstance(raw, dict):
        return {}

    kpis: list[dict[str, Any]] = []
    for item in raw.get("kpis") or []:
        if not isinstance(item, dict):
            continue
        kpi = {key: value for key, value in item.items()}
        if not str(kpi.get("label", "")).strip():
            kpi["label"] = str(kpi.get("column") or "KPI")
        validate_kpi(kpi, df)
        if kpi.get("column") is not None or kpi.get("aggregation") in {"count", "ratio", "margin"}:
            kpis.append(kpi)
        if len(kpis) >= MAX_KPIS:
            break

    filters: list[dict[str, Any]] = []
    for item in raw.get("filters") or []:
        if not isinstance(item, dict):
            continue
        filt = {key: value for key, value in item.items()}
        if not str(filt.get("label", "")).strip():
            filt["label"] = str(filt.get("column") or "Filter")
        validate_filter(filt, df)
        if filt.get("column"):
            filters.append(filt)
        if len(filters) >= MAX_FILTERS:
            break

    charts: list[dict[str, Any]] = []
    for item in raw.get("charts") or []:
        if not isinstance(item, dict):
            continue
        chart = {key: value for key, value in item.items()}
        if not str(chart.get("id", "")).strip():
            chart["id"] = f"chart_{len(charts) + 1}"
        if not str(chart.get("title", "")).strip():
            chart["title"] = f"Chart {len(charts) + 1}"
        validate_chart(chart, df)
        if chart.get("measures"):
            charts.append(chart)
        if len(charts) >= MAX_CHARTS:
            break

    time_dimension = raw.get("time_dimension")
    if time_dimension is not None and str(time_dimension) not in df.columns:
        time_dimension = None

    primary_metric = raw.get("primary_metric")
    if primary_metric is not None and str(primary_metric) not in df.columns:
        primary_metric = None

    insight_topics = [
        str(topic).strip().lower()
        for topic in (raw.get("insight_topics") or [])
        if str(topic).strip().lower()
        in {"trend", "best_segment", "worst_segment", "anomaly", "profitability", "opportunity"}
    ]

    return {
        "title": str(raw.get("title") or "Executive Dashboard")[:120],
        "subtitle": str(raw.get("subtitle") or "")[:240],
        "data_source": str(raw.get("data_source") or "unknown")[:20],
        "generated_at": str(raw.get("generated_at") or "")[:40],
        "row_count": int(raw.get("row_count") or 0),
        "column_count": int(raw.get("column_count") or 0),
        "primary_metric": primary_metric,
        "time_dimension": time_dimension,
        "kpis": kpis,
        "filters": filters,
        "charts": charts,
        "insight_topics": insight_topics[:4],
    }


"""Deterministic runtime engine for the AI-planned dashboard.

Given a validated dashboard configuration, a DataFrame and the user's active
filter selections, this engine produces everything the UI needs:

- the filtered DataFrame
- KPI card payloads
- renderable chart payloads (embedded data records, no Python code)
- computed insight facts

The engine is deterministic and cache-friendly: filter changes recompute only
this layer, never the configuration or the LLM.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.chart_selector import pretty
    from dashboard.filters import apply_filters
    from dashboard.formatting import classify_trend, looks_like_money, prettify_name
    from dashboard.kpis import _aggregate_series, compute_kpis
    from dashboard.models import (
        MAX_BAR_CATEGORIES,
        MAX_CHART_DATA_ROWS,
        MAX_DONUT_CATEGORIES,
        MAX_HIST_VALUES,
        MAX_SCATTER_POINTS,
        MAX_TABLE_ROWS,
        MAX_TREND_POINTS,
    )
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.chart_selector import pretty
    from project.dashboard.filters import apply_filters
    from project.dashboard.formatting import classify_trend, looks_like_money, prettify_name
    from project.dashboard.kpis import _aggregate_series, compute_kpis
    from project.dashboard.models import (
        MAX_BAR_CATEGORIES,
        MAX_CHART_DATA_ROWS,
        MAX_DONUT_CATEGORIES,
        MAX_HIST_VALUES,
        MAX_SCATTER_POINTS,
        MAX_TABLE_ROWS,
        MAX_TREND_POINTS,
    )


def _round(value: Any, digits: int = 2) -> Any:
    """Round a number for JSON-safe chart records."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if pd.isna(number):
        return None
    return round(number, digits)


def _time_granularity(series: pd.Series) -> str:
    """Pick a period bucket for a parsed datetime series based on its span."""
    valid = series.dropna()
    if valid.empty:
        return "D"
    span_days = max((valid.max() - valid.min()).days, 0)
    if span_days > 120:
        return "M"
    if span_days > 21:
        return "W"
    return "D"


# ---------- Chart data aggregation ----------


def _aggregate_grouped(
    df: pd.DataFrame,
    dimension: str,
    measure: dict[str, Any],
    limit: int,
    descending: bool = True,
) -> pd.Series:
    """Aggregate one measure by a categorical dimension, capped to top-N."""
    if dimension not in df.columns:
        return pd.Series(dtype=float)
    agg = str(measure.get("aggregation", "sum"))
    column = measure.get("column")
    if column not in df.columns:
        return pd.Series(dtype=float)
    grouped = df.groupby(df[dimension].astype(str))[column].agg(agg)
    grouped = grouped.sort_values(ascending=not descending)
    return grouped.head(limit)


def _time_series_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate a time-series chart (line/area) by an appropriate period bucket."""
    dimension = chart.get("dimension")
    measures = chart.get("measures") or []
    if not dimension or dimension not in df.columns or not measures:
        return None
    split_by = chart.get("split_by")
    if split_by is not None and split_by not in df.columns:
        split_by = None

    parsed = _as_datetime(df, dimension)
    working = df.copy()
    working["_period"] = parsed.dt.to_period(_time_granularity(parsed))
    working = working.dropna(subset=["_period"])

    measure = measures[0]
    column = measure.get("column")
    agg = str(measure.get("aggregation", "sum"))
    if column not in df.columns:
        return None

    records: list[dict[str, Any]] = []
    if split_by:
        grouped = working.groupby(["_period", df[split_by].astype(str)], sort=True)[column].agg(agg)
        for (period, series_name), value in grouped.items():
            if pd.isna(value):
                continue
            records.append(
                {"x": str(period), "series": str(series_name), "value": _round(value)}
            )
    else:
        grouped = working.groupby("_period")[column].agg(agg).sort_index()
        for period, value in grouped.items():
            if pd.isna(value):
                continue
            records.append({"x": str(period), "value": _round(value)})

    records = records[:MAX_TREND_POINTS]
    if not records:
        return None

    metric_label = prettify_name(column)
    period_label = {"D": "day", "W": "week", "M": "month"}.get(_time_granularity(parsed), "period")
    return {
        "chart_id": str(chart.get("id") or "chart_trend"),
        "chart_type": str(chart.get("chart_type", "line")),
        "title": str(chart.get("title") or f"{metric_label} over time"),
        "subtitle": f"{pretty(agg)} per {period_label}.",
        "x": "x",
        "y": "value",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


def _as_datetime(df: pd.DataFrame, column: str) -> pd.Series:
    """Parse a column into a datetime Series (safe on any dtype)."""
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return df[column]
    return pd.to_datetime(df[column], errors="coerce")

def _categorical_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate a categorical chart (bar/hbar/donut/pie) by a dimension."""
    dimension = chart.get("dimension")
    measures = chart.get("measures") or []
    if not dimension or dimension not in df.columns or not measures:
        return None
    split_by = chart.get("split_by")
    if split_by is not None and split_by not in df.columns:
        split_by = None

    measure = measures[0]
    column = measure.get("column")
    agg = str(measure.get("aggregation", "sum"))
    if column not in df.columns:
        return None

    chart_type = str(chart.get("chart_type", "bar"))
    limit = MAX_DONUT_CATEGORIES if chart_type in {"donut", "pie"} else MAX_BAR_CATEGORIES
    limit = min(limit, int(chart.get("max_points") or limit))

    records: list[dict[str, Any]] = []
    if split_by:
        working = df[[dimension, split_by, column]].copy()
        working[dimension] = working[dimension].astype(str)
        grouped = (
            working.groupby([dimension, split_by])[column]
            .agg(agg)
            .reset_index()
            .sort_values(column, ascending=False)
        )
        for row in grouped.itertuples(index=False):
            records.append(
                {"x": str(getattr(row, dimension)), "series": str(getattr(row, split_by)), "value": _round(getattr(row, column))}
            )
    else:
        grouped = _aggregate_grouped(df, dimension, measure, limit)
        for label, value in grouped.items():
            records.append({"x": str(label), "value": _round(value)})

    if not records:
        return None

    metric_label = prettify_name(column)
    return {
        "chart_id": str(chart.get("id") or "chart_segments"),
        "chart_type": chart_type,
        "title": str(chart.get("title") or f"{metric_label} by {prettify_name(dimension)}"),
        "subtitle": f"{pretty(agg)} of {metric_label} per {prettify_name(dimension).lower()}.",
        "x": "x",
        "y": "value",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


def _table_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Build a tabular chart payload: dimension column + one column per measure."""
    dimension = chart.get("dimension")
    measures = chart.get("measures") or []
    if not measures:
        return None

    if dimension and dimension in df.columns:
        grouped = df.groupby(df[dimension].astype(str), sort=False)
        records: list[dict[str, Any]] = []
        for label, group in grouped:
            if len(records) >= MAX_TABLE_ROWS:
                break
            record: dict[str, Any] = {"x": str(label)}
            for measure in measures:
                column = measure.get("column")
                if column not in df.columns:
                    continue
                record[prettify_name(column)] = _round(_aggregate_series(group[column], str(measure.get("aggregation", "sum"))))
            records.append(record)
    else:
        column = measures[0].get("column")
        if column not in df.columns:
            return None
        records = [
            {prettify_name(column): _round(value)}
            for value in pd.to_numeric(df[column], errors="coerce").dropna().head(MAX_TABLE_ROWS).tolist()
        ]
        if not records:
            return None

    if not records:
        return None

    return {
        "chart_id": str(chart.get("id") or "chart_table"),
        "chart_type": "table",
        "title": str(chart.get("title") or "Data table"),
        "subtitle": str(chart.get("subtitle") or ""),
        "x": "x",
        "y": "",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


def _scatter_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate a scatter chart from up to three numeric measures."""
    measures = chart.get("measures") or []
    dimension = chart.get("dimension")
    columns: list[str] = []
    if dimension and pd.api.types.is_numeric_dtype(df[dimension]) and dimension in df.columns:
        columns.append(dimension)
    for measure in measures[:3]:
        column = measure.get("column")
        if column and column in df.columns and column not in columns:
            columns.append(column)
    if len(columns) < 2:
        return None

    x_col, y_col = columns[0], columns[1]
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    mask = x_values.notna() & y_values.notna()
    x_values, y_values = x_values[mask], y_values[mask]
    if x_values.empty:
        return None

    records: list[dict[str, Any]] = []
    size_col = columns[2] if len(columns) >= 3 else None
    size_values = pd.to_numeric(df[size_col], errors="coerce") if size_col else None
    for index in range(len(x_values))[:MAX_SCATTER_POINTS]:
        record = {"x": _round(x_values.iloc[index], 4), "y": _round(y_values.iloc[index], 4)}
        if size_values is not None:
            record["size"] = _round(size_values.iloc[index], 4)
        records.append(record)

    if not records:
        return None
    return {
        "chart_id": str(chart.get("id") or "chart_scatter"),
        "chart_type": "scatter",
        "title": str(chart.get("title") or f"{prettify_name(x_col)} vs {prettify_name(y_col)}"),
        "subtitle": "One point per record; color highlights the largest Y value.",
        "x": "x",
        "y": "y",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


def _hist_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate a histogram for a single numeric measure."""
    measures = chart.get("measures") or []
    if not measures:
        return None
    column = measures[0].get("column")
    if not column or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().head(MAX_HIST_VALUES)
    if values.empty:
        return None
    records = [{"value": _round(float(value))} for value in values.tolist()]
    return {
        "chart_id": str(chart.get("id") or "chart_hist"),
        "chart_type": "hist",
        "title": str(chart.get("title") or f"Distribution of {prettify_name(column)}"),
        "subtitle": "Spread of values across all matching records.",
        "x": "value",
        "y": "count",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


def _heatmap_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Correlation heatmap over the numeric columns referenced by measures."""
    measures = chart.get("measures") or []
    columns = [m.get("column") for m in measures if m.get("column") in df.columns][:6]
    if len(columns) < 3:
        return None
    try:
        matrix = df[columns].corr(numeric_only=True)
    except Exception:
        return None
    if matrix.empty:
        return None

    records: list[dict[str, Any]] = []
    for row in matrix.index:
        for col in matrix.columns:
            value = matrix.loc[row, col]
            if pd.isna(value):
                continue
            records.append({"x": str(row), "y": str(col), "z": _round(float(value), 3)})
    if not records:
        return None

    return {
        "chart_id": str(chart.get("id") or "chart_correlation"),
        "chart_type": "heatmap",
        "title": str(chart.get("title") or "Metric correlation heatmap"),
        "subtitle": "Pearson correlation between numeric metrics.",
        "x": "x",
        "y": "y",
        "data": records,
        "data_note": "",
        "width_span": int(chart.get("width_span", 6)),
    }


_CHART_BUILDERS = {
    "line": _time_series_chart,
    "area": _time_series_chart,
    "bar": _categorical_chart,
    "hbar": _categorical_chart,
    "donut": _categorical_chart,
    "pie": _categorical_chart,
    "table": _table_chart,
    "scatter": _scatter_chart,
    "hist": _hist_chart,
    "heatmap": _heatmap_chart,
}


def compute_chart(df: pd.DataFrame, chart: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate one chart configuration into a renderable payload."""
    if not isinstance(chart, dict):
        return None
    chart_type = str(chart.get("chart_type", "")).lower()
    builder = _CHART_BUILDERS.get(chart_type)
    if builder is None:
        return None
    try:
        return builder(df, chart)
    except Exception:
        return None


def compute_charts(df: pd.DataFrame, charts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate every chart configuration into renderable payloads."""
    payloads: list[dict[str, Any]] = []
    for chart in charts:
        payload = compute_chart(df, chart)
        if payload is not None:
            payloads.append(payload)
    return payloads


def compute_dashboard(
    config: dict[str, Any],
    df: pd.DataFrame,
    active_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the full dashboard runtime payload for the current filter state.

    Returns a dict with ``filtered_df``, ``kpis``, ``charts`` and ``row_count``.
    This is the single deterministic entry point cached by the UI.
    """
    filtered = apply_filters(df, config.get("filters") or [], active_filters)
    kpis = compute_kpis(filtered, config.get("kpis") or [], config.get("time_dimension"))
    charts = compute_charts(filtered, config.get("charts") or [])
    return {
        "config": config,
        "filtered_df": filtered,
        "kpis": kpis,
        "charts": charts,
        "row_count": int(len(filtered)),
        "column_count": int(len(filtered.columns)),
    }


# ---------- Insight facts ----------


def compute_insight_facts(
    df: pd.DataFrame, config: dict[str, Any], kpis: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compute deterministic analytical facts for the AI insight engine.

    Facts are computed from the (filtered) data and passed to the LLM so the
    insights are grounded in actual numbers rather than raw row inspection.
    """
    facts: dict[str, Any] = {"row_count": int(len(df))}
    primary = config.get("primary_metric")
    if primary is None or primary not in df.columns:
        primary = next(iter(df.select_dtypes(include="number").columns), None)

    if primary is not None and primary in df.columns:
        series = pd.to_numeric(df[primary], errors="coerce").dropna()
        if not series.empty:
            facts["primary_metric"] = primary
            facts["primary_total"] = _round(float(series.sum()))
            facts["primary_mean"] = _round(float(series.mean()))

    # Trend direction over the time dimension.
    time_dimension = config.get("time_dimension")
    if time_dimension and time_dimension in df.columns and primary in df.columns:
        parsed = _as_datetime(df, time_dimension)
        working = pd.DataFrame(
            {"_period": parsed.dt.to_period(_time_granularity(parsed)), "_value": df[primary]}
        ).dropna(subset=["_period"])
        if not working.empty:
            trend_series = working.groupby("_period")["_value"].agg("sum")
            facts["trend"] = classify_trend(trend_series)
            facts["trend_periods"] = int(len(trend_series))
            if len(trend_series) >= 2:
                last_period = float(trend_series.iloc[-1])
                prev_period = float(trend_series.iloc[-2])
                if prev_period != 0:
                    facts["trend_change_pct"] = _round(
                        (last_period - prev_period) / abs(prev_period) * 100.0
                    )

    # Best/worst segment by primary metric (skip the time dimension).
    if primary is not None and primary in df.columns:
        time_dimension = config.get("time_dimension")
        dim = None
        for c in (config.get("charts") or []):
            candidate = c.get("dimension")
            if candidate and candidate in df.columns and candidate != time_dimension:
                dim = candidate
                break
        if dim is None:
            for column in df.columns:
                if column == time_dimension:
                    continue
                if df[column].dtype == object and 2 <= df[column].nunique() <= 50:
                    dim = column
                    break
        if dim is not None:
            grouped = (
                pd.to_numeric(df[primary], errors="coerce")
                .groupby(df[dim].astype(str))
                .sum()
                .dropna()
                .sort_values(ascending=False)
            )
            if not grouped.empty:
                facts["segment_dimension"] = dim
                facts["best_segment"] = {
                    "name": str(grouped.index[0]),
                    "value": _round(float(grouped.iloc[0])),
                }
                facts["worst_segment"] = {
                    "name": str(grouped.index[-1]),
                    "value": _round(float(grouped.iloc[-1])),
                }
                total = float(grouped.sum())
                if total:
                    facts["best_segment_share_pct"] = _round(float(grouped.iloc[0]) / total * 100.0)

    # Profitability proxy: compute a directional margin with a sensible pair order.
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c in df.columns]
    numerator_col = next((c for c in numeric_cols if any(t in str(c).lower() for t in ("profit", "margin", "income", "earnings"))), None)
    denominator_col = next(
        (c for c in numeric_cols if c != numerator_col and any(t in str(c).lower() for t in ("revenue", "sales", "amount", "income", "gross"))),
        None,
    )
    if numerator_col is None or denominator_col is None:
        if len(numeric_cols) >= 2:
            numerator_col, denominator_col = numeric_cols[0], numeric_cols[1]
        else:
            numerator_col = denominator_col = None
    if numerator_col and denominator_col and numerator_col != denominator_col:
        n_value = float(pd.to_numeric(df[numerator_col], errors="coerce").sum())
        d_value = float(pd.to_numeric(df[denominator_col], errors="coerce").sum())
        if d_value != 0:
            facts["margin_columns"] = [numerator_col, denominator_col]
            facts["margin_pct"] = _round(n_value / d_value * 100.0)

    facts["kpis"] = kpis
    return facts
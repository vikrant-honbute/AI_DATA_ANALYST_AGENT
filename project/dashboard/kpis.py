"""Deterministic KPI engine.

Computes KPI cards from a (filtered) DataFrame and a validated list of KPI
definitions. Only the allowed aggregations in ``models.VALID_AGGREGATIONS`` are
supported, and every value is JSON-safe. Derived metrics (ratio, margin) and
period-over-period deltas are computed from the data, never hallucinated.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

try:
    from dashboard.formatting import format_compact, format_kpi_value, looks_like_money
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.formatting import format_compact, format_kpi_value, looks_like_money


def _to_number(value: Any) -> float | None:
    """Coerce a scalar into a float or None (JSON-safe)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _aggregate_series(series: pd.Series, aggregation: str) -> float | None:
    """Apply one aggregation to a series; returns None on empty input."""
    if aggregation == "nunique":
        cleaned = series.dropna()
        return _to_number(float(cleaned.nunique())) if not cleaned.empty else None
    if aggregation == "count":
        cleaned = series.dropna()
        return _to_number(float(len(cleaned))) if not cleaned.empty else None

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    if aggregation == "sum":
        result = numeric.sum()
    elif aggregation == "mean":
        result = numeric.mean()
    elif aggregation == "median":
        result = numeric.median()
    elif aggregation == "min":
        result = numeric.min()
    elif aggregation == "max":
        result = numeric.max()
    else:
        return None
    return _to_number(result)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator/denominator*100 without division-by-zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _to_number(numerator / denominator * 100.0)


def period_change(
    df: pd.DataFrame, kpi: dict[str, Any], time_dimension: str
):
    """Compute the period-over-period change for a KPI when a time column exists.

    Returns ``(label, delta, tone)`` or None when the trend cannot be computed.
    """
    if not time_dimension or time_dimension not in df.columns:
        return None
    column = kpi.get("column")
    aggregation = str(kpi.get("aggregation", "sum")).lower()
    if not column or aggregation not in {"sum", "count", "nunique", "mean", "median", "min", "max"}:
        if aggregation in {"ratio", "margin"}:
            column = kpi.get("column") or kpi.get("denominator")
            aggregation = "sum"
        else:
            return None
    if not column or column not in df.columns:
        return None

    parsed = pd.to_datetime(df[time_dimension], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None
    span_days = max((valid.max() - valid.min()).days, 0)
    granularity = "M" if span_days > 120 else ("W" if span_days > 21 else "D")
    label = {"D": "day", "W": "week", "M": "month"}.get(granularity, "period")

    working = pd.DataFrame(
        {"_period": parsed.dt.to_period(granularity), "_value": df[column]}
    ).dropna(subset=["_period"])
    if working.empty:
        return None
    agg_series = working.groupby("_period")["_value"].agg(aggregation)
    values = [float(v) for v in agg_series.values if pd.notna(v)]
    if len(values) < 2:
        return None
    previous, last = values[-2], values[-1]
    if previous == 0:
        return None
    pct = (last - previous) / abs(previous) * 100.0
    if abs(pct) < 0.05:
        return f"→ flat vs prev {label}", "0.0%", "neutral"
    if pct > 0:
        return f"▲ +{pct:.1f}% vs prev {label}", f"+{pct:.1f}%", "success"
    return f"▼ {pct:.1f}% vs prev {label}", f"{pct:.1f}%", "danger"


def prettify_agg(aggregation: str) -> str:
    """Human label for an aggregation operation."""
    return {
        "sum": "Total",
        "count": "Count",
        "nunique": "Distinct",
        "mean": "Average",
        "median": "Median",
        "min": "Minimum",
        "max": "Maximum",
        "ratio": "Ratio",
        "margin": "Margin",
    }.get(aggregation, aggregation.title())



def compute_kpi_value(df: pd.DataFrame, kpi: dict[str, Any]) -> dict[str, Any]:
    """Compute the display payload for one KPI definition."""
    aggregation = str(kpi.get("aggregation", "sum")).lower()
    column = kpi.get("column")
    denominator = kpi.get("denominator")
    fmt = str(kpi.get("format", "number")).lower()
    label = str(kpi.get("label", "")).strip() or str(column or "Value")

    raw: float | None = None
    sub = str(kpi.get("sub", "") or "")
    tone = str(kpi.get("tone", "neutral")).lower()

    if aggregation == "count" and (column is None or column not in df.columns):
        raw = float(len(df))
    elif aggregation in {"ratio", "margin"} and column and denominator and denominator in df.columns:
        num_series = pd.to_numeric(df[column], errors="coerce")
        den_series = pd.to_numeric(df[denominator], errors="coerce")
        if aggregation == "margin":
            raw = _safe_ratio(_aggregate_series(num_series, "sum"), _aggregate_series(den_series, "sum"))
        else:
            raw = _safe_ratio(_aggregate_series(num_series, "mean"), _aggregate_series(den_series, "mean"))
    elif column and column in df.columns:
        raw = _aggregate_series(df[column], aggregation)
    else:
        raw = None

    if raw is None:
        value_text = "—"
    else:
        if fmt == "money" or (fmt == "number" and looks_like_money(label)):
            value_text = format_kpi_value(raw, money=True)
        elif fmt == "percent":
            value_text = f"{raw:.1f}%"
        elif fmt == "int":
            value_text = f"{int(raw):,}"
        else:
            value_text = format_compact(raw)

    if not sub:
        metric = str(column or "")
        if metric:
            sub = f"{prettify_agg(aggregation)} of {metric}"
        elif aggregation == "count":
            sub = "records matching filters"

    if aggregation in {"ratio", "margin"}:
        tone = "info"
    else:
        tone = tone if tone in {"neutral", "info", "success", "danger", "warning"} else "neutral"

    return {
        "label": label,
        "icon": str(kpi.get("icon", "📊")),
        "value": value_text,
        "sub": sub,
        "delta": str(kpi.get("delta", "")) or "",
        "tone": tone,
        "raw_value": raw,
    }


def compute_kpis(
    df: pd.DataFrame, kpis: list[dict[str, Any]], time_dimension: str | None = None
) -> list[dict[str, Any]]:
    """Compute all KPI cards, including period-over-period deltas when enabled."""
    computed: list[dict[str, Any]] = []
    for kpi in kpis:
        if not isinstance(kpi, dict):
            continue
        payload = compute_kpi_value(df, kpi)
        if kpi.get("delta") and time_dimension:
            change = period_change(df, kpi, time_dimension)
            if change is not None:
                _, delta_text, delta_tone = change
                payload["delta"] = delta_text
                if delta_tone != "neutral":
                    payload["tone"] = delta_tone
        computed.append(payload)
    return computed

"""Dashboard spec schema: validation and normalization helpers.

The dashboard spec is a plain JSON-safe dict so it can live in agent state,
flow through the graph, be persisted in MongoDB session memory, and be
rendered by any UI layer.
"""

from __future__ import annotations

import math
from typing import Any

MAX_CHARTS = 8
MAX_CHART_DATA_ROWS = 200
MAX_KPI_CARDS = 6
MAX_INSIGHT_BULLETS = 6
MAX_RECOMMENDATION_BULLETS = 4
MAX_TEXT_LEN = 1000

VALID_CHART_TYPES = {"line", "bar", "donut", "hist", "heatmap"}
VALID_TONES = {"success", "danger", "info", "warning", "neutral"}


def _clean_text(value: Any, limit: int = MAX_TEXT_LEN) -> str:
    """Coerce a value into a bounded, stripped string."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _clean_int(value: Any, default: int = 0) -> int:
    """Coerce a value into an int with a fallback default."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_number(value: Any) -> Any:
    """Coerce a record value into a JSON-safe number, keeping text as-is."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _clean_records(raw_records: Any, limit: int = MAX_CHART_DATA_ROWS) -> list[dict[str, Any]]:
    """Normalize chart data records into bounded JSON-safe dicts."""
    if not isinstance(raw_records, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for record in raw_records[:limit]:
        if not isinstance(record, dict):
            continue
        cleaned_record: dict[str, Any] = {}
        for key, value in list(record.items())[:10]:
            cleaned_record[str(key)] = _clean_number(value)
        if cleaned_record:
            cleaned.append(cleaned_record)
    return cleaned


def _clean_kpis(raw_kpis: Any) -> list[dict[str, Any]]:
    """Normalize KPI card payloads."""
    if not isinstance(raw_kpis, list):
        return []

    kpis: list[dict[str, Any]] = []
    for item in raw_kpis[:MAX_KPI_CARDS]:
        if not isinstance(item, dict):
            continue
        label = _clean_text(item.get("label"), 60)
        value = _clean_text(item.get("value"), 60)
        if not label or not value:
            continue
        tone = _clean_text(item.get("tone"), 20) or "neutral"
        if tone not in VALID_TONES:
            tone = "neutral"
        kpis.append(
            {
                "label": label,
                "icon": _clean_text(item.get("icon"), 8) or "📊",
                "value": value,
                "sub": _clean_text(item.get("sub"), 120),
                "tone": tone,
                "delta": _clean_text(item.get("delta"), 120) or None,
            }
        )
    return kpis


def _clean_charts(raw_charts: Any) -> list[dict[str, Any]]:
    """Normalize chart payloads, dropping unusable entries."""
    if not isinstance(raw_charts, list):
        return []

    charts: list[dict[str, Any]] = []
    for item in raw_charts[:MAX_CHARTS]:
        if not isinstance(item, dict):
            continue
        chart_type = _clean_text(item.get("chart_type"), 20).lower()
        if chart_type not in VALID_CHART_TYPES:
            continue
        data = _clean_records(item.get("data"))
        if not data:
            continue
        charts.append(
            {
                "chart_id": _clean_text(item.get("chart_id"), 60) or f"chart_{len(charts) + 1}",
                "chart_type": chart_type,
                "title": _clean_text(item.get("title"), 120) or "Chart",
                "subtitle": _clean_text(item.get("subtitle"), 240),
                "x": _clean_text(item.get("x"), 80),
                "y": _clean_text(item.get("y"), 80),
                "data": data,
                "data_note": _clean_text(item.get("data_note"), 240),
            }
        )
    return charts


def _clean_bullet_list(raw_items: Any, limit: int) -> list[str]:
    """Normalize a list of insight/recommendation bullets."""
    if not isinstance(raw_items, list):
        return []
    cleaned: list[str] = []
    for item in raw_items:
        text = _clean_text(item, 300)
        if text:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def normalize_spec(raw: Any) -> dict[str, Any] | None:
    """Validate and repair a raw dashboard spec.

    Returns a clean, JSON-safe spec dict, or None when the payload is not a
    usable dashboard (e.g., missing both KPIs and charts).
    """
    if not isinstance(raw, dict):
        return None

    kpis = _clean_kpis(raw.get("kpis"))
    charts = _clean_charts(raw.get("charts"))
    if not kpis and not charts:
        return None

    raw_time_range = raw.get("time_range")
    time_range: dict[str, Any] | None = None
    if isinstance(raw_time_range, dict):
        start = _clean_text(raw_time_range.get("start"), 40)
        end = _clean_text(raw_time_range.get("end"), 40)
        if start and end:
            time_range = {"start": start, "end": end}

    return {
        "title": _clean_text(raw.get("title"), 120) or "Executive Dashboard",
        "subtitle": _clean_text(raw.get("subtitle"), 240),
        "data_source": _clean_text(raw.get("data_source"), 20) or "unknown",
        "generated_at": _clean_text(raw.get("generated_at"), 40),
        "row_count": _clean_int(raw.get("row_count")),
        "column_count": _clean_int(raw.get("column_count")),
        "time_range": time_range,
        "kpis": kpis,
        "charts": charts,
        "executive_summary": _clean_text(raw.get("executive_summary")),
        "insights": _clean_bullet_list(raw.get("insights"), MAX_INSIGHT_BULLETS),
        "recommendations": _clean_bullet_list(raw.get("recommendations"), MAX_RECOMMENDATION_BULLETS),
    }

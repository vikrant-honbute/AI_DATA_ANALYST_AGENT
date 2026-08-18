"""Generic, dataset-aware dashboard filter engine.

Filter definitions come from the AI dashboard configuration. This module builds
the available options from real data and applies the user's active selections to
a DataFrame. Nothing here is hardcoded to any dataset: filter columns and types
are derived per dataset by the dashboard planner.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.models import MAX_FILTER_OPTIONS
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.models import MAX_FILTER_OPTIONS


# ---------- Option building ----------


def _build_year_options(df: pd.DataFrame, column: str) -> list[Any]:
    """Return sorted distinct years for a datetime column."""
    parsed = pd.to_datetime(df[column], errors="coerce")
    valid = parsed.dropna()
    return sorted({int(value.year) for value in valid})


def _build_categorical_options(df: pd.DataFrame, column: str, limit: int) -> list[Any]:
    """Return sorted distinct string values, capped for safety."""
    values = df[column].dropna().astype(str).unique()
    return sorted(values[:limit])


def build_filter_options(
    df: pd.DataFrame, filter_specs: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Build the available options for each filter from the actual data.

    Returns a dict keyed by filter id with entries shaped as::

        {"type": ..., "column": ..., "label": ..., "options": [...]}
        # or {"type": "numeric_range", "min": float, "max": float}
        # or {"type": "date_range", "min": str, "max": str}
    """
    options_map: dict[str, dict[str, Any]] = {}
    for filt in filter_specs:
        if not isinstance(filt, dict):
            continue
        filter_id = str(filt.get("id", "")).strip()
        column = str(filt.get("column", "")).strip()
        filter_type = str(filt.get("type", "categorical_multi")).lower()
        label = str(filt.get("label", "")).strip() or column or "Filter"
        limit = int(filt.get("options_limit") or MAX_FILTER_OPTIONS)
        if not filter_id or not column or column not in df.columns:
            continue

        entry: dict[str, Any] = {
            "type": filter_type,
            "column": column,
            "label": label,
        }
        if filter_type == "date_year":
            entry["options"] = _build_year_options(df, column)
        elif filter_type == "date_range":
            parsed = pd.to_datetime(df[column], errors="coerce").dropna()
            if parsed.empty:
                entry["options"] = []
                entry["min"] = None
                entry["max"] = None
            else:
                entry["min"] = str(parsed.min().date())
                entry["max"] = str(parsed.max().date())
                entry["options"] = [entry["min"], entry["max"]]
        elif filter_type == "numeric_range":
            numeric = pd.to_numeric(df[column], errors="coerce").dropna()
            if numeric.empty:
                entry["min"] = None
                entry["max"] = None
            else:
                entry["min"] = float(numeric.min())
                entry["max"] = float(numeric.max())
        else:
            entry["options"] = _build_categorical_options(df, column, limit)

        options_map[filter_id] = entry
    return options_map


# ---------- Filter application ----------


def _filter_year(df: pd.DataFrame, column: str, years: list[Any]) -> pd.DataFrame:
    parsed = pd.to_datetime(df[column], errors="coerce")
    try:
        wanted = {int(value) for value in years if value is not None}
    except (TypeError, ValueError):
        return df
    if not wanted:
        return df
    return df[parsed.dt.year.isin(wanted).fillna(False)]


def _filter_date_range(df: pd.DataFrame, column: str, start: Any, end: Any) -> pd.DataFrame:
    parsed = pd.to_datetime(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None and str(start).strip():
        start_date = pd.to_datetime(str(start), errors="coerce")
        if pd.notna(start_date):
            mask &= parsed >= start_date
    if end is not None and str(end).strip():
        end_date = pd.to_datetime(str(end), errors="coerce")
        if pd.notna(end_date):
            mask &= parsed <= end_date
    return df[mask.fillna(False)]


def _filter_categorical_single(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    if value is None or str(value).strip() in {"", "All", "None", "—"}:
        return df
    return df[df[column].astype(str) == str(value)]


def _filter_categorical_multi(df: pd.DataFrame, column: str, values: list[Any]) -> pd.DataFrame:
    if not values:
        return df
    wanted = {str(value) for value in values}
    return df[df[column].astype(str).isin(wanted)]


def _filter_numeric_range(df: pd.DataFrame, column: str, minimum: Any, maximum: Any) -> pd.DataFrame:
    numeric = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if minimum is not None and str(minimum).strip() not in {"", "None"}:
        mask &= numeric >= float(minimum)
    if maximum is not None and str(maximum).strip() not in {"", "None"}:
        mask &= numeric <= float(maximum)
    return df[mask.fillna(False)]


def apply_filters(
    df: pd.DataFrame,
    filter_specs: list[dict[str, Any]],
    active_filters: dict[str, Any] | None,
) -> pd.DataFrame:
    """Apply all active filter selections to a DataFrame.

    ``active_filters`` maps filter id -> value. The value shape depends on the
    filter type:

    - date_year / categorical_multi -> list of values
    - date_range -> dict {"start": str, "end": str} or (start, end)
    - categorical_single -> scalar
    - numeric_range -> dict {"min": float, "max": float} or (min, max)
    """
    if not active_filters or not filter_specs:
        return df

    working = df
    for filt in filter_specs:
        if not isinstance(filt, dict):
            continue
        filter_id = str(filt.get("id", "")).strip()
        column = str(filt.get("column", "")).strip()
        filter_type = str(filt.get("type", "categorical_multi")).lower()
        if not filter_id or not column or column not in working.columns:
            continue
        if filter_id not in active_filters:
            continue
        value = active_filters[filter_id]
        if value is None:
            continue

        if filter_type == "date_year":
            working = _filter_year(working, column, value)
        elif filter_type == "date_range":
            if isinstance(value, (tuple, list)) and len(value) == 2:
                start, end = value
            elif isinstance(value, dict):
                start, end = value.get("start"), value.get("end")
            else:
                continue
            working = _filter_date_range(working, column, start, end)
        elif filter_type == "categorical_single":
            working = _filter_categorical_single(working, column, value)
        elif filter_type == "categorical_multi":
            if not isinstance(value, (list, tuple)):
                value = [value]
            working = _filter_categorical_multi(working, column, value)
        elif filter_type == "numeric_range":
            if isinstance(value, (tuple, list)) and len(value) == 2:
                minimum, maximum = value
            elif isinstance(value, dict):
                minimum, maximum = value.get("min"), value.get("max")
            else:
                continue
            working = _filter_numeric_range(working, column, minimum, maximum)

        if working.empty:
            break

    return working

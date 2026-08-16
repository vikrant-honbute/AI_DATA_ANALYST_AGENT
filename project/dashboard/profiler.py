"""Deterministic data profiler for the dashboard engine.

Detects column roles (time, numeric metric, categorical, id-like) and derives
aggregation hints so the layout builder can assemble a professional dashboard
without any LLM involvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_TIME_NAME_HINTS = (
    "date", "time", "day", "month", "year", "week", "period", "quarter", "dt", "ym",
)
_ID_NAME_HINTS = (
    "id", "uuid", "guid", "key", "sku", "zip", "phone", "email", "code", "ssn",
)
_RATE_NAME_HINTS = ("price", "rate", "margin", "avg", "average", "ratio", "score", "rate_of")
_METRIC_NAME_PRIORITY = (
    ("revenue", 100),
    ("sales", 95),
    ("profit", 90),
    ("amount", 85),
    ("income", 80),
    ("cost", 70),
    ("total", 65),
    ("quantity", 60),
    ("qty", 60),
    ("price", 55),
    ("value", 50),
    ("count", 40),
)
_MIN_TIME_PARSE_RATIO = 0.6
_MIN_TIME_UNIQUE = 3


@dataclass
class DataProfile:
    """Structured profile of a DataFrame used for dashboard layout decisions."""

    rows: int
    columns: list[str]
    time_column: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    time_granularity: str = "none"  # "D" | "W" | "M" | "none"
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    id_like_columns: list[str] = field(default_factory=list)
    metric_candidates: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    categorical_unique_counts: dict[str, int] = field(default_factory=dict)
    missing_ratio: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-safe summary suitable for LLM prompts."""
        return {
            "rows": self.rows,
            "columns": list(self.columns),
            "time_column": self.time_column,
            "time_range": (
                {"start": self.time_start, "end": self.time_end}
                if self.time_column
                else None
            ),
            "time_granularity": self.time_granularity,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "id_like_columns": list(self.id_like_columns),
            "primary_metric": self.primary_metric,
            "metric_candidates": list(self.metric_candidates),
            "categorical_unique_counts": dict(self.categorical_unique_counts),
        }


def is_id_like_name(name: Any) -> bool:
    """Return True when a column name looks like an identifier."""
    lowered = str(name).lower().strip()
    return any(hint in lowered for hint in _ID_NAME_HINTS)


def aggregate_for_metric(name: Any) -> str:
    """Pick the natural aggregation for a metric name: 'sum' or 'mean'."""
    lowered = str(name).lower()
    if any(hint in lowered for hint in _RATE_NAME_HINTS):
        return "mean"
    return "sum"


def parse_time_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    """Parse a column into a datetime Series, or None when not time-like."""
    if column not in df.columns:
        return None

    parsed = pd.to_datetime(df[column], errors="coerce")
    if parsed.dropna().empty:
        return None
    return parsed


def _score_metric(name: str, series: pd.Series) -> float:
    """Score a numeric column as a dashboard metric candidate."""
    lowered = str(name).lower()
    name_score = 0.0
    for hint, weight in _METRIC_NAME_PRIORITY:
        if hint in lowered:
            name_score = max(name_score, weight)
            break

    valid_ratio = float(series.notna().mean())
    unique_count = int(series.dropna().nunique())
    variability = 0.0
    if unique_count > 1 and valid_ratio > 0:
        std = float(series.std())
        mean = abs(float(series.mean()))
        variability = std / max(mean, 1e-9) if mean else 1.0

    return name_score + valid_ratio * 40.0 + min(variability, 1.0) * 20.0


def _pick_time_column(df: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """Find the best time column (datetime dtype first, then parseable strings)."""
    best: tuple[float, str, pd.Series] | None = None

    for column in df.columns:
        series = df[column]
        parsed: pd.Series | None = None

        if pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
        elif series.dtype == object:
            sample = series.dropna().head(200)
            if sample.empty:
                continue
            probed = pd.to_datetime(sample, errors="coerce")
            if probed.notna().mean() < _MIN_TIME_PARSE_RATIO:
                continue
            parsed = pd.to_datetime(series, errors="coerce")

        if parsed is None:
            continue

        valid = parsed.dropna()
        if valid.empty or int(valid.nunique()) < _MIN_TIME_UNIQUE:
            continue

        score = float(valid.notna().mean()) * 100.0 + len(valid) / max(len(df), 1)
        if best is None or score > best[0]:
            best = (score, str(column), parsed)

    if best is None:
        return None
    return best[1], best[2]


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Profile a DataFrame and return column roles for dashboard building."""
    profile = DataProfile(rows=int(len(df)), columns=[str(c) for c in df.columns])
    if df.empty:
        return profile

    # --- Time column ---
    time_hit = _pick_time_column(df)
    if time_hit is not None:
        column, parsed = time_hit
        valid = parsed.dropna()
        profile.time_column = column
        profile.time_start = str(valid.min().date())
        profile.time_end = str(valid.max().date())
        span_days = max((valid.max() - valid.min()).days, 0)
        profile.time_granularity = "M" if span_days > 120 else ("W" if span_days > 21 else "D")

    # --- Numeric columns and metric candidates ---
    numeric_columns: list[str] = []
    for column in df.select_dtypes(include="number").columns:
        if column == profile.time_column:
            continue
        numeric_columns.append(str(column))
    profile.numeric_columns = numeric_columns

    for column in numeric_columns:
        series = df[column].dropna()
        if series.empty:
            continue
        lowered = column.lower()
        if is_id_like_name(column):
            profile.id_like_columns.append(column)
            continue
        if int(series.nunique()) <= 2 and (
            series.dtype == bool
            or lowered.startswith(("is_", "has_"))
            or lowered.endswith(("_flag", "_indicator"))
        ):
            continue
        profile.metric_candidates.append(column)

    profile.metric_candidates = sorted(
        profile.metric_candidates,
        key=lambda name: _score_metric(name, df[name]),
        reverse=True,
    )
    if profile.metric_candidates:
        profile.primary_metric = profile.metric_candidates[0]

    # --- Categorical columns and id-like detection ---
    for column in df.columns:
        if column in numeric_columns or column == profile.time_column:
            continue
        series = df[column]
        if series.dtype not in (object, "category") and not pd.api.types.is_string_dtype(series):
            continue
        unique_count = int(series.dropna().nunique())
        if is_id_like_name(column) or unique_count >= max(len(df) // 2, 200):
            profile.id_like_columns.append(str(column))
            continue
        profile.categorical_columns.append(str(column))
        profile.categorical_unique_counts[str(column)] = unique_count

    # --- Missing value ratios (rounded for prompt use) ---
    for column in df.columns:
        ratio = float(df[column].isna().mean())
        profile.missing_ratio[str(column)] = round(ratio, 3)

    return profile

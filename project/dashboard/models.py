"""Dashboard configuration schema for the AI-planned dashboard.

The dashboard configuration describes WHAT to show (KPIs, filters, charts)
without embedding any computed data. A deterministic runtime engine
(``dashboard.engine``) turns the configuration plus a DataFrame into rendered
KPIs, chart payloads and insight facts. Keeping the configuration separate from
the data means filter changes never require an LLM call: only the engine
recomputes.

The Pydantic models below mirror the JSON-safe dicts that flow through agent
state, so the AI dashboard planner can parse LLM output directly into them and
any UI can read them back as plain dicts via ``as_dict()``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------- Allowed vocabularies (deterministic, LLM-constrained) ----------

VALID_AGGREGATIONS = {
    "sum",
    "count",
    "nunique",
    "mean",
    "median",
    "min",
    "max",
    "ratio",
    "margin",
}

VALID_CHART_TYPES = {
    "line",
    "bar",
    "hbar",
    "area",
    "donut",
    "pie",
    "scatter",
    "hist",
    "heatmap",
    "table",
}

VALID_FILTER_TYPES = {
    "date_year",
    "date_range",
    "categorical_single",
    "categorical_multi",
    "numeric_range",
}

VALID_INSIGHT_TOPICS = {
    "trend",
    "best_segment",
    "worst_segment",
    "anomaly",
    "profitability",
    "opportunity",
}

VALID_FORMATS = {"number", "money", "percent", "int"}
VALID_TONES = {"neutral", "info", "success", "danger", "warning"}

# ---------- Hard limits (fail-graceful caps) ----------

MAX_KPIS = 6
MAX_FILTERS = 6
MAX_CHARTS = 8
MAX_CHART_DATA_ROWS = 200
MAX_TREND_POINTS = 200
MAX_BAR_CATEGORIES = 12
MAX_DONUT_CATEGORIES = 8
MAX_TABLE_ROWS = 50
MAX_SCATTER_POINTS = 500
MAX_HIST_VALUES = 1000
MAX_FILTER_OPTIONS = 500

DEFAULT_KPI_ICON = "📊"

# ---------- Pydantic schemas ----------


class KpiSpec(BaseModel):
    """One KPI card: a measure column plus an aggregation operation."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    label: str = ""
    column: str | None = None
    aggregation: str = "sum"
    denominator: str | None = Field(
        default=None, description="Second column used by ratio/margin KPIs."
    )
    format: str = "number"
    tone: str = "neutral"
    icon: str = DEFAULT_KPI_ICON
    delta: bool = Field(
        default=False,
        description="When True, also show the period-over-period change as the card sub-line.",
    )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {key: value for key, value in self.model_dump().items()}


class FilterSpec(BaseModel):
    """One interactive dashboard filter bound to a real column."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    label: str = ""
    column: str = ""
    type: str = "categorical_multi"
    options_limit: int = MAX_FILTER_OPTIONS

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {key: value for key, value in self.model_dump().items()}


class MeasureSpec(BaseModel):
    """A measure inside a chart: a column plus an aggregation."""

    model_config = ConfigDict(extra="ignore")

    column: str
    aggregation: str = "sum"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {key: value for key, value in self.model_dump().items()}


class ChartSpec(BaseModel):
    """One chart definition. The engine aggregates data from this spec."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    chart_type: str = "bar"
    title: str = ""
    subtitle: str = ""
    dimension: str | None = Field(
        default=None, description="X/category column (date or categorical)."
    )
    measures: list[MeasureSpec] = Field(default_factory=list)
    split_by: str | None = Field(
        default=None, description="Optional categorical column for grouped series."
    )
    max_points: int = Field(default=60, ge=1, le=MAX_CHART_DATA_ROWS)
    width_span: int = Field(default=6, ge=1, le=12)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        data = self.model_dump()
        data["measures"] = [measure.as_dict() for measure in self.measures]
        return data


class DashboardConfig(BaseModel):
    """Top-level AI-generated dashboard configuration (no embedded data)."""

    model_config = ConfigDict(extra="ignore")

    title: str = "Executive Dashboard"
    subtitle: str = ""
    data_source: str = "unknown"
    generated_at: str = ""
    row_count: int = 0
    column_count: int = 0
    primary_metric: str | None = None
    time_dimension: str | None = None
    kpis: list[KpiSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    insight_topics: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "data_source": self.data_source,
            "generated_at": self.generated_at,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "primary_metric": self.primary_metric,
            "time_dimension": self.time_dimension,
            "kpis": [kpi.as_dict() for kpi in self.kpis],
            "filters": [filt.as_dict() for filt in self.filters],
            "charts": [chart.as_dict() for chart in self.charts],
            "insight_topics": list(self.insight_topics),
        }


def config_from_dict(raw: Any) -> DashboardConfig | None:
    """Parse a plain dict (or None) into a DashboardConfig, or None on failure."""
    if not isinstance(raw, dict):
        return None
    try:
        return DashboardConfig(**raw)
    except Exception:
        return None

    """A measure inside a chart: a column plus an aggregation."""

    model_config = ConfigDict(extra="ignore")

    column: str
    aggregation: str = "sum"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation."""
        return {key: value for key, value in self.model_dump().items()}


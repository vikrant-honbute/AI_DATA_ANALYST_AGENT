"""Dashboard engine: AI-planned configuration + deterministic runtime rendering."""

from .analyst import refine_spec_with_llm
from .chart_selector import recommend_chart_type, suggest_chart_configs
from .charts import build_figure_for_chart
from .context import build_dashboard_context, build_filter_context_text
from .engine import compute_chart, compute_charts, compute_dashboard, compute_insight_facts
from .filters import apply_filters, build_filter_options
from .formatting import classify_trend, format_compact, prettify_name
from .insights import build_dashboard_insights, refine_insights_with_llm
from .kpis import compute_kpi_value, compute_kpis
from .layout import build_dashboard_spec
from .models import (
    MAX_CHARTS,
    MAX_FILTERS,
    MAX_KPIS,
    VALID_AGGREGATIONS,
    VALID_CHART_TYPES,
    VALID_FILTER_TYPES,
    KpiSpec,
    ChartSpec,
    DashboardConfig,
    FilterSpec,
)
from .planner import build_dashboard_config, fallback_config
from .profiler import DataProfile, aggregate_for_metric, parse_time_column, profile_dataframe
from .spec import normalize_spec
from .validate import validate_config

__all__ = [
    "ChartSpec",
    "DataProfile",
    "DashboardConfig",
    "FilterSpec",
    "KpiSpec",
    "MAX_CHARTS",
    "MAX_FILTERS",
    "MAX_KPIS",
    "VALID_AGGREGATIONS",
    "VALID_CHART_TYPES",
    "VALID_FILTER_TYPES",
    "aggregate_for_metric",
    "apply_filters",
    "build_dashboard_config",
    "build_dashboard_context",
    "build_dashboard_insights",
    "build_dashboard_spec",
    "build_filter_context_text",
    "build_filter_options",
    "build_figure_for_chart",
    "classify_trend",
    "compute_chart",
    "compute_charts",
    "compute_dashboard",
    "compute_insight_facts",
    "compute_kpi_value",
    "compute_kpis",
    "fallback_config",
    "format_compact",
    "normalize_spec",
    "parse_time_column",
    "prettify_name",
    "profile_dataframe",
    "recommend_chart_type",
    "refine_insights_with_llm",
    "refine_spec_with_llm",
    "suggest_chart_configs",
    "validate_config",
]

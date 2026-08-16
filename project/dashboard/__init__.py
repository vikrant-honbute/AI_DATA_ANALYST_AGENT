"""Dashboard engine: deterministic spec building with optional LLM narrative."""

from .analyst import refine_spec_with_llm
from .charts import build_figure_for_chart
from .formatting import classify_trend, format_compact, prettify_name
from .layout import build_dashboard_spec
from .profiler import DataProfile, aggregate_for_metric, parse_time_column, profile_dataframe
from .spec import normalize_spec

__all__ = [
    "DataProfile",
    "aggregate_for_metric",
    "build_dashboard_spec",
    "build_figure_for_chart",
    "classify_trend",
    "format_compact",
    "normalize_spec",
    "parse_time_column",
    "prettify_name",
    "profile_dataframe",
    "refine_spec_with_llm",
]
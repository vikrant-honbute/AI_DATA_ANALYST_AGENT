"""Dashboard context for the Data Analyst Agent chat integration.

When the user asks a follow-up question while interacting with a dashboard, the
active filter state and a compact summary of the configuration are serialized so
the planner/critic/insight LLM calls understand the current view
(e.g. ``Region = West``). This is the bridge between the interactive dashboard
and the agent's graph.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from dashboard.formatting import prettify_name
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.formatting import prettify_name


def _value_label(value: Any, filter_type: str) -> str:
    """Render one active filter value in a human-readable form."""
    if value is None:
        return "All"
    if filter_type == "numeric_range":
        if isinstance(value, dict):
            minimum, maximum = value.get("min"), value.get("max")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            minimum, maximum = value
        else:
            return str(value)
        return f"{minimum} to {maximum}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "All"
    return str(value)


def build_filter_context_text(
    config: dict[str, Any] | None, active_filters: dict[str, Any] | None
) -> str:
    """Render the active filter state as a concise sentence for LLM prompts."""
    if not config or not active_filters:
        return "None"
    parts: list[str] = []
    for filt in config.get("filters") or []:
        if not isinstance(filt, dict):
            continue
        filter_id = str(filt.get("id", "")).strip()
        if filter_id not in active_filters:
            continue
        value = active_filters[filter_id]
        if value is None:
            continue
        label = str(filt.get("label", "")).strip() or prettify_name(filt.get("column", ""))
        rendered = _value_label(value, str(filt.get("type", "categorical_multi")))
        if rendered in {"All", "None", "—", ""}:
            continue
        parts.append(f"{label} = {rendered}")
    return "; ".join(parts) if parts else "None"


def build_dashboard_context(
    config: dict[str, Any] | None,
    active_filters: dict[str, Any] | None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the full dashboard context dict stored in graph state.

    Includes the active filters, key config facts, and (optionally) the latest
    computed KPI values so the agent can reason about the current view.
    """
    context: dict[str, Any] = {
        "dashboard": True,
        "active_filters_text": build_filter_context_text(config, active_filters),
    }
    if config:
        context["title"] = str(config.get("title") or "")
        context["time_dimension"] = config.get("time_dimension")
        context["primary_metric"] = config.get("primary_metric")
    if active_filters:
        context["active_filters"] = {
            str(key): value for key, value in active_filters.items() if value is not None
        }
    if runtime:
        context["row_count"] = int(runtime.get("row_count") or 0)
        kpis = runtime.get("kpis") or []
        context["kpis"] = [
            {"label": kpi.get("label"), "value": kpi.get("value"), "delta": kpi.get("delta")}
            for kpi in kpis
        ]
    return context

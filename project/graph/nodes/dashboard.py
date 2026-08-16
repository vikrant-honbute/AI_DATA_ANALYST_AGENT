"""Dashboard node: builds a professional dashboard spec from the routed data.

Runs instead of the executor when the planner flags dashboard intent. It
acquires a DataFrame (uploaded CSV or a capped PostgreSQL sample), profiles it,
builds a deterministic spec, optionally refines the narrative with the LLM,
and writes a JSON-safe dashboard_spec plus a readable final_result.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    from config import get_settings
    from dashboard import build_dashboard_spec, normalize_spec, profile_dataframe, refine_spec_with_llm
    from graph.state import AgentState
    from tools.sql_tool import fetch_postgres_schema, query_postgres
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.config import get_settings
    from project.dashboard import (
        build_dashboard_spec,
        normalize_spec,
        profile_dataframe,
        refine_spec_with_llm,
    )
    from project.graph.state import AgentState
    from project.tools.sql_tool import fetch_postgres_schema, query_postgres

logger = logging.getLogger(__name__)


def _acquire_dataframe(state: AgentState) -> tuple[pd.DataFrame | None, str]:
    """Fetch the source DataFrame for the dashboard from the routed source."""
    uploaded = state.get("uploaded_dataframe")
    if isinstance(uploaded, pd.DataFrame) and not uploaded.empty:
        return uploaded, "csv"

    if state.get("data_source") != "postgres":
        return None, str(state.get("data_source", "unknown"))

    try:
        settings = get_settings()
        schema_map = fetch_postgres_schema()
    except Exception as exc:
        logger.warning("dashboard[_acquire_dataframe] schema fetch failed: %s", exc)
        return None, "postgres"

    if not schema_map:
        return None, "postgres"

    table = next(iter(schema_map))
    schema_name, _, table_name = table.partition(".")
    max_rows = max(int(getattr(settings, "postgres_max_rows", 1000) or 1000), 1)
    sql = f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT {max_rows}'
    try:
        dataframe = query_postgres(sql)
    except Exception as exc:
        logger.warning("dashboard[_acquire_dataframe] sample query failed: %s", exc)
        return None, "postgres"

    if dataframe is None or dataframe.empty:
        return None, "postgres"
    return dataframe, "postgres"


def _render_spec_summary(spec: dict[str, Any]) -> str:
    """Render a readable text summary of the spec for CLI, critic, and memory."""
    lines = [
        f"Dashboard: {spec.get('title', 'Executive Dashboard')}",
        f"Scope: {spec.get('subtitle', '')}".strip(),
        "",
        "KPIs:",
    ]
    for kpi in spec.get("kpis", []):
        delta = kpi.get("delta")
        suffix = f" ({delta})" if delta else ""
        lines.append(f"- {kpi.get('label')}: {kpi.get('value')}{suffix}")

    charts = spec.get("charts", [])
    if charts:
        lines.append("")
        lines.append("Charts:")
        for index, chart in enumerate(charts, start=1):
            rows = len(chart.get("data", []))
            lines.append(
                f"{index}. {chart.get('title')} ({chart.get('chart_type')}, {rows} data points)"
            )

    summary = str(spec.get("executive_summary", "")).strip()
    if summary:
        lines.extend(["", f"Executive summary: {summary}"])

    insights = spec.get("insights", [])
    if insights:
        lines.extend(["", "Key facts:"])
        lines.extend(f"- {item}" for item in insights)

    return "\n".join(lines)


def dashboard_node(state: AgentState) -> AgentState:
    """Build the executive dashboard and return updated state."""
    query = str(state.get("query", "")).strip()
    run_id = str(state.get("run_id") or "").strip() or re.sub(
        r"[^a-f0-9]", "", str(state.get("session_id") or "")
    ).lower()
    retry_count = state.get("retry_count", 0)
    retry_count = retry_count if isinstance(retry_count, int) else 0

    base_result: dict[str, Any] = {
        **state,
        "last_execution_node": "dashboard",
        "run_id": run_id,
    }

    dataframe, source_label = _acquire_dataframe(state)
    if dataframe is None or dataframe.empty:
        reason = (
            "No tabular data was available for the routed source "
            f"('{source_label}'). Upload a CSV file or make sure a PostgreSQL table "
            "exists in the allowed schemas."
        )
        return {
            **base_result,
            "dashboard": True,
            "dashboard_spec": None,
            "final_result": f"Dashboard could not be built: {reason}",
            "intermediate_results": [
                {
                    "step": "Build executive dashboard",
                    "tool": "visualization",
                    "action": "dashboard",
                    "result": {"type": "dashboard", "status": "unavailable", "reason": reason},
                }
            ],
        }

    profile = profile_dataframe(dataframe)
    spec = build_dashboard_spec(dataframe, profile, query=query, focus_offset=retry_count)
    spec = refine_spec_with_llm(
        spec, profile.to_dict(), query, feedback=str(state.get("insights", ""))
    )
    spec["data_source"] = source_label
    spec["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    spec = normalize_spec(spec)
    if spec is None:
        spec = {
            "title": "Executive Dashboard",
            "subtitle": "",
            "data_source": source_label,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "time_range": None,
            "kpis": [],
            "charts": [],
            "executive_summary": "The dashboard could not be assembled from this data shape.",
            "insights": [],
            "recommendations": [],
        }

    final_result = _render_spec_summary(spec)
    chart_types = [chart.get("chart_type", "unknown") for chart in spec.get("charts", [])]

    return {
        **base_result,
        "dashboard": True,
        "dashboard_spec": spec,
        "final_result": final_result,
        "plan": [
            {
                "step": "Build executive dashboard",
                "tool": "visualization",
                "action": "dashboard",
            }
        ],
        "intermediate_results": [
            {
                "step": "Build executive dashboard",
                "tool": "visualization",
                "action": "dashboard",
                "result": {
                    "type": "dashboard",
                    "status": "built",
                    "title": spec.get("title", ""),
                    "kpi_count": len(spec.get("kpis", [])),
                    "chart_count": len(spec.get("charts", [])),
                    "chart_types": chart_types,
                },
            }
        ],
    }
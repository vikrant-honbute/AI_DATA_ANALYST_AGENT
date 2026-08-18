"""Dashboard node: builds an AI-planned dashboard configuration.

This node runs instead of the executor when the planner flags dashboard intent. It
acquires a DataFrame (uploaded CSV or a capped PostgreSQL sample), profiles it,
asks the AI dashboard planner for a validated configuration, then computes a
default runtime rendering so the CLI, critic and memory still receive a readable
``final_result``. The interactive Streamlit dashboard re-renders from the same
configuration + the engine when filters change (no LLM call).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    from config import get_settings
    from dashboard import build_dashboard_config, compute_dashboard, profile_dataframe
    from graph.state import AgentState
    from tools.sql_tool import fetch_postgres_schema, query_postgres
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.config import get_settings
    from project.dashboard import build_dashboard_config, compute_dashboard, profile_dataframe
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

def _render_config_summary(config: dict[str, Any], runtime: dict[str, Any]) -> str:
    """Render a readable text summary of the config + runtime for CLI/critic."""
    lines = [
        f"Dashboard: {config.get('title', 'Executive Dashboard')}",
        f"Scope: {config.get('subtitle', '')}".strip(),
        "",
        "KPIs:",
    ]
    for kpi in runtime.get("kpis", []):
        delta = kpi.get("delta")
        suffix = f" ({delta})" if delta and str(delta).strip() not in {"False", ""} else ""
        lines.append(f"- {kpi.get('label')}: {kpi.get('value')}{suffix}")

    charts = runtime.get("charts", [])
    if charts:
        lines.append("")
        lines.append("Charts:")
        for index, chart in enumerate(charts, start=1):
            rows = len(chart.get("data", []))
            lines.append(
                f"{index}. {chart.get('title')} ({chart.get('chart_type')}, {rows} data points)"
            )

    filters = config.get("filters", [])
    if filters:
        lines.append("")
        lines.append("Filters:")
        for filt in filters:
            lines.append(f"- {filt.get('label')} ({filt.get('type')})")
    return "\n".join(lines)


def dashboard_node(state: AgentState) -> AgentState:
    """Build the AI-planned dashboard configuration and default rendering."""
    query = str(state.get("query", "")).strip()
    retry_count = state.get("retry_count", 0)
    retry_count = retry_count if isinstance(retry_count, int) else 0

    base_result: dict[str, Any] = {
        **state,
        "last_execution_node": "dashboard",
        "dashboard": True,
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
            "dashboard_config": None,
            "dashboard_spec": None,
            "final_result": f"Dashboard could not be built: {reason}",
            "intermediate_results": [
                {
                    "step": "Plan dashboard configuration",
                    "tool": "visualization",
                    "action": "dashboard",
                    "result": {"type": "dashboard", "status": "unavailable", "reason": reason},
                }
            ],
        }

    profile = profile_dataframe(dataframe)
    config = build_dashboard_config(
        dataframe, profile, query=query, feedback=str(state.get("insights", ""))
    )
    config["data_source"] = source_label
    config["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    runtime = compute_dashboard(config, dataframe)
    final_result = _render_config_summary(config, runtime)

    return {
        **base_result,
        "dashboard_config": config,
        "dashboard_spec": runtime,
        "final_result": final_result,
        "plan": [
            {"step": "Plan dashboard configuration", "tool": "visualization", "action": "dashboard"}
        ],
        "intermediate_results": [
            {
                "step": "Plan dashboard configuration",
                "tool": "visualization",
                "action": "dashboard",
                "result": {
                    "type": "dashboard",
                    "status": "built",
                    "title": config.get("title", ""),
                    "kpi_count": len(config.get("kpis", [])),
                    "chart_count": len(config.get("charts", [])),
                    "filters": [f.get("id") for f in config.get("filters", [])],
                },
            }
        ],
    }


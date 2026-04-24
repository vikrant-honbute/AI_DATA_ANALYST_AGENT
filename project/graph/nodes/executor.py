"""Executor node for deterministic plan step execution."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from graph.state import AgentState
from tools import generate_plot, query_postgres, run_pandas_code


def _normalize_plan_steps(raw_plan: Any) -> list[dict[str, str]]:
    """Convert untyped plan input into a normalized step list."""
    if not isinstance(raw_plan, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue

        tool = item.get("tool")
        action = item.get("action")
        step_desc = item.get("step")

        if not isinstance(tool, str):
            continue

        normalized.append(
            {
                "step": step_desc.strip() if isinstance(step_desc, str) else "",
                "tool": tool.strip().lower(),
                "action": action.strip() if isinstance(action, str) else "",
            }
        )

    return normalized


def _result_to_text(result: Any) -> str:
    """Render a deterministic text representation for final output."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "Empty DataFrame"
        return result.to_string(index=False)

    if isinstance(result, pd.Series):
        return result.to_string()

    if isinstance(result, (dict, list, tuple)):
        return json.dumps(result, default=str, ensure_ascii=True)

    return str(result)


def _build_final_result(step_results: list[dict[str, Any]]) -> str:
    """Combine all step outputs into one final string."""
    if not step_results:
        return "No steps were executed."

    lines: list[str] = []
    for index, item in enumerate(step_results, start=1):
        lines.append(f"Step {index}: {item.get('step', 'Unnamed step')}")
        lines.append(f"Tool: {item.get('tool', 'unknown')}")

        if "error" in item:
            lines.append(f"Error: {item['error']}")
        else:
            lines.append(_result_to_text(item.get("result")))

        lines.append("")

    return "\n".join(lines).strip()


def _initial_dataframe_from_state(state: AgentState) -> pd.DataFrame:
    """Return the starting DataFrame from state when available."""
    raw_df = state.get("uploaded_dataframe")
    if isinstance(raw_df, pd.DataFrame):
        return raw_df.copy(deep=True)
    return pd.DataFrame()


def executor_node(state: AgentState) -> AgentState:
    """Execute planner steps in order without using an LLM."""
    plan_steps = _normalize_plan_steps(state.get("plan", []))

    if not plan_steps:
        return {
            **state,
            "intermediate_results": [],
            "final_result": "No valid plan steps were provided.",
        }

    working_df = _initial_dataframe_from_state(state)
    step_results: list[dict[str, Any]] = []

    for index, step in enumerate(plan_steps, start=1):
        tool = step["tool"]
        action = step["action"]
        step_label = step["step"] or f"step_{index}"

        try:
            if tool == "sql":
                sql_query = action if action else "SELECT 1 AS ok"
                sql_result = query_postgres(sql_query)
                working_df = sql_result.copy(deep=True)
                result: Any = sql_result
            elif tool == "pandas":
                pandas_code = action if action else "result = df"
                pandas_result = run_pandas_code(pandas_code, working_df)
                if isinstance(pandas_result, pd.DataFrame):
                    working_df = pandas_result.copy(deep=True)
                elif isinstance(pandas_result, pd.Series):
                    working_df = pandas_result.to_frame()
                result = pandas_result
            elif tool == "visualization":
                result = generate_plot(working_df, action, step_index=index)
            else:
                raise ValueError(f"Unsupported tool '{tool}'.")

            step_results.append(
                {
                    "step": step_label,
                    "tool": tool,
                    "action": action,
                    "result": result,
                }
            )
        except Exception as exc:
            step_results.append(
                {
                    "step": step_label,
                    "tool": tool,
                    "action": action,
                    "error": str(exc),
                }
            )

    return {
        **state,
        "intermediate_results": step_results,
        "final_result": _build_final_result(step_results),
    }

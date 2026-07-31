"""Executor node for deterministic plan step execution."""

from __future__ import annotations

import json
import re
from uuid import uuid4
from typing import Any

import pandas as pd

from graph.state import AgentState
from tools import generate_plot, query_postgres, run_pandas_code


_PREVIEW_ROW_LIMIT = 8
_TEXT_CHAR_LIMIT = 1200


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
        preview = result.head(_PREVIEW_ROW_LIMIT).to_string(index=False)
        remaining = max(len(result) - _PREVIEW_ROW_LIMIT, 0)
        remaining_text = f"\n... (+{remaining} more rows)" if remaining else ""
        column_list = ", ".join(str(col) for col in result.columns)
        return (
            f"DataFrame[{len(result)} rows x {len(result.columns)} columns]\n"
            f"Columns: {column_list}\n"
            f"Preview:\n{preview}{remaining_text}"
        )

    if isinstance(result, pd.Series):
        preview = result.head(_PREVIEW_ROW_LIMIT).to_string()
        remaining = max(len(result) - _PREVIEW_ROW_LIMIT, 0)
        remaining_text = f"\n... (+{remaining} more items)" if remaining else ""
        return f"Series[{len(result)} items]\n{preview}{remaining_text}"

    if isinstance(result, (dict, list, tuple)):
        serialized = json.dumps(result, default=str, ensure_ascii=True)
        if len(serialized) <= _TEXT_CHAR_LIMIT:
            return serialized
        return serialized[:_TEXT_CHAR_LIMIT].rstrip() + "..."

    text_value = str(result)
    if len(text_value) <= _TEXT_CHAR_LIMIT:
        return text_value
    return text_value[:_TEXT_CHAR_LIMIT].rstrip() + "..."


def _build_final_result(step_results: list[dict[str, Any]]) -> str:
    """Combine all step outputs into one final string."""
    if not step_results:
        return "No steps were executed."

    lines: list[str] = []
    previous_payload = ""
    for index, item in enumerate(step_results, start=1):
        lines.append(f"Step {index}: {item.get('step', 'Unnamed step')}")
        lines.append(f"Tool: {item.get('tool', 'unknown')}")

        if "error" in item:
            lines.append(f"Error: {item['error']}")
        else:
            payload = _result_to_text(item.get("result"))
            if payload == previous_payload:
                lines.append("Result: Same as previous step output.")
            else:
                lines.append(payload)
            previous_payload = payload

        lines.append("")

    return "\n".join(lines).strip()


def _initial_dataframe_from_state(state: AgentState) -> pd.DataFrame:
    """Return the starting DataFrame from state when available."""
    raw_df = state.get("uploaded_dataframe")
    if isinstance(raw_df, pd.DataFrame):
        return raw_df.copy(deep=True)
    return pd.DataFrame()


def _extract_referenced_columns(action: str) -> list[str]:
    """Extract column names referenced as df['column'] in pandas actions."""
    if not action:
        return []
    return re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", action)


def _select_execution_dataframe(
    action: str,
    working_df: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:
    """Choose working data, falling back to source data for missing referenced columns."""
    if source_df.empty:
        return working_df

    referenced = _extract_referenced_columns(action)
    if not referenced:
        return working_df

    working_columns = {str(col).lower() for col in working_df.columns}
    source_columns = {str(col).lower() for col in source_df.columns}

    missing_in_working = [
        column for column in referenced if column.lower() not in working_columns
    ]
    if not missing_in_working:
        return working_df

    if all(column.lower() in source_columns for column in missing_in_working):
        return source_df

    return working_df


def _series_to_step_dataframe(series: pd.Series) -> pd.DataFrame:
    """Convert a pandas Series into a one-row DataFrame for safe step chaining."""
    one_row = series.to_frame().T
    one_row.columns = [str(col) for col in one_row.columns]
    return one_row.reset_index(drop=True)


def executor_node(state: AgentState) -> AgentState:
    """Execute planner steps in order without using an LLM."""
    plan_steps = _normalize_plan_steps(state.get("plan", []))
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = uuid4().hex

    if not plan_steps:
        return {
            **state,
            "run_id": run_id,
            "intermediate_results": [],
            "final_result": "No valid plan steps were provided.",
        }

    source_df = _initial_dataframe_from_state(state)
    working_df = source_df.copy(deep=True)
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
                source_df = sql_result.copy(deep=True)
                result: Any = sql_result
            elif tool == "pandas":
                pandas_code = action if action else "result = df"
                execution_df = _select_execution_dataframe(action, working_df, source_df)
                pandas_result = run_pandas_code(pandas_code, execution_df)
                if isinstance(pandas_result, pd.DataFrame):
                    working_df = pandas_result.copy(deep=True)
                elif isinstance(pandas_result, pd.Series):
                    working_df = _series_to_step_dataframe(pandas_result)
                result = pandas_result
            elif tool == "visualization":
                result = generate_plot(working_df, action, step_index=index, run_id=run_id)
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
        "run_id": run_id,
        "intermediate_results": step_results,
        "final_result": _build_final_result(step_results),
    }

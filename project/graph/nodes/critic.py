"""Critic node for validating executor output and producing actionable fixes."""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from graph.state import AgentState
from llm import get_llm
from prompts import render_prompt


MAX_RETRIES = 2

_FILE_LOAD_MARKERS = (
    "read_csv(",
    "read_excel(",
    "read_parquet(",
    "read_json(",
    "read_table(",
    "read_pickle(",
)


class CriticStepModel(BaseModel):
    """Structured model for one corrected plan step."""

    step: str = Field(..., description="Short step description.")
    tool: Literal["sql", "pandas", "visualization"]
    action: str = Field(..., description="Concrete deterministic action.")


class CriticOutputModel(BaseModel):
    """Structured model for critic JSON output."""

    has_issue: bool = False
    issues: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    corrected_steps: list[CriticStepModel] = Field(default_factory=list)


def _normalize_plan_steps(raw_plan: Any) -> list[dict[str, str]]:
    """Normalize plan into a list of typed steps."""
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
        tool_value = tool.strip().lower()
        if tool_value not in {"sql", "pandas", "visualization"}:
            continue

        normalized.append(
            {
                "step": step_desc.strip() if isinstance(step_desc, str) else "",
                "tool": tool_value,
                "action": action.strip() if isinstance(action, str) else "",
            }
        )

    return normalized


def _build_critic_prompt(
    plan_steps: list[dict[str, str]],
    intermediate_results: list[Any],
    data_source: str,
    csv_columns: list[str],
    format_instructions: str,
) -> str:
    """Build strict JSON prompt for issue detection and plan correction."""
    serialized_plan = json.dumps(plan_steps, default=str, ensure_ascii=True)
    serialized_results = json.dumps(intermediate_results, default=str, ensure_ascii=True)

    return render_prompt(
        "critic_prompt.txt",
        data_source=data_source,
        csv_columns=json.dumps(csv_columns, ensure_ascii=True),
        serialized_plan=serialized_plan,
        serialized_results=serialized_results,
        format_instructions=format_instructions,
    )


def _extract_text(content: Any) -> str:
    """Normalize LangChain response content into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)

    return str(content)


def _has_execution_error(intermediate_results: list[Any]) -> bool:
    """Check whether any executor step produced an error entry."""
    for item in intermediate_results:
        if isinstance(item, dict) and isinstance(item.get("error"), str):
            return True
    return False


def _with_sql_limit(sql_text: str, limit: int = 200) -> str:
    """Add a LIMIT clause when missing, preserving existing SQL intent."""
    cleaned = sql_text.strip().rstrip(";")
    if not cleaned:
        return "SELECT table_name FROM information_schema.tables LIMIT 20"

    if " limit " in cleaned.lower():
        return cleaned

    return f"{cleaned} LIMIT {limit}"


def _extract_csv_columns_from_state(state: AgentState) -> list[str]:
    """Read uploaded CSV columns from state when present."""
    raw_df = state.get("uploaded_dataframe")
    columns = getattr(raw_df, "columns", None)
    if columns is None:
        return []

    try:
        return [str(column) for column in list(columns)]
    except Exception:
        return []


def _normalize_column_token(column_name: str) -> str:
    """Normalize column tokens for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def _closest_csv_column(target: str, csv_columns: list[str]) -> str | None:
    """Find the nearest available CSV column for a requested name."""
    normalized_target = _normalize_column_token(target)
    if not normalized_target:
        return None

    normalized_pairs: list[tuple[str, str]] = []
    for column in csv_columns:
        normalized = _normalize_column_token(column)
        if normalized:
            normalized_pairs.append((normalized, column))

    if not normalized_pairs:
        return None

    for normalized, column in normalized_pairs:
        if normalized == normalized_target:
            return column

    for normalized, column in normalized_pairs:
        if normalized_target in normalized or normalized in normalized_target:
            return column

    candidates = [normalized for normalized, _ in normalized_pairs]
    match = difflib.get_close_matches(normalized_target, candidates, n=1, cutoff=0.75)
    if not match:
        return None

    matched_token = match[0]
    for normalized, column in normalized_pairs:
        if normalized == matched_token:
            return column

    return None


def _extract_referenced_columns(action: str) -> list[str]:
    """Extract df['column'] references from pandas action text."""
    return re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", action)


def _has_unknown_column_reference(action: str, csv_columns: list[str]) -> bool:
    """Return True if action references columns not present in uploaded CSV."""
    if not csv_columns:
        return False

    known = {column.lower() for column in csv_columns}
    for referenced in _extract_referenced_columns(action):
        if referenced.lower() not in known:
            return True

    return False


def _rewrite_csv_column_references(action: str, csv_columns: list[str]) -> str:
    """Rewrite unknown df['col'] references to closest available CSV columns."""
    if not csv_columns:
        return action

    pattern = re.compile(r"df\[\s*(['\"])([^'\"]+)\1\s*\]")

    def _replace(match: re.Match[str]) -> str:
        quote = match.group(1)
        raw_column = match.group(2)
        replacement = _closest_csv_column(raw_column, csv_columns)
        if replacement is None:
            return match.group(0)
        return f"df[{quote}{replacement}{quote}]"

    return pattern.sub(_replace, action)


def _contains_file_load_action(action: str) -> bool:
    """Return True if pandas action tries to read files directly."""
    lowered = action.lower()
    return any(marker in lowered for marker in _FILE_LOAD_MARKERS)


def _find_metric_column(csv_columns: list[str], keywords: list[str]) -> str | None:
    """Find the first matching metric column by keyword."""
    lowered_map = {column.lower(): column for column in csv_columns}
    for keyword in keywords:
        for lowered, original in lowered_map.items():
            if keyword in lowered:
                return original
    return None


def _fallback_csv_retry_action(query: str, csv_columns: list[str]) -> str:
    """Build a deterministic pandas retry action for CSV workflows."""
    lowered = query.lower()

    if any(token in lowered for token in ["total", "sum"]):
        metric_col = _find_metric_column(
            csv_columns,
            ["total_revenue", "revenue", "sales", "amount", "income", "total"],
        )
        if metric_col is not None:
            return (
                "result = pd.DataFrame([{'metric': 'total', "
                f"'column': {metric_col!r}, 'value': float(df[{metric_col!r}].sum())}}])"
            )
        return (
            "result = pd.DataFrame([{'metric': 'total_numeric_sum', "
            "'value': float(df.select_dtypes(include='number').sum().sum())}])"
        )

    if any(token in lowered for token in ["average", "avg", "mean"]):
        return (
            "result = df.select_dtypes(include='number').mean().reset_index()"
            ".rename(columns={'index': 'metric', 0: 'value'})"
        )

    return "result = df.head(20)"


def _sanitize_corrected_plan(
    corrected_plan: list[dict[str, str]],
    state: AgentState,
) -> list[dict[str, str]]:
    """Normalize corrected steps to remain executable and CSV-safe."""
    query = str(state.get("query", ""))
    data_source_value = state.get("data_source", "csv")
    data_source = data_source_value if data_source_value in {"csv", "postgres", "mongo"} else "csv"
    csv_columns = _extract_csv_columns_from_state(state)

    sanitized: list[dict[str, str]] = []
    for index, step in enumerate(corrected_plan, start=1):
        tool_raw = step.get("tool", "pandas")
        action_raw = step.get("action", "")
        step_raw = step.get("step", "")

        tool = tool_raw.strip().lower() if isinstance(tool_raw, str) else "pandas"
        action = action_raw.strip() if isinstance(action_raw, str) else ""
        step_text = step_raw.strip() if isinstance(step_raw, str) else ""

        if tool not in {"sql", "pandas", "visualization"}:
            tool = "pandas"

        if not step_text:
            step_text = f"Corrected step {index}"

        if data_source in {"csv", "mongo"} and tool == "sql":
            tool = "pandas"
            action = _fallback_csv_retry_action(query, csv_columns)

        if tool == "pandas":
            if data_source == "csv":
                if _contains_file_load_action(action):
                    action = _fallback_csv_retry_action(query, csv_columns)
                else:
                    action = _rewrite_csv_column_references(action, csv_columns)

                if _has_unknown_column_reference(action, csv_columns):
                    action = _fallback_csv_retry_action(query, csv_columns)

                if "result" not in action:
                    action = _fallback_csv_retry_action(query, csv_columns)
            elif "result" not in action:
                action = "result = df.head(20)"

        if tool == "visualization" and not action:
            action = "Create a bar chart using the first numeric column"

        normalized_step = {
            "step": step_text,
            "tool": tool,
            "action": action,
        }

        if sanitized and normalized_step == sanitized[-1]:
            continue

        sanitized.append(normalized_step)

    if sanitized:
        return sanitized

    if data_source == "postgres":
        return [
            {
                "step": "Retry SQL with safe row limit",
                "tool": "sql",
                "action": "SELECT table_name FROM information_schema.tables LIMIT 20",
            }
        ]

    return [
        {
            "step": "Retry pandas step with explicit result output",
            "tool": "pandas",
            "action": _fallback_csv_retry_action(query, csv_columns),
        }
    ]


def _build_fallback_corrected_plan(
    plan_steps: list[dict[str, str]],
    intermediate_results: list[Any],
) -> list[dict[str, str]]:
    """Build deterministic correction plan when parsing fails."""
    corrected = [
        {
            "step": step.get("step", ""),
            "tool": step.get("tool", "pandas"),
            "action": step.get("action", ""),
        }
        for step in plan_steps
    ]

    error_indices: list[int] = []
    for index, item in enumerate(intermediate_results):
        if isinstance(item, dict) and isinstance(item.get("error"), str):
            error_indices.append(index)

    for index in error_indices:
        if index >= len(corrected):
            continue

        tool = corrected[index]["tool"]
        original_action = corrected[index]["action"]

        if tool == "sql":
            corrected[index]["step"] = "Retry SQL with safe row limit"
            corrected[index]["action"] = _with_sql_limit(original_action)
        elif tool == "pandas":
            corrected[index]["step"] = "Retry pandas step with explicit result output"
            corrected[index]["action"] = (
                original_action if "result" in original_action else "result = df.head(20)"
            )
        else:
            corrected[index]["step"] = "Retry visualization using first numeric column"
            corrected[index]["action"] = (
                original_action
                if original_action
                else "Create a bar chart using the first numeric column"
            )

    if corrected:
        return corrected

    return [
        {
            "step": "Produce deterministic fallback summary",
            "tool": "pandas",
            "action": "result = {'summary': 'Fallback plan after critic validation'}",
        }
    ]


def _format_actionable_insights(issues: list[str], fixes: list[str]) -> str:
    """Render concise issue and remediation text for state insights."""
    issue_text = "; ".join(issues) if issues else "Execution issues detected."
    fix_text = "; ".join(fixes) if fixes else "Apply corrected plan and retry."
    return f"Issues: {issue_text}. Fix: {fix_text}."


def critic_node(state: AgentState) -> AgentState:
    """Validate outputs, generate corrected plan, and manage bounded retries."""
    raw_plan = state.get("plan", [])
    plan_steps = _normalize_plan_steps(raw_plan)

    raw_results = state.get("intermediate_results", [])
    intermediate_results = raw_results if isinstance(raw_results, list) else [raw_results]
    data_source_value = state.get("data_source", "csv")
    data_source = data_source_value if data_source_value in {"csv", "postgres", "mongo"} else "csv"
    csv_columns = _extract_csv_columns_from_state(state)

    parser = PydanticOutputParser(pydantic_object=CriticOutputModel)
    prompt = _build_critic_prompt(
        plan_steps,
        intermediate_results,
        data_source,
        csv_columns,
        parser.get_format_instructions(),
    )

    parsed: CriticOutputModel

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_text = _extract_text(response.content)
        parsed = parser.parse(raw_text)
    except Exception:
        fallback_has_issue = _has_execution_error(intermediate_results)
        fallback_steps = _build_fallback_corrected_plan(plan_steps, intermediate_results)
        parsed = CriticOutputModel(
            has_issue=fallback_has_issue,
            issues=["Critic parsing failed; fallback correction applied."],
            fixes=["Use fallback corrected steps from failed tool outputs."],
            corrected_steps=[CriticStepModel(**item) for item in fallback_steps],
        )

    has_issue = bool(parsed.has_issue) or _has_execution_error(intermediate_results)
    issues = [item.strip() for item in parsed.issues if item.strip()]
    fixes = [item.strip() for item in parsed.fixes if item.strip()]

    corrected_plan = [
        {
            "step": item.step,
            "tool": item.tool,
            "action": item.action,
        }
        for item in parsed.corrected_steps
    ]
    if has_issue and not corrected_plan:
        corrected_plan = _build_fallback_corrected_plan(plan_steps, intermediate_results)
    if has_issue:
        corrected_plan = _sanitize_corrected_plan(corrected_plan, state)

    current_retry_count = state.get("retry_count", 0)
    retry_count = current_retry_count if isinstance(current_retry_count, int) else 0

    if has_issue:
        actionable_text = _format_actionable_insights(issues, fixes)
        if retry_count < MAX_RETRIES:
            next_retry_count = retry_count + 1
            return {
                **state,
                "plan": corrected_plan,
                "retry": True,
                "retry_count": next_retry_count,
                "insights": (
                    f"{actionable_text} "
                    f"Retrying with corrected plan ({next_retry_count}/{MAX_RETRIES})."
                ),
            }

        return {
            **state,
            "plan": corrected_plan,
            "retry": False,
            "retry_count": retry_count,
            "insights": (
                f"{actionable_text} Retry limit reached "
                f"({retry_count}/{MAX_RETRIES}); proceeding to insight node."
            ),
        }

    return {
        **state,
        "retry": False,
        "retry_count": retry_count,
        "insights": "Validation passed with no execution issues detected.",
    }

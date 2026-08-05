"""Planner node for generating execution steps and choosing a data source."""

from __future__ import annotations

import difflib
import json
import logging
import re

from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
import pandas as pd
from pydantic import BaseModel, Field

try:
    from graph.state import AgentState, DataSource
    from llm import get_llm
    from prompts import render_prompt
    from tools.memory_tool import get_recent_memory
    from tools.sql_tool import fetch_postgres_schema
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.graph.state import AgentState, DataSource
    from project.llm import get_llm
    from project.prompts import render_prompt
    from project.tools.memory_tool import get_recent_memory
    from project.tools.sql_tool import fetch_postgres_schema

logger = logging.getLogger(__name__)


class PlannerStepModel(BaseModel):
    """Structured model for one planning step."""

    step: str = Field(..., description="Short step description.")
    tool: Literal["sql", "pandas", "visualization"]
    action: str = Field(..., description="Concrete action to execute.")


class PlannerOutputModel(BaseModel):
    """Structured model for planner JSON output."""

    steps: list[PlannerStepModel] = Field(default_factory=list)
    data_source: DataSource = "csv"
    final_output: Literal["table", "chart", "summary"] = "summary"


_FILE_LOAD_MARKERS = (
    "read_csv(",
    "read_excel(",
    "read_parquet(",
    "read_json(",
    "read_table(",
    "read_pickle(",
)


def _build_router_prompt(query: str, has_uploaded_csv: bool) -> str:
    """Build a strict one-word data source routing prompt."""
    return render_prompt(
        "router_prompt.txt",
        query=query,
        has_uploaded_csv=str(has_uploaded_csv).lower(),
    )


def _build_planner_prompt(
    query: str,
    routed_data_source: DataSource,
    schema_text: str,
    csv_columns_text: str,
    memory_context_text: str,
    use_memory_context: bool,
    format_instructions: str,
) -> str:
    """Build a strict JSON planning prompt for the LLM."""
    return render_prompt(
        "planner_prompt.txt",
        query=query,
        routed_data_source=routed_data_source,
        schema_text=schema_text,
        csv_columns_text=csv_columns_text,
        memory_context_text=memory_context_text,
        use_memory_context=str(use_memory_context).lower(),
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


def _is_explicit_database_request(query: str) -> bool:
    """Return True when query explicitly asks for SQL/database backends."""
    return bool(
        re.search(
            r"\b(?:postgres|postgresql|database|db|sql|schema|database\s+table)\b",
            query,
            re.IGNORECASE,
        )
    )


def _heuristic_route(query: str, has_uploaded_csv: bool) -> DataSource:
    """Fallback deterministic route when LLM output is invalid."""
    lowered = query.lower()

    if any(keyword in lowered for keyword in ["history", "context", "memory", "previous", "past"]):
        return "mongo"

    if has_uploaded_csv and not _is_explicit_database_request(query):
        return "csv"

    if any(
        keyword in lowered
        for keyword in ["csv", "excel", "file", "upload", "spreadsheet", "dataset"]
    ):
        return "csv"

    if _is_explicit_database_request(query):
        return "postgres"

    return "postgres"


def _parse_routed_source(raw_text: str) -> DataSource | None:
    """Extract one valid source token from router output."""
    tokens = re.findall(r"[a-z]+", raw_text.lower())
    for token in tokens:
        if token in {"postgres", "csv", "mongo"}:
            return token
    return None


def _query_relates_to_past(query: str) -> bool:
    """Return True when a query references previous context or time periods."""
    lowered = query.lower()
    indicators = [
        "last month",
        "last week",
        "last quarter",
        "last year",
        "previous",
        "prior",
        "earlier",
        "before",
        "again",
        "history",
        "context",
        "memory",
        "past",
        "last time",
        "compare with",
        "compared to",
        "same as",
    ]
    return any(token in lowered for token in indicators)


def _normalize_memory_item(item: dict[str, Any], result_char_limit: int = 300) -> dict[str, str]:
    """Normalize one memory record for prompt-safe context reuse."""
    query_text = str(item.get("query", "")).strip()
    result_obj = item.get("result", "")

    try:
        result_text = json.dumps(result_obj, ensure_ascii=True, default=str)
    except Exception as exc:
        logger.warning(
            "planner[_normalize_memory_item] failed to JSON-serialize a memory result; "
            "falling back to str(): %s",
            exc,
        )
        result_text = str(result_obj)

    if len(result_text) > result_char_limit:
        result_text = result_text[:result_char_limit].rstrip() + "..."

    return {
        "query": query_text,
        "result": result_text,
    }


def _fetch_recent_memory(limit: int = 5, session_id: str = "") -> list[dict[str, Any]]:
    """Fetch recent memory records from MongoDB with safe fallback."""
    if not session_id:
        return []
    try:
        return get_recent_memory(session_id=session_id, limit=limit)
    except Exception as exc:
        logger.warning(
            "planner[_fetch_recent_memory] failed to fetch recent memory; "
            "proceeding without memory context: %s",
            exc,
        )
        return []


def _select_relevant_memory(query: str, recent_memory: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Select memory records to reuse when query refers to past context."""
    if not recent_memory or not _query_relates_to_past(query):
        return []

    normalized = [_normalize_memory_item(item) for item in recent_memory]
    return normalized[:3]


def _build_memory_context_text(memory_context: list[dict[str, str]]) -> str:
    """Serialize selected memory context for planner prompt grounding."""
    if not memory_context:
        return "No relevant memory context."
    return json.dumps(memory_context, ensure_ascii=True, indent=2)


def _plan_has_memory_context(plan: list[dict[str, Any]]) -> bool:
    """Check whether plan already contains a memory-context step."""
    for step in plan:
        step_text = str(step.get("step", "")).lower()
        action_text = str(step.get("action", "")).lower()
        if any(token in f"{step_text} {action_text}" for token in ["memory", "previous", "past"]):
            return True
    return False


def _prepend_memory_step(
    plan: list[dict[str, Any]], memory_context: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Ensure relevant, scoped memory is displayed through a trusted operation."""
    if not memory_context or _plan_has_memory_context(plan):
        return plan
    return [
        {
            "step": "Load relevant historical context",
            "tool": "pandas",
            "action": json.dumps({"operation": "memory_records", "limit": 3}),
        },
        *plan,
    ]


def route_data_source(query: str, has_uploaded_csv: bool = False) -> DataSource:
    """Route a user query to one data source using LLM output."""
    prompt = _build_router_prompt(query, has_uploaded_csv)

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_text = _extract_text(response.content)
        parsed = _parse_routed_source(raw_text)
        if parsed is not None:
            if parsed == "postgres" and has_uploaded_csv and not _is_explicit_database_request(query):
                return "csv"
            return parsed
    except Exception:
        pass

    return _heuristic_route(query, has_uploaded_csv)


def _get_schema_text_for_prompt(data_source: DataSource) -> str:
    """Fetch and serialize schema context for prompt grounding."""
    if data_source != "postgres":
        return "Schema not required for selected data source."

    try:
        schema_map = fetch_postgres_schema()
    except Exception as exc:
        return f"Schema unavailable: {exc}"

    if not schema_map:
        return "No PostgreSQL tables found."

    return json.dumps(schema_map, ensure_ascii=True, indent=2, sort_keys=True)


def _extract_csv_columns(uploaded_dataframe: Any) -> list[str]:
    """Extract uploaded CSV columns from state when available."""
    if isinstance(uploaded_dataframe, pd.DataFrame):
        return [str(col) for col in uploaded_dataframe.columns]
    return []


def _build_csv_columns_text(csv_columns: list[str]) -> str:
    """Serialize CSV column names for prompt grounding."""
    if not csv_columns:
        return "No uploaded CSV columns available."
    return json.dumps(csv_columns, ensure_ascii=True)


def _find_metric_column(csv_columns: list[str], keywords: list[str]) -> str | None:
    """Find the best matching column using keyword heuristics."""
    lowered_map = {col.lower(): col for col in csv_columns}

    for preferred in keywords:
        for lowered, original in lowered_map.items():
            if preferred in lowered:
                return original

    return None


def _fallback_csv_action(query: str, csv_columns: list[str]) -> str:
    """Build deterministic pandas action for CSV-focused fallback."""
    lowered = query.lower()

    if any(token in lowered for token in ["total", "sum"]):
        metric_col = _find_metric_column(
            csv_columns,
            ["total_revenue", "revenue", "sales", "amount", "income", "total"],
        )
        if metric_col is not None:
            return json.dumps(
                {"operation": "aggregate", "column": metric_col, "function": "sum"}
            )
        return json.dumps({"operation": "head", "limit": 20})

    if any(token in lowered for token in ["average", "avg", "mean"]):
        metric_col = _find_metric_column(
            csv_columns, ["revenue", "sales", "amount", "income", "price", "quantity"]
        )
        if metric_col is not None:
            return json.dumps(
                {"operation": "aggregate", "column": metric_col, "function": "mean"}
            )

    return json.dumps({"operation": "head", "limit": 20})


def _contains_file_load_action(action: str) -> bool:
    """Return True if pandas action attempts to load files directly."""
    lowered = action.lower()
    return any(marker in lowered for marker in _FILE_LOAD_MARKERS)


def _normalize_column_token(column_name: str) -> str:
    """Normalize column tokens for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def _closest_csv_column(target: str, csv_columns: list[str]) -> str | None:
    """Find the closest available CSV column for a requested name."""
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
    """Return True if action references columns missing from uploaded CSV."""
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


def _sanitize_csv_pandas_action(action: str, query: str, csv_columns: list[str]) -> str:
    """Accept only declarative operations referencing known CSV columns."""
    try:
        spec = json.loads(action)
    except (TypeError, json.JSONDecodeError):
        return _fallback_csv_action(query, csv_columns)
    if not isinstance(spec, dict) or not isinstance(spec.get("operation"), str):
        return _fallback_csv_action(query, csv_columns)
    known = {column.lower() for column in csv_columns}
    requested: list[str] = []
    for key in ("column",):
        if spec.get(key) is not None:
            requested.append(str(spec[key]))
    for key in ("columns", "by"):
        if isinstance(spec.get(key), list):
            requested.extend(str(item) for item in spec[key])
    if any(column.lower() not in known for column in requested):
        return _fallback_csv_action(query, csv_columns)
    return json.dumps(spec, ensure_ascii=True)


def _fallback_output(
    query: str,
    routed_data_source: DataSource,
    csv_columns: list[str],
) -> PlannerOutputModel:
    """Return a safe default planner output when parsing fails."""
    if routed_data_source == "postgres":
        step = PlannerStepModel(
            step="Fetch structured data",
            tool="sql",
            action="SELECT 1 AS sample_value",
        )
    elif routed_data_source == "csv":
        step = PlannerStepModel(
            step="Compute requested metric from uploaded CSV",
            tool="pandas",
            action=_fallback_csv_action(query, csv_columns),
        )
    else:
        step = PlannerStepModel(
            step="Summarize memory context",
            tool="pandas",
            action=json.dumps({"operation": "memory_records", "limit": 5}),
        )

    return PlannerOutputModel(
        steps=[step],
        data_source=routed_data_source,
        final_output="summary",
    )


def _sanitize_plan_for_data_source(
    plan: list[dict[str, Any]],
    data_source: DataSource,
    query: str,
    csv_columns: list[str],
) -> list[dict[str, Any]]:
    """Normalize planner output so tool choices match selected data source."""
    sanitized: list[dict[str, Any]] = []

    for step in plan:
        if not isinstance(step, dict):
            continue

        tool = str(step.get("tool", "")).strip().lower()
        action = str(step.get("action", "")).strip()
        step_text = str(step.get("step", "")).strip() or "Execute analysis step"

        if tool not in {"sql", "pandas", "visualization"}:
            tool = "pandas"

        if data_source in {"csv", "mongo"} and tool == "sql":
            tool = "pandas"
            action = _fallback_csv_action(query, csv_columns)

        if tool == "pandas":
            if data_source == "csv":
                action = _sanitize_csv_pandas_action(action, query, csv_columns)
            else:
                try:
                    parsed_action = json.loads(action)
                    if not isinstance(parsed_action, dict) or "operation" not in parsed_action:
                        raise ValueError
                    action = json.dumps(parsed_action, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    action = json.dumps({"operation": "memory_records", "limit": 5})

        if tool == "visualization" and not action:
            action = "Create a line chart using the primary numeric column"

        sanitized.append(
            {
                "step": step_text,
                "tool": tool,
                "action": action,
            }
        )

    return sanitized


def _ensure_required_steps(
    plan: list[dict[str, Any]],
    data_source: DataSource,
    query: str,
    csv_columns: list[str],
) -> list[dict[str, Any]]:
    """Guarantee at least one executable step for each routed source."""
    if data_source == "csv" and not any(step.get("tool") == "pandas" for step in plan):
        return [
            {
                "step": "Compute requested metric from uploaded CSV",
                "tool": "pandas",
                "action": _fallback_csv_action(query, csv_columns),
            },
            *plan,
        ]

    if data_source == "postgres" and not any(step.get("tool") == "sql" for step in plan):
        return [
            {
                "step": "Fetch structured data",
                "tool": "sql",
                "action": "SELECT 1 AS sample_value",
            },
            *plan,
        ]

    if data_source == "mongo" and not any(step.get("tool") == "pandas" for step in plan):
        return [
            {
                "step": "Summarize memory context",
                "tool": "pandas",
                "action": json.dumps({"operation": "memory_records", "limit": 5}),
            },
            *plan,
        ]

    return plan


def planner_node(state: AgentState) -> AgentState:
    """Return a new state with structured plan and selected data source."""
    query = state.get("query", "").strip()
    uploaded_dataframe = state.get("uploaded_dataframe")
    csv_columns = _extract_csv_columns(uploaded_dataframe)
    has_uploaded_csv = bool(csv_columns)

    routed_data_source = route_data_source(query, has_uploaded_csv=has_uploaded_csv)
    schema_text = _get_schema_text_for_prompt(routed_data_source)
    session_id = str(state.get("session_id", "")).strip()
    recent_memory = _fetch_recent_memory(limit=5, session_id=session_id)
    selected_memory = _select_relevant_memory(query, recent_memory)

    parser = PydanticOutputParser(pydantic_object=PlannerOutputModel)
    prompt = _build_planner_prompt(
        query,
        routed_data_source,
        schema_text,
        _build_csv_columns_text(csv_columns),
        _build_memory_context_text(selected_memory),
        bool(selected_memory),
        parser.get_format_instructions(),
    )

    parsed: PlannerOutputModel

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        raw_text = _extract_text(response.content)
        parsed = parser.parse(raw_text)
    except Exception:
        parsed = _fallback_output(query, routed_data_source, csv_columns)

    plan = [
        {
            "step": item.step,
            "tool": item.tool,
            "action": item.action,
        }
        for item in parsed.steps
    ]

    if not plan:
        fallback = _fallback_output(query, routed_data_source, csv_columns)
        plan = [
            {
                "step": item.step,
                "tool": item.tool,
                "action": item.action,
            }
            for item in fallback.steps
        ]

    plan = _sanitize_plan_for_data_source(plan, routed_data_source, query, csv_columns)
    plan = _ensure_required_steps(plan, routed_data_source, query, csv_columns)
    plan = _prepend_memory_step(plan, selected_memory)

    return {
        **state,
        "plan": plan,
        "data_source": routed_data_source,
        "final_output": parsed.final_output,
        "memory": recent_memory,
    }

"""Insight node for extracting key insights and business meaning."""

from __future__ import annotations

import logging
from typing import Any

try:
    from graph.state import AgentState
    from graph.nodes.critic import MAX_RETRIES
    from llm import get_llm
    from prompts import render_prompt
    from tools.memory_tool import save_memory
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.graph.state import AgentState
    from project.graph.nodes.critic import MAX_RETRIES
    from project.llm import get_llm
    from project.prompts import render_prompt
    from project.tools.memory_tool import save_memory


_MEMORY_RESULT_CHAR_LIMIT = 1200
logger = logging.getLogger(__name__)


def _build_insight_prompt(final_result: str) -> str:
    """Build a prompt that asks for concise analytical insights."""
    return render_prompt("insight_prompt.txt", final_result=final_result)


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


def _compact_text(value: Any, char_limit: int = _MEMORY_RESULT_CHAR_LIMIT) -> str:
    """Render a compact string suitable for lightweight memory storage."""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= char_limit:
        return text
    return text[:char_limit].rstrip() + "..."


def _should_save_memory(state: AgentState, final_result: str) -> bool:
    """Return True when the completed run is safe to persist as memory."""
    if not final_result.strip():
        return False

    raw_retry_count = state.get("retry_count", 0)
    retry_count = raw_retry_count if isinstance(raw_retry_count, int) else 0

    raw_results = state.get("intermediate_results", [])
    has_error = isinstance(raw_results, list) and any(
        isinstance(step, dict) and "error" in step for step in raw_results
    )
    return not has_error and retry_count <= MAX_RETRIES


def insight_node(state: AgentState) -> AgentState:
    """Generate insights from final_result and return updated state."""
    raw_final_result = state.get("final_result", "")
    final_result = raw_final_result.strip() if isinstance(raw_final_result, str) else str(raw_final_result)

    if not final_result:
        return {
            **state,
            "insights": "No final_result available to generate insights.",
        }

    try:
        llm = get_llm()
        response = llm.invoke(_build_insight_prompt(final_result))
        insights = _extract_text(response.content).strip()
    except Exception as exc:
        insights = f"Insight generation error: {exc}"

    if _should_save_memory(state, final_result):
        session_id = str(state.get("session_id", "")).strip()
        try:
            if session_id:
                save_memory(
                    session_id,
                    state["query"],
                    {
                        "final_result": _compact_text(final_result),
                        "data_source": str(state.get("data_source", "unknown")),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist scoped memory: %s", exc)

    return {
        **state,
        "insights": insights or "No insights generated.",
    }

"""Insight node for extracting key insights and business meaning."""

from __future__ import annotations

from typing import Any

from graph.state import AgentState
from llm import get_llm
from prompts import render_prompt


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

    return {
        **state,
        "insights": insights or "No insights generated.",
    }
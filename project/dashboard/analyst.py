"""LLM analyst pass for dashboard narrative refinement.

The deterministic layout builder produces all numbers. This module optionally
asks the LLM to write the professional narrative (title, executive summary,
insight bullets, recommendations) grounded strictly in the computed facts.
Any failure falls back to the deterministic narrative already in the spec.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

try:
    from llm import get_llm
    from prompts import render_prompt
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.llm import get_llm
    from project.prompts import render_prompt

logger = logging.getLogger(__name__)

MAX_INSIGHTS = 6
MAX_RECOMMENDATIONS = 4


class AnalystNarrativeModel(BaseModel):
    """Structured model for the LLM analyst narrative output."""

    title: str = Field(..., description="Professional dashboard title (max 10 words).")
    subtitle: str = Field(..., description="One-line scope description of the dashboard.")
    executive_summary: str = Field(
        ..., description="2-3 sentence executive summary grounded in the provided facts."
    )
    insights: list[str] = Field(
        default_factory=list, description="3-5 insight bullets grounded in the provided facts."
    )
    recommendations: list[str] = Field(
        default_factory=list, description="2-3 actionable recommendation bullets."
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


def _sanitize_bullets(items: Any, limit: int) -> list[str]:
    """Coerce LLM bullets into bounded, non-empty strings."""
    if not isinstance(items, list):
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text[:300])
        if len(cleaned) >= limit:
            break
    return cleaned


def refine_spec_with_llm(
    spec: dict[str, Any],
    profile: dict[str, Any],
    query: str,
    feedback: str = "",
) -> dict[str, Any]:
    """Refine a deterministic spec with an LLM narrative; safe on any failure."""
    try:
        parser = PydanticOutputParser(pydantic_object=AnalystNarrativeModel)
        facts_payload = {
            "kpis": spec.get("kpis", []),
            "chart_titles": [chart.get("title", "") for chart in spec.get("charts", [])],
            "computed_facts": spec.get("insights", []),
            "row_count": spec.get("row_count"),
            "time_range": spec.get("time_range"),
        }
        prompt = render_prompt(
            "dashboard_prompt.txt",
            query=query or "Build a dashboard",
            profile_text=json.dumps(profile, ensure_ascii=True, default=str),
            facts_text=json.dumps(facts_payload, ensure_ascii=True, default=str),
            feedback_text=feedback.strip() or "None",
            format_instructions=parser.get_format_instructions(),
        )

        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = parser.parse(_extract_text(response.content))

        refined = dict(spec)
        title = str(parsed.title).strip()[:120]
        subtitle = str(parsed.subtitle).strip()[:240]
        summary = str(parsed.executive_summary).strip()[:1000]
        if title:
            refined["title"] = title
        if subtitle:
            refined["subtitle"] = subtitle
        if summary:
            refined["executive_summary"] = summary
        insights = _sanitize_bullets(parsed.insights, MAX_INSIGHTS)
        if insights:
            refined["insights"] = insights
        recommendations = _sanitize_bullets(parsed.recommendations, MAX_RECOMMENDATIONS)
        if recommendations:
            refined["recommendations"] = recommendations
        return refined
    except Exception as exc:
        logger.warning(
            "dashboard[refine_spec_with_llm] LLM refinement failed; keeping deterministic narrative: %s",
            exc,
        )
        return spec

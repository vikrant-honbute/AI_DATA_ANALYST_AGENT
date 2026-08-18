"""AI dashboard configuration planner.

This module is the single LLM entry point for the dashboard. It produces a
structured *configuration* (which KPIs, filters and charts) — never Python/UI
code — constrained to the columns and operations found by the deterministic
profiler. Every output passes through ``validate.validate_config`` and any LLM
failure falls back to the deterministic heuristic suggestions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser

try:
    from dashboard.chart_selector import (
        suggest_chart_configs,
        suggest_filter_configs,
        suggest_kpi_configs,
    )
    from dashboard.models import (
        MAX_CHARTS,
        MAX_FILTERS,
        MAX_KPIS,
        VALID_AGGREGATIONS,
        VALID_CHART_TYPES,
        VALID_FILTER_TYPES,
        VALID_INSIGHT_TOPICS,
        DashboardConfig,
    )
    from dashboard.profiler import DataProfile
    from dashboard.validate import validate_config
    from llm import get_llm
    from prompts import render_prompt
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.chart_selector import (
        suggest_chart_configs,
        suggest_filter_configs,
        suggest_kpi_configs,
    )
    from project.dashboard.models import (
        MAX_CHARTS,
        MAX_FILTERS,
        MAX_KPIS,
        VALID_AGGREGATIONS,
        VALID_CHART_TYPES,
        VALID_FILTER_TYPES,
        VALID_INSIGHT_TOPICS,
        DashboardConfig,
    )
    from project.dashboard.profiler import DataProfile
    from project.dashboard.validate import validate_config
    from project.llm import get_llm
    from project.prompts import render_prompt

logger = logging.getLogger(__name__)


def fallback_config(df: pd.DataFrame, profile: DataProfile) -> dict[str, Any]:
    """Deterministic fallback configuration used when the LLM is unavailable."""
    config: dict[str, Any] = {
        "title": (
            f"{profile.primary_metric.title() if profile.primary_metric else 'Data'} Performance Dashboard"
        ),
        "subtitle": f"{profile.rows:,} records · {len(profile.columns)} columns",
        "data_source": "csv",
        "generated_at": "",
        "row_count": profile.rows,
        "column_count": len(profile.columns),
        "primary_metric": profile.primary_metric,
        "time_dimension": profile.time_column,
        "kpis": suggest_kpi_configs(df, profile),
        "filters": suggest_filter_configs(df, profile),
        "charts": suggest_chart_configs(df, profile),
        "insight_topics": ["trend", "best_segment", "worst_segment", "profitability"],
    }
    return validate_config(config, df)


def _profile_text(profile: DataProfile) -> str:
    """Render a compact prompt-friendly profile description."""
    return json.dumps(profile.to_dict(), ensure_ascii=True, default=str)


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


def _safe_json_text(raw: str) -> str:
    """Strip markdown fences/extra prose so a JSON payload can be parsed."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text or text[0] not in "{[":
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                if _is_balanced(candidate, opener, closer):
                    text = candidate
                    break
    return text


def _is_balanced(text: str, opener: str, closer: str) -> bool:
    """Return True when braces/brackets in text are balanced."""
    depth = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def build_dashboard_config(
    df: pd.DataFrame,
    profile: DataProfile,
    query: str = "",
    feedback: str = "",
) -> dict[str, Any]:
    """Generate a validated dashboard configuration.

    Uses the LLM when available, then validates/repairs the result against the
    DataFrame. Any failure returns the deterministic fallback configuration.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=DashboardConfig)
        prompt = render_prompt(
            "dashboard_config_prompt.txt",
            query=query.strip() or "Build a professional dashboard for this dataset.",
            profile_text=_profile_text(profile),
            aggregations_text=", ".join(sorted(VALID_AGGREGATIONS)),
            chart_types_text=", ".join(sorted(VALID_CHART_TYPES)),
            filter_types_text=", ".join(sorted(VALID_FILTER_TYPES)),
            insight_topics_text=", ".join(sorted(VALID_INSIGHT_TOPICS)),
            limits_text=(
                f"at most {MAX_KPIS} KPIs, {MAX_FILTERS} filters and {MAX_CHARTS} charts"
            ),
            feedback_text=feedback.strip() or "None",
            format_instructions=parser.get_format_instructions(),
        )
        llm = get_llm()
        response = llm.invoke(prompt)
        parsed = parser.parse(_safe_json_text(_extract_text(response.content)))
        config = parsed.as_dict()
    except Exception as exc:
        logger.warning("dashboard[planner] LLM planning failed; using fallback config: %s", exc)
        config = fallback_config(df, profile)

    config["data_source"] = "csv"
    config["row_count"] = int(len(df))
    config["column_count"] = int(len(df.columns))
    if profile.time_start and profile.time_end:
        subtitle_bits = [str(config.get("subtitle") or "").strip()]
        subtitle_bits.append(f"{profile.time_start} to {profile.time_end}")
        config["subtitle"] = " · ".join(bit for bit in subtitle_bits if bit)

    return validate_config(config, df)

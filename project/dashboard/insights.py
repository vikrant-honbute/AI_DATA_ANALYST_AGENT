"""AI insight engine for the dashboard.

Produces insight bullets grounded in *computed* analytical facts (from
``engine.compute_insight_facts``), never from raw row inspection or
hallucination. A deterministic bullet set is always produced; an optional LLM
narration pass can rephrase/prioritize the same facts with a safe fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

try:
    from dashboard.context import build_filter_context_text
    from dashboard.formatting import format_compact, looks_like_money, prettify_name
    from llm import get_llm
    from prompts import render_prompt
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.dashboard.context import build_filter_context_text
    from project.dashboard.formatting import format_compact, looks_like_money, prettify_name
    from project.llm import get_llm
    from project.prompts import render_prompt

logger = logging.getLogger(__name__)


def _num(value: Any, fmt: str = "number") -> str:
    """Format a single fact number for display in an insight bullet."""
    if value is None:
        return "—"
    money = fmt == "money" or looks_like_money(str(fmt))
    if money:
        return f"${format_compact(value)}"
    return format_compact(value)


def build_dashboard_insights(
    facts: dict[str, Any], config: dict[str, Any] | None = None, df: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    """Build deterministic insight cards from computed facts.

    Returns a list of ``{"kind", "title", "body", "evidence"}`` records sorted
    by analytical importance.
    """
    insights: list[dict[str, Any]] = []
    primary = facts.get("primary_metric")
    primary_label = prettify_name(primary) if primary else "primary metric"

    trend = facts.get("trend")
    if trend and trend != "mixed":
        body = (
            f"{primary_label} shows a {trend} trajectory across "
            f"{int(facts.get('trend_periods') or 0)} time periods"
        )
        change = facts.get("trend_change_pct")
        if change is not None:
            direction = "up" if float(change) >= 0 else "down"
            body += f", with a {direction} of {abs(float(change)):.1f}% in the latest period"
        insights.append(
            {"kind": "trend", "title": "Key trend", "body": body + ".", "evidence": facts.get("trend")}
        )
    elif trend == "mixed":
        insights.append(
            {
                "kind": "trend",
                "title": "Key trend",
                "body": f"{primary_label} follows a mixed pattern over time — worth inspecting "
                        "for seasonality or outlier periods.",
                "evidence": "mixed",
            }
        )

    best = facts.get("best_segment")
    if best:
        share = facts.get("best_segment_share_pct")
        share_text = f" ({share:.1f}% of total)" if share is not None else ""
        insights.append(
            {
                "kind": "best_segment",
                "title": "Best-performing segment",
                "body": f"'{best.get('name')}' leads {facts.get('segment_dimension', 'segments')} "
                        f"at {_num(best.get('value'))}{share_text}.",
                "evidence": best,
            }
        )

    worst = facts.get("worst_segment")
    if worst:
        insights.append(
            {
                "kind": "worst_segment",
                "title": "Worst-performing segment",
                "body": f"'{worst.get('name')}' trails at {_num(worst.get('value'))} — "
                        "a candidate for deeper investigation.",
                "evidence": worst,
            }
        )

    margin = facts.get("margin_pct")
    if margin is not None:
        margin_columns = facts.get("margin_columns") or []
        column_label = " and ".join(prettify_name(c) for c in margin_columns)
        if float(margin) < 0:
            title = "Profitability issue"
            tone_body = (
                f"Margin between {column_label} is negative ({margin:.1f}%). The combination is "
                "loss-making on average."
            )
        else:
            title = "Profitability"
            tone_body = f"Margin between {column_label} stands at {margin:.1f}%."
        insights.append({"kind": "profitability", "title": title, "body": tone_body, "evidence": margin})

    if not insights:
        insights.append(
            {
                "kind": "opportunity",
                "title": "Opportunity",
                "body": f"Analyzed {int(facts.get('row_count') or 0)} matching records. Add a "
                        "time dimension and numeric measures to surface deeper trends.",
                "evidence": None,
            }
        )

    return insights

def _sanitize_bullets(items: Any, limit: int = 6) -> list[str]:
    """Coerce LLM bullets into bounded, non-empty strings."""
    if not isinstance(items, list):
        return []
    cleaned: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            cleaned.append(text[:400])
        if len(cleaned) >= limit:
            break
    return cleaned


def refine_insights_with_llm(
    facts: dict[str, Any],
    config: dict[str, Any],
    deterministic: list[dict[str, Any]],
    active_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Optionally have the LLM narrate the computed facts into professional bullets.

    Always falls back to the deterministic insights on any failure so the
    dashboard never loses its insight section.
    """
    try:
        facts_text = json.dumps(facts, ensure_ascii=True, default=str)
        context_text = build_filter_context_text(config, active_filters)
        prompt = render_prompt(
            "dashboard_insight_prompt.txt",
            facts_text=facts_text,
            context_text=context_text,
            title=str(config.get("title") or "Dashboard"),
        )
        llm = get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        bullets = _sanitize_bullets(_extract_list(raw))
        if bullets:
            return [
                {
                    "kind": f"llm_{index}",
                    "title": "AI insight",
                    "body": bullet,
                    "evidence": deterministic[index]["evidence"]
                    if index < len(deterministic)
                    else None,
                }
                for index, bullet in enumerate(bullets)
            ]
    except Exception as exc:  # pragma: no cover - graceful degradation.
        logger.warning("dashboard[insights] LLM narration failed; keeping deterministic: %s", exc)
    return deterministic


def _extract_list(raw: str) -> list[Any]:
    """Parse a JSON array (or markdown list) from LLM output into strings."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    lines = [line.strip().lstrip("-•*").strip() for line in raw.splitlines() if line.strip()]
    return [line for line in lines if line]


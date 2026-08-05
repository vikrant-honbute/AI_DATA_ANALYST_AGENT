"""CLI entry point for the AI Data Analysis Agent."""

from __future__ import annotations

import os
import sys
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from graph import build_workflow
    from graph.state import AgentState
    from config import get_settings
    from tools.pandas_tool import cleanup_plot_files
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.graph import build_workflow
    from project.graph.state import AgentState
    from project.config import get_settings
    from project.tools.pandas_tool import cleanup_plot_files


def _build_initial_state(query: str) -> AgentState:
    """Create a minimal initial state for graph execution."""
    return {
        "query": query,
        "plan": [],
        "data_source": "csv",
        "intermediate_results": [],
        "final_result": "",
        "insights": "",
        "memory": [],
        "retry_count": 0,
        "session_id": uuid4().hex,
    }


def _extract_result(final_state: AgentState) -> str:
    """Return final_result, or the latest intermediate output as fallback."""
    raw_final_result = final_state.get("final_result", "")
    if isinstance(raw_final_result, str) and raw_final_result.strip():
        return raw_final_result.strip()
    if raw_final_result:
        return str(raw_final_result)

    intermediate_results = final_state.get("intermediate_results", [])
    if isinstance(intermediate_results, list) and intermediate_results:
        return str(intermediate_results[-1])

    return ""


def main() -> None:
    """Read a user query, run the graph, and print result + insights."""
    _ = get_settings()
    cleanup_plot_files()
    app = build_workflow()

    query = input("Enter your analysis query: ").strip()
    if not query:
        print("No query provided.")
        return

    initial_state = _build_initial_state(query)

    try:
        final_state: AgentState = app.invoke(
            initial_state,
            config={"recursion_limit": 25},
        )
    except Exception as exc:
        print(f"Graph execution failed: {exc}")
        return

    result = _extract_result(final_state)
    raw_insights = final_state.get("insights", "")
    insights = raw_insights.strip() if isinstance(raw_insights, str) else str(raw_insights)

    print("\nResult:")
    print(result or "No result produced.")

    print("\nInsights:")
    print(insights or "No insights produced.")


if __name__ == "__main__":
    main()

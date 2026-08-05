"""LangGraph package for agent orchestration."""

try:
    from graph.edges import build_workflow
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.graph.edges import build_workflow

__all__ = ["build_workflow"]

"""Graph construction and routing for the LangGraph workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
import pandas as pd

try:
    from graph.nodes import critic_node, dashboard_node, executor_node, insight_node, planner_node
    from graph.state import AgentState
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.graph.nodes import critic_node, dashboard_node, executor_node, insight_node, planner_node
    from project.graph.state import AgentState


MAX_RETRIES = 2


def _route_after_planner(state: AgentState) -> str:
	"""Route to the dashboard pipeline when the planner selected a dashboard."""
	if not bool(state.get("dashboard", False)):
		return "executor"
	uploaded = state.get("uploaded_dataframe")
	if isinstance(uploaded, pd.DataFrame) and not uploaded.empty:
		return "dashboard"
	if state.get("data_source") == "postgres":
		return "dashboard"
	return "executor"


def _route_after_critic(state: AgentState) -> str:
	"""Route with bounded retries to avoid infinite execution loops."""
	raw_retry_count = state.get("retry_count", 0)
	retry_count = raw_retry_count if isinstance(raw_retry_count, int) else 0
	should_retry = bool(state.get("retry", False)) and retry_count < MAX_RETRIES
	if not should_retry:
		return "insight"
	if state.get("last_execution_node") == "dashboard":
		return "dashboard"
	return "executor"


def build_workflow():
	"""Build and compile the workflow:

	START -> planner -> (dashboard | executor) -> critic
	critic (retry=True) -> the node that last executed (dashboard or executor)
	critic (retry=False) -> insight -> END
	"""
	graph = StateGraph(AgentState)

	graph.add_node("planner", planner_node)
	graph.add_node("executor", executor_node)
	graph.add_node("dashboard", dashboard_node)
	graph.add_node("critic", critic_node)
	graph.add_node("insight", insight_node)

	graph.add_edge(START, "planner")
	graph.add_conditional_edges(
		"planner",
		_route_after_planner,
		{
			"dashboard": "dashboard",
			"executor": "executor",
		},
	)
	graph.add_edge("dashboard", "critic")
	graph.add_edge("executor", "critic")
	graph.add_conditional_edges(
		"critic",
		_route_after_critic,
		{
			"executor": "executor",
			"dashboard": "dashboard",
			"insight": "insight",
		},
	)
	graph.add_edge("insight", END)

	return graph.compile()

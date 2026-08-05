"""Shared state schema for graph execution."""

from typing import Any, Literal, NotRequired, TypedDict


DataSource = Literal["csv", "postgres", "mongo"]
StepTool = Literal["sql", "pandas", "visualization"]
FinalOutput = Literal["table", "chart", "summary"]


class PlanStep(TypedDict):
    """Structured planner step."""

    step: str
    tool: StepTool
    action: str


class AgentState(TypedDict):
    """Simple shared state for the AI data analysis graph."""

    query: str
    plan: list[PlanStep]
    data_source: DataSource
    intermediate_results: list[Any]
    final_result: str
    insights: str
    memory: list[Any]
    run_id: NotRequired[str]
    session_id: NotRequired[str]
    uploaded_dataframe: NotRequired[Any]
    final_output: NotRequired[FinalOutput]
    retry: NotRequired[bool]
    retry_count: NotRequired[int]

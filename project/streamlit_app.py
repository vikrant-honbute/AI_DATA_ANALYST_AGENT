"""Streamlit UI for running the AI Data Analysis Agent graph."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

from config import get_settings
from graph import build_workflow
from graph.state import AgentState


def _build_initial_state(query: str, uploaded_df: pd.DataFrame | None) -> AgentState:
    """Build the graph state from UI inputs."""
    state: AgentState = {
        "query": query,
        "plan": [],
        "data_source": "csv",
        "intermediate_results": [],
        "final_result": "",
        "insights": "",
        "memory": [],
        "retry_count": 0,
    }

    if uploaded_df is not None:
        state["uploaded_dataframe"] = uploaded_df.copy(deep=True)

    return state


def _extract_result(final_state: AgentState) -> str:
    """Return final result text with a deterministic fallback."""
    raw_final_result = final_state.get("final_result", "")
    if isinstance(raw_final_result, str) and raw_final_result.strip():
        return raw_final_result.strip()
    if raw_final_result:
        return str(raw_final_result)

    intermediate = final_state.get("intermediate_results", [])
    if isinstance(intermediate, list) and intermediate:
        return str(intermediate[-1])

    return ""


def _extract_insights(final_state: AgentState) -> str:
    """Return insight text from final state."""
    raw_insights = final_state.get("insights", "")
    if isinstance(raw_insights, str):
        return _dedupe_bullet_lines(raw_insights)
    return _dedupe_bullet_lines(str(raw_insights))


def _dedupe_bullet_lines(text: str) -> str:
    """Remove repeated bullet lines so insight sections stay concise."""
    if not text.strip():
        return ""

    cleaned_lines: list[str] = []
    seen_bullets: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("-"):
            normalized = re.sub(r"\s+", " ", stripped.lower())
            if normalized in seen_bullets:
                continue
            seen_bullets.add(normalized)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _extract_step_results(final_state: AgentState) -> list[dict[str, Any]]:
    """Return structured intermediate step results from final state."""
    raw_results = final_state.get("intermediate_results", [])
    if not isinstance(raw_results, list):
        return []

    return [item for item in raw_results if isinstance(item, dict)]


def _count_step_errors(step_results: list[dict[str, Any]]) -> int:
    """Count execution steps that contain an error payload."""
    return sum(
        1
        for step in step_results
        if isinstance(step.get("error"), str) and step.get("error", "").strip()
    )


def _render_step_payload(result: Any) -> None:
    """Render one step payload in a readable visual format."""
    if isinstance(result, pd.DataFrame):
        st.caption(f"DataFrame: {len(result)} rows x {len(result.columns)} columns")
        st.dataframe(result, use_container_width=True)
        return

    if isinstance(result, pd.Series):
        st.caption(f"Series: {len(result)} items")
        st.dataframe(result.to_frame(name=result.name or "value"), use_container_width=True)
        return

    if isinstance(result, dict):
        if result.get("type") == "plot":
            st.json(_json_safe(result), expanded=False)
            raw_path = result.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = Path.cwd() / image_path
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
            return

        st.json(_json_safe(result), expanded=False)
        return

    if isinstance(result, (list, tuple)):
        st.json(_json_safe(result), expanded=False)
        return

    rendered = str(result).strip()
    if rendered:
        st.code(rendered, language="text")
    else:
        st.info("No result payload for this step.")


def _collect_chart_paths(intermediate_results: Any) -> list[dict[str, Any]]:
    """Collect plot files and metadata from intermediate step results."""
    if not isinstance(intermediate_results, list):
        return []

    charts: list[dict[str, Any]] = []
    for step_number, item in enumerate(intermediate_results, start=1):
        if not isinstance(item, dict):
            continue

        result = item.get("result")
        if not isinstance(result, dict):
            continue

        if result.get("type") != "plot":
            continue

        raw_path = result.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue

        chart_path = Path(raw_path)
        if not chart_path.is_absolute():
            chart_path = Path.cwd() / chart_path

        if not chart_path.exists():
            continue

        raw_chart_type = result.get("chart_type")
        chart_type = (
            raw_chart_type.strip().lower()
            if isinstance(raw_chart_type, str) and raw_chart_type.strip()
            else "unknown"
        )

        raw_rows = result.get("rows")
        row_count = int(raw_rows) if isinstance(raw_rows, int) else 0

        raw_columns = result.get("columns")
        columns = [str(col) for col in raw_columns] if isinstance(raw_columns, list) else []

        charts.append(
            {
                "step_number": step_number,
                "step": str(item.get("step", "")),
                "tool": str(item.get("tool", "")),
                "action": str(item.get("action", "")).strip(),
                "path": str(chart_path),
                "chart_type": chart_type,
                "rows": row_count,
                "columns": columns,
            }
        )

    return charts


def _result_to_dataframe(result: Any) -> pd.DataFrame | None:
    """Convert step result payload into a DataFrame when possible."""
    if isinstance(result, pd.DataFrame):
        return result.copy(deep=True)

    if isinstance(result, pd.Series):
        return result.to_frame().T.reset_index(drop=True)

    return None


def _find_chart_source_dataframe(
    step_results: list[dict[str, Any]],
    chart_step_number: int,
) -> pd.DataFrame | None:
    """Find the most recent tabular result before the chart step."""
    if chart_step_number <= 1:
        return None

    for index in range(chart_step_number - 2, -1, -1):
        item = step_results[index]
        if not isinstance(item, dict):
            continue

        if isinstance(item.get("error"), str):
            continue

        dataframe = _result_to_dataframe(item.get("result"))
        if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
            return dataframe

    return None


def _pick_date_like_column(df: pd.DataFrame) -> str | None:
    """Pick a likely date/time column name for contextual chart explanations."""
    for column in df.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in ["date", "time", "month", "day", "year"]):
            return str(column)
    return None


def _format_number(value: float) -> str:
    """Format numeric values for concise UI display."""
    if pd.isna(value):
        return "n/a"

    if float(value).is_integer():
        return f"{int(value):,}"

    return f"{float(value):,.2f}"


def _build_chart_explanation(
    chart: dict[str, Any],
    source_df: pd.DataFrame | None,
) -> str:
    """Build deterministic chart-side explanation text."""
    lines: list[str] = []

    chart_type = str(chart.get("chart_type", "unknown")).title()
    rows = int(chart.get("rows", 0)) if isinstance(chart.get("rows"), int) else 0
    columns = chart.get("columns", [])
    action = str(chart.get("action", "")).strip()

    if action:
        lines.append(f"- Intent: {action}")

    if rows > 0:
        lines.append(f"- Chart type: {chart_type} built from {rows} rows.")
    else:
        lines.append(f"- Chart type: {chart_type}.")

    if isinstance(columns, list) and columns:
        shown_columns = ", ".join(str(col) for col in columns[:6])
        lines.append(f"- Columns used: {shown_columns}")

    if isinstance(source_df, pd.DataFrame) and not source_df.empty:
        numeric_cols = list(source_df.select_dtypes(include="number").columns)
        if numeric_cols:
            metric_col = str(numeric_cols[0])
            metric_series = pd.to_numeric(source_df[metric_col], errors="coerce").dropna()
            if not metric_series.empty:
                max_idx = metric_series.idxmax()
                min_value = float(metric_series.min())
                max_value = float(metric_series.max())
                avg_value = float(metric_series.mean())

                lines.append(
                    "- Primary metric "
                    f"{metric_col}: max {_format_number(max_value)}, "
                    f"min {_format_number(min_value)}, "
                    f"avg {_format_number(avg_value)}."
                )

                date_col = _pick_date_like_column(source_df)
                if date_col is not None and max_idx in source_df.index:
                    peak_point = source_df.loc[max_idx, date_col]
                    lines.append(
                        f"- Peak {metric_col} occurs at {date_col} = {peak_point}."
                    )

    if len(lines) == 2:
        lines.append("- Add a comparison query to generate deeper trend insights.")

    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    """Convert runtime objects into Streamlit JSON-safe objects."""
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(col) for col in value.columns],
            "preview": value.head(10).to_dict(orient="records"),
        }

    if isinstance(value, pd.Series):
        return {
            "type": "series",
            "name": str(value.name),
            "preview": value.head(10).to_dict(),
        }

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


@st.cache_resource(show_spinner=False)
def _get_compiled_graph():
    """Build and cache the compiled LangGraph application."""
    _ = get_settings()
    return build_workflow()


def main() -> None:
    """Render Streamlit controls and run the analysis workflow."""
    st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
    st.title("AI Data Analysis Agent")

    explain_mode = st.toggle("Explain mode", value=False)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    uploaded_df: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(
                f"Loaded {uploaded_file.name}: {len(uploaded_df)} rows x {len(uploaded_df.columns)} columns"
            )
            with st.expander("CSV Preview", expanded=False):
                st.dataframe(uploaded_df.head(25), use_container_width=True)
        except Exception as exc:
            st.error(f"Failed to read uploaded CSV: {exc}")

    query = st.text_area(
        "Query",
        placeholder="Example: Compare this month revenue with last month and show a chart.",
        height=120,
    )

    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    if "final_state" not in st.session_state:
        st.session_state["final_state"] = None
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""

    if run_clicked:
        if not query.strip():
            st.warning("Please enter a query before running analysis.")
        else:
            app = _get_compiled_graph()
            initial_state = _build_initial_state(query.strip(), uploaded_df)

            with st.spinner("Running workflow..."):
                try:
                    final_state: AgentState = app.invoke(
                        initial_state,
                        config={"recursion_limit": 25},
                    )
                except Exception as exc:
                    st.error(f"Workflow execution failed: {exc}")
                    return

            st.session_state["final_state"] = final_state
            st.session_state["last_query"] = query.strip()

    final_state = st.session_state.get("final_state")
    if not isinstance(final_state, dict):
        st.info("Run a query to see results.")
        return

    result = _extract_result(final_state)
    insights = _extract_insights(final_state)
    step_results = _extract_step_results(final_state)
    chart_entries = _collect_chart_paths(step_results)

    raw_retry_count = final_state.get("retry_count", 0)
    retry_count = raw_retry_count if isinstance(raw_retry_count, int) else 0
    data_source = str(final_state.get("data_source", "unknown")).upper()

    st.subheader("Run Summary")
    metric_col_source, metric_col_steps, metric_col_errors, metric_col_retries = st.columns(4)
    metric_col_source.metric("Data Source", data_source)
    metric_col_steps.metric("Steps", len(step_results))
    metric_col_errors.metric("Errors", _count_step_errors(step_results))
    metric_col_retries.metric("Retries", retry_count)

    tab_result, tab_steps, tab_insights, tab_charts = st.tabs(
        ["Final Output", "Execution Steps", "Insights", "Charts"]
    )

    with tab_result:
        st.markdown("### Result")
        if result:
            st.code(result, language="text")
        else:
            st.info("No result produced.")

    with tab_steps:
        if not step_results:
            st.info("No execution steps were captured.")
        else:
            for index, item in enumerate(step_results, start=1):
                step_name = str(item.get("step", "Unnamed step")).strip() or "Unnamed step"
                tool_name = str(item.get("tool", "unknown")).strip() or "unknown"
                error_text = item.get("error")
                has_error = isinstance(error_text, str) and bool(error_text.strip())
                status = "Failed" if has_error else "Completed"

                with st.expander(
                    f"Step {index}: {step_name} | Tool: {tool_name} | Status: {status}",
                    expanded=index == 1,
                ):
                    action = str(item.get("action", "")).strip()
                    if action:
                        st.caption(f"Action: {action}")

                    if has_error:
                        st.error(error_text)
                    else:
                        _render_step_payload(item.get("result"))

    with tab_insights:
        st.markdown("### Insights")
        if insights:
            st.markdown(insights)
        else:
            st.info("No insights produced.")

    with tab_charts:
        if chart_entries:
            for chart in chart_entries:
                step_name = str(chart.get("step", "Chart step")).strip() or "Chart step"
                tool_name = str(chart.get("tool", "visualization")).strip() or "visualization"
                st.markdown(f"### {step_name} ({tool_name})")

                chart_col, explain_col = st.columns([2, 1])

                with chart_col:
                    st.image(str(chart.get("path", "")), use_container_width=True)

                with explain_col:
                    st.markdown("#### Chart Explanation")
                    chart_step_number = int(chart.get("step_number", 0))
                    source_df = _find_chart_source_dataframe(step_results, chart_step_number)
                    st.markdown(_build_chart_explanation(chart, source_df))

                st.divider()
        else:
            st.info("No charts generated for this run.")

    if explain_mode:
        st.subheader("Execution Trace")
        trace_payload = {
            "query": st.session_state.get("last_query", ""),
            "data_source": final_state.get("data_source"),
            "plan": final_state.get("plan", []),
            "retry": final_state.get("retry", False),
            "retry_count": retry_count,
            "intermediate_results": step_results,
        }
        st.json(_json_safe(trace_payload), expanded=False)


if __name__ == "__main__":
    main()
"""Tests for executor step chaining and memory persistence decisions."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from project.graph.nodes.executor import (
    _extract_referenced_columns,
    _series_to_step_dataframe,
    executor_node,
)
from project.graph.nodes.insight import _should_save_memory


class TestSeriesConversion:
    def test_groupby_series_preserves_index_as_rows(self):
        df = pd.DataFrame({"region": ["N", "S", "N"], "sales": [10, 20, 30]})
        series = df.groupby("region")["sales"].sum()
        result = _series_to_step_dataframe(series)
        assert list(result["region"]) == ["N", "S"]
        assert list(result["sales"]) == [40, 20]

    def test_result_to_dataframe_matches(self):
        from project.streamlit_app import _result_to_dataframe

        df = pd.DataFrame({"region": ["N", "S", "N"], "sales": [10, 20, 30]})
        series = df.groupby("region")["sales"].sum()
        frame = _result_to_dataframe(series)
        assert frame is not None
        assert set(frame["region"]) == {"N", "S"}


class TestReferencedColumnExtraction:
    def test_json_operation_columns(self):
        action = json.dumps(
            {"operation": "groupby", "by": ["region"], "column": "sales", "function": "sum"}
        )
        assert set(_extract_referenced_columns(action)) == {"region", "sales"}

    def test_legacy_expression_columns(self):
        assert set(_extract_referenced_columns("result = df['sales'].sum()")) == {"sales"}


class TestExecutorNode:
    def test_pandas_step_runs_without_exec(self):
        df = pd.DataFrame({"region": ["N", "S", "N"], "sales": [10, 20, 30]})
        state = {
            "query": "sum sales by region",
            "plan": [
                {
                    "step": "Group sales by region",
                    "tool": "pandas",
                    "action": json.dumps(
                        {
                            "operation": "groupby",
                            "by": ["region"],
                            "column": "sales",
                            "function": "sum",
                        }
                    ),
                }
            ],
            "data_source": "csv",
            "intermediate_results": [],
            "final_result": "",
            "insights": "",
            "memory": [],
            "retry_count": 0,
            "uploaded_dataframe": df.copy(deep=True),
            "session_id": "test-session",
        }
        final = executor_node(state)
        assert not any("error" in item for item in final["intermediate_results"])
        assert "region" in final["final_result"]

    def test_invalid_plan_step_records_error(self):
        state = {
            "query": "bad plan",
            "plan": [
                {
                    "step": "Run arbitrary code",
                    "tool": "pandas",
                    "action": "result = df.to_csv('pwned.csv')",
                }
            ],
            "data_source": "csv",
            "intermediate_results": [],
            "final_result": "",
            "insights": "",
            "memory": [],
            "retry_count": 0,
            "uploaded_dataframe": pd.DataFrame({"a": [1]}),
            "session_id": "test-session",
        }
        final = executor_node(state)
        assert any("error" in item for item in final["intermediate_results"])

    def test_empty_plan_returns_message(self):
        state = {
            "query": "nothing",
            "plan": [],
            "data_source": "csv",
            "intermediate_results": [],
            "final_result": "",
            "insights": "",
            "memory": [],
            "retry_count": 0,
        }
        final = executor_node(state)
        assert final["final_result"] == "No valid plan steps were provided."


class TestShouldSaveMemory:
    def test_saves_when_no_errors(self):
        state = {
            "intermediate_results": [{"step": "ok", "tool": "pandas", "result": "x"}],
            "retry_count": 0,
        }
        assert _should_save_memory(state, "some result")

    def test_skips_when_any_step_failed(self):
        state = {
            "intermediate_results": [
                {"step": "bad", "tool": "pandas", "error": "boom"},
                {"step": "ok", "tool": "pandas", "result": "x"},
            ],
            "retry_count": 0,
        }
        assert not _should_save_memory(state, "some result")

    def test_skips_empty_result(self):
        state = {"intermediate_results": [], "retry_count": 0}
        assert not _should_save_memory(state, "  ")

"""Tests for the declarative pandas operation executor."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from project.tools.pandas_tool import _best_chart_for_data, _pick_chart_type, run_pandas_code


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "South", "North"],
            "sales": [10, 20, 30, 40, 50],
            "units": [1, 2, 3, 4, 5],
        }
    )


def _run(action, df):
    return run_pandas_code(json.dumps(action), df)


class TestDeclarativeOperations:
    def test_head(self, sample_df):
        result = _run({"operation": "head", "limit": 2}, sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_select(self, sample_df):
        result = _run({"operation": "select", "columns": ["region", "sales"]}, sample_df)
        assert list(result.columns) == ["region", "sales"]

    def test_filter_gt(self, sample_df):
        result = _run(
            {"operation": "filter", "column": "sales", "operator": "gt", "value": 20},
            sample_df,
        )
        assert list(result["sales"]) == [30, 40, 50]

    def test_filter_contains(self, sample_df):
        result = _run(
            {"operation": "filter", "column": "region", "operator": "contains", "value": "no"},
            sample_df,
        )
        assert list(result["region"]) == ["North", "North", "North"]

    def test_sort_descending(self, sample_df):
        result = _run(
            {"operation": "sort", "by": ["sales"], "ascending": False}, sample_df
        )
        assert list(result["sales"]) == [50, 40, 30, 20, 10]

    def test_value_counts(self, sample_df):
        result = _run(
            {"operation": "value_counts", "column": "region", "limit": 10}, sample_df
        )
        assert list(result.columns) == ["region", "count"]
        assert result["count"].sum() == 5

    def test_aggregate_sum(self, sample_df):
        result = _run({"operation": "aggregate", "column": "sales", "function": "sum"}, sample_df)
        assert result.loc[0, "value"] == 150

    def test_groupby_sum(self, sample_df):
        result = _run(
            {"operation": "groupby", "by": ["region"], "column": "sales", "function": "sum"},
            sample_df,
        )
        assert set(result["region"]) == {"North", "South"}
        assert int(result.loc[result["region"] == "North", "sales"].iloc[0]) == 90

    def test_correlation(self, sample_df):
        result = _run(
            {"operation": "correlation", "columns": ["sales", "units"]}, sample_df
        )
        assert result.loc["sales", "units"] == pytest.approx(1.0)

    def test_aggregate_rejects_unknown_function(self, sample_df):
        with pytest.raises(ValueError):
            _run({"operation": "aggregate", "column": "sales", "function": "cumsum"}, sample_df)

    def test_unknown_operation_rejected(self, sample_df):
        with pytest.raises(ValueError):
            _run({"operation": "eval", "code": "os.system('echo hi')"}, sample_df)

    def test_unknown_column_rejected(self, sample_df):
        with pytest.raises(ValueError):
            _run({"operation": "aggregate", "column": "nope", "function": "sum"}, sample_df)

    def test_plain_python_code_rejected(self, sample_df):
        with pytest.raises(ValueError):
            run_pandas_code("result = df.to_csv('x.csv')", sample_df)

    def test_invalid_json_rejected(self, sample_df):
        with pytest.raises(ValueError):
            run_pandas_code("result = df.head(20)", sample_df)

    def test_result_row_cap(self, sample_df):
        result = _run({"operation": "head", "limit": 10_000_000}, sample_df)
        assert len(result) <= 1000

    def test_dict_action_accepted(self, sample_df):
        result = run_pandas_code({"operation": "head", "limit": 1}, sample_df)
        assert len(result) == 1


class TestChartSelection:
    def test_auto_keyword_recognized(self):
        assert _pick_chart_type("auto") == "auto"
        assert _pick_chart_type("") == "auto"

    def test_keyword_mapping(self):
        assert _pick_chart_type("show a bar chart of revenue") == "bar"
        assert _pick_chart_type("plot the trend") == "line"

    def test_category_value_data_prefers_bar_or_pie(self):
        df = pd.DataFrame({"region": ["A", "B", "C"], "sales": [10, 20, 30]})
        assert _best_chart_for_data(df) in {"bar", "pie"}

    def test_three_segments_prefers_pie(self):
        df = pd.DataFrame({"region": ["A", "B", "C"], "sales": [10, 20, 30]})
        assert _best_chart_for_data(df) == "pie"

    def test_many_segments_prefers_bar(self):
        regions = [f"R{i}" for i in range(12)]
        df = pd.DataFrame({"region": regions, "sales": list(range(12))})
        assert _best_chart_for_data(df) == "bar"

    def test_two_segments_prefers_pie(self):
        df = pd.DataFrame({"region": ["A", "B"], "sales": [10, 20]})
        assert _best_chart_for_data(df) == "pie"

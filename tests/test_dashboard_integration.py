"""Integration tests: graph dashboard node + chart figure builders."""

from __future__ import annotations

import pandas as pd

from project.dashboard.charts import build_figure_for_chart
from project.dashboard.engine import compute_charts, compute_dashboard
from project.graph.nodes.dashboard import dashboard_node


def _superstore_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2021-01-01", "2021-02-01"]
            ),
            "region": ["West", "East", "West", "East", "West"],
            "category": ["Tech", "Furniture", "Tech", "Office", "Furniture"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0],
            "profit": [10.0, 20.0, -5.0, 30.0, 40.0],
            "quantity": [1, 2, 3, 4, 1],
        }
    )


def _state_with_df():
    return {
        "query": "dashboard",
        "plan": [],
        "data_source": "csv",
        "intermediate_results": [],
        "final_result": "",
        "insights": "",
        "memory": [],
        "retry_count": 0,
        "uploaded_dataframe": _superstore_like_df(),
    }


class TestDashboardNode:
    def test_node_produces_config_and_default_runtime(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("no llm")

        monkeypatch.setattr("project.dashboard.planner.get_llm", boom)
        out = dashboard_node(_state_with_df())
        assert out["dashboard"] is True
        assert out["dashboard_config"]
        assert isinstance(out["dashboard_config"], dict)
        assert out["dashboard_config"]["kpis"]
        assert out["dashboard_spec"]["kpis"]
        assert out["final_result"].startswith("Dashboard:")
        assert out["last_execution_node"] == "dashboard"

    def test_node_without_data_reports_unavailable(self):
        state = _state_with_df()
        state["uploaded_dataframe"] = None
        state["data_source"] = "csv"
        out = dashboard_node(state)
        assert out["dashboard_spec"] is None
        assert "could not be built" in out["final_result"]

    def test_config_in_state_is_engine_compatible(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("no llm")

        monkeypatch.setattr("project.dashboard.planner.get_llm", boom)
        out = dashboard_node(_state_with_df())
        runtime = compute_dashboard(out["dashboard_config"], _superstore_like_df())
        assert runtime["kpis"]
        assert runtime["charts"]


class TestChartFigures:
    def _payload_for(self, chart_spec):
        charts = compute_charts(_superstore_like_df(), [chart_spec])
        assert charts
        return charts[0]

    def test_line_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "line",
                "dimension": "order_date",
                "measures": [{"column": "sales", "aggregation": "sum"}],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None

    def test_bar_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "bar",
                "dimension": "category",
                "measures": [{"column": "sales", "aggregation": "sum"}],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None

    def test_donut_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "donut",
                "dimension": "category",
                "measures": [{"column": "sales", "aggregation": "sum"}],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None

    def test_hbar_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "hbar",
                "dimension": "region",
                "measures": [{"column": "sales", "aggregation": "sum"}],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None

    def test_scatter_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "scatter",
                "dimension": None,
                "measures": [
                    {"column": "sales", "aggregation": "sum"},
                    {"column": "profit", "aggregation": "sum"},
                ],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None

    def test_heatmap_figure_builds(self):
        payload = self._payload_for(
            {
                "id": "c",
                "chart_type": "heatmap",
                "dimension": None,
                "measures": [
                    {"column": "sales", "aggregation": "sum"},
                    {"column": "profit", "aggregation": "sum"},
                    {"column": "quantity", "aggregation": "sum"},
                ],
            }
        )
        figure = build_figure_for_chart(payload)
        assert figure is not None
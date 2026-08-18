"""Tests for the dashboard configuration schema, validation and fallback planner."""

from __future__ import annotations

import pandas as pd

from project.dashboard.models import DashboardConfig, config_from_dict
from project.dashboard.validate import validate_config


def _superstore_like() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2021-01-01", "2021-02-01", "2021-03-01"]
            ),
            "region": ["West", "East", "West", "East", "West", "East"],
            "category": ["Tech", "Furniture", "Tech", "Office", "Furniture", "Tech"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0, 400.0],
            "profit": [10.0, 20.0, -5.0, 30.0, 40.0, 50.0],
            "quantity": [1, 2, 3, 4, 1, 2],
        }
    )


class TestConfigSchema:
    def test_config_roundtrip(self):
        config = DashboardConfig(
            title="Test",
            kpis=[{"id": "k1", "column": "sales", "aggregation": "sum"}],
            filters=[{"id": "f1", "column": "region", "type": "categorical_multi"}],
            charts=[
                {
                    "id": "c1",
                    "chart_type": "line",
                    "dimension": "order_date",
                    "measures": [{"column": "sales", "aggregation": "sum"}],
                }
            ],
        )
        data = config.as_dict()
        assert data["title"] == "Test"
        assert data["kpis"][0]["column"] == "sales"
        assert data["charts"][0]["measures"][0]["column"] == "sales"

    def test_config_from_dict_invalid(self):
        assert config_from_dict(None) is None
        assert config_from_dict("nope") is None


class TestValidation:
    def test_unknown_columns_removed(self):
        df = _superstore_like()
        raw = {
            "kpis": [{"label": "x", "column": "not_a_column", "aggregation": "sum"}],
            "filters": [{"id": "f1", "column": "does_not_exist", "type": "categorical_multi"}],
            "charts": [
                {
                    "id": "c1",
                    "chart_type": "bar",
                    "dimension": "missing",
                    "measures": [{"column": "ghost", "aggregation": "avg"}],
                }
            ],
        }
        cleaned = validate_config(raw, df)
        assert cleaned["kpis"] == []
        assert cleaned["filters"] == []
        assert cleaned["charts"] == []

    def test_invalid_aggregation_fixed(self):
        df = _superstore_like()
        cleaned = validate_config({"kpis": [{"label": "x", "column": "sales", "aggregation": "avg"}]}, df)
        assert cleaned["kpis"][0]["aggregation"] == "sum"

    def test_non_numeric_sum_downgraded_to_count(self):
        df = _superstore_like()
        cleaned = validate_config(
            {"kpis": [{"label": "x", "column": "region", "aggregation": "sum"}]}, df
        )
        assert cleaned["kpis"][0]["aggregation"] == "count"

    def test_donut_high_cardinality_downgraded_to_bar(self):
        df = _superstore_like()
        raw = {
            "charts": [
                {
                    "id": "c1",
                    "chart_type": "donut",
                    "dimension": "order_date",
                    "measures": [{"column": "sales", "aggregation": "sum"}],
                }
            ]
        }
        cleaned = validate_config(raw, df)
        # order_date has > MAX_DONUT_CATEGORIES unique values -> bar
        assert cleaned["charts"][0]["chart_type"] == "bar"

    def test_line_on_non_time_downgraded_to_bar(self):
        df = _superstore_like()
        raw = {
            "charts": [
                {
                    "id": "c1",
                    "chart_type": "line",
                    "dimension": "region",
                    "measures": [{"column": "sales", "aggregation": "sum"}],
                }
            ]
        }
        cleaned = validate_config(raw, df)
        assert cleaned["charts"][0]["chart_type"] == "bar"

    def test_scatter_requires_two_measures(self):
        df = _superstore_like()
        raw = {
            "charts": [
                {
                    "id": "c1",
                    "chart_type": "scatter",
                    "dimension": None,
                    "measures": [{"column": "sales", "aggregation": "sum"}],
                }
            ]
        }
        cleaned = validate_config(raw, df)
        assert cleaned["charts"][0]["chart_type"] == "bar"

    def test_ratio_needs_denominator(self):
        df = _superstore_like()
        raw = {"kpis": [{"label": "margin", "column": "profit", "aggregation": "margin"}]}
        cleaned = validate_config(raw, df)
        assert cleaned["kpis"][0]["aggregation"] == "sum"

    def test_limit_caps(self):
        df = _superstore_like()
        raw = {
            "kpis": [{"label": f"k{i}", "column": "sales", "aggregation": "sum"} for i in range(20)],
            "filters": [
                {"id": f"f{i}", "column": "region", "type": "categorical_multi"} for i in range(20)
            ],
        }
        cleaned = validate_config(raw, df)
        assert len(cleaned["kpis"]) <= 6
        assert len(cleaned["filters"]) <= 6


class TestFallbackPlanner:
    def test_fallback_config_valid(self):
        df = _superstore_like()
        from project.dashboard.chart_selector import suggest_chart_configs, suggest_kpi_configs
        from project.dashboard.profiler import profile_dataframe

        profile = profile_dataframe(df)
        assert suggest_kpi_configs(df, profile)
        assert suggest_chart_configs(df, profile)

        from project.dashboard.planner import fallback_config

        config = fallback_config(df, profile)
        revalidated = validate_config(config, df)
        assert revalidated["kpis"]
        assert revalidated["filters"]
        assert revalidated["charts"]
        assert revalidated["time_dimension"] == "order_date"

    def test_build_dashboard_config_falls_back_without_llm(self, monkeypatch):
        from project.dashboard.planner import build_dashboard_config
        from project.dashboard.profiler import profile_dataframe

        df = _superstore_like()
        profile = profile_dataframe(df)

        def boom(*args, **kwargs):
            raise RuntimeError("no llm")

        monkeypatch.setattr("project.dashboard.planner.get_llm", boom)
        config = build_dashboard_config(df, profile, query="dashboard")
        assert config["kpis"]
        assert config["charts"]

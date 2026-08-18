"""Tests for the dashboard runtime engine: KPIs, filters, charts, facts, insights."""

from __future__ import annotations

import pandas as pd
import pytest

from project.dashboard.context import build_dashboard_context, build_filter_context_text
from project.dashboard.engine import compute_charts, compute_dashboard, compute_insight_facts
from project.dashboard.filters import apply_filters, build_filter_options
from project.dashboard.insights import build_dashboard_insights
from project.dashboard.kpis import compute_kpi_value, compute_kpis, period_change


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


def _config():
    return {
        "title": "Sales Dashboard",
        "time_dimension": "order_date",
        "primary_metric": "sales",
        "kpis": [
            {"id": "k_sales", "label": "Total Sales", "column": "sales", "aggregation": "sum", "format": "money", "delta": False},
            {"id": "k_profit", "label": "Total Profit", "column": "profit", "aggregation": "sum", "format": "money", "delta": True},
            {"id": "k_orders", "label": "Orders", "column": "order_date", "aggregation": "count", "format": "int"},
        ],
        "filters": [
            {"id": "f_year", "label": "Year", "column": "order_date", "type": "date_year"},
            {"id": "f_region", "label": "Region", "column": "region", "type": "categorical_multi"},
        ],
        "charts": [
            {
                "id": "c_trend",
                "chart_type": "line",
                "dimension": "order_date",
                "measures": [{"column": "sales", "aggregation": "sum"}],
                "max_points": 60,
                "width_span": 6,
            },
            {
                "id": "c_cat",
                "chart_type": "donut",
                "dimension": "category",
                "measures": [{"column": "sales", "aggregation": "sum"}],
                "max_points": 12,
                "width_span": 6,
            },
        ],
    }
class TestKpis:
    def test_sum_and_count(self):
        df = _superstore_like()
        payload = compute_kpi_value(df, {"column": "sales", "aggregation": "sum", "format": "money"})
        assert payload["raw_value"] == pytest.approx(1400.0)
        assert "$" in payload["value"]

    def test_count_without_column_counts_rows(self):
        df = _superstore_like()
        payload = compute_kpi_value(df, {"column": None, "aggregation": "count"})
        assert payload["raw_value"] == 6

    def test_nunique(self):
        df = _superstore_like()
        payload = compute_kpi_value(df, {"column": "category", "aggregation": "nunique"})
        assert payload["raw_value"] == 3

    def test_margin(self):
        df = _superstore_like()
        payload = compute_kpi_value(
            df, {"column": "profit", "denominator": "sales", "aggregation": "margin"}
        )
        assert payload["raw_value"] == pytest.approx(145.0 / 1400.0 * 100)

    def test_division_by_zero_safe(self):
        df = _superstore_like().copy()
        df["zero"] = 0.0
        payload = compute_kpi_value(
            df, {"column": "profit", "denominator": "zero", "aggregation": "margin"}
        )
        assert payload["raw_value"] is None
        assert payload["value"] == "—"

    def test_period_change(self):
        df = _superstore_like()
        change = period_change(df, {"column": "sales", "aggregation": "sum"}, "order_date")
        assert change is not None
        label, delta, tone = change
        assert delta
        assert tone in {"success", "danger", "neutral"}

    def test_compute_kpis_all(self):
        df = _superstore_like()
        kpis = compute_kpis(df, _config()["kpis"], "order_date")
        assert len(kpis) == 3
        orders = next(k for k in kpis if k["label"] == "Orders")
        assert orders["value"] == "6"


class TestFilters:
    def test_options(self):
        df = _superstore_like()
        specs = _config()["filters"]
        options = build_filter_options(df, specs)
        assert options["f_year"]["options"] == [2020, 2021]
        assert options["f_region"]["options"] == ["East", "West"]

    def test_apply_multi(self):
        df = _superstore_like()
        filtered = apply_filters(df, _config()["filters"], {"f_region": ["West"]})
        assert len(filtered) == 3
        assert set(filtered["region"]) == {"West"}

    def test_apply_year(self):
        df = _superstore_like()
        filtered = apply_filters(df, _config()["filters"], {"f_year": [2021]})
        assert len(filtered) == 3
        assert filtered["order_date"].dt.year.unique().tolist() == [2021]

    def test_apply_combined(self):
        df = _superstore_like()
        filtered = apply_filters(df, _config()["filters"], {"f_year": [2021], "f_region": ["East"]})
        assert len(filtered) == 2

    def test_empty_selection_is_noop(self):
        df = _superstore_like()
        filtered = apply_filters(df, _config()["filters"], {"f_region": []})
        assert len(filtered) == 6


class TestEngine:
    def test_compute_dashboard(self):
        df = _superstore_like()
        runtime = compute_dashboard(_config(), df)
        assert runtime["row_count"] == 6
        assert runtime["kpis"]
        assert runtime["charts"]
        chart_types = [c["chart_type"] for c in runtime["charts"]]
        assert "line" in chart_types
        assert "donut" in chart_types

    def test_compute_dashboard_with_filters_recomputes(self):
        df = _superstore_like()
        base = compute_dashboard(_config(), df)
        filtered = compute_dashboard(_config(), df, {"f_region": ["West"]})
        assert filtered["row_count"] == 3
        assert filtered["kpis"][0]["raw_value"] != base["kpis"][0]["raw_value"]

    def test_chart_payloads_have_data(self):
        df = _superstore_like()
        charts = compute_charts(df, _config()["charts"])
        for chart in charts:
            assert chart["data"]
            assert chart["chart_type"] in {"line", "donut"}

    def test_scatter_chart_payload(self):
        df = _superstore_like()
        chart_spec = {
            "id": "c",
            "chart_type": "scatter",
            "dimension": None,
            "measures": [
                {"column": "sales", "aggregation": "sum"},
                {"column": "profit", "aggregation": "sum"},
            ],
        }
        charts = compute_charts(df, [chart_spec])
        assert charts and charts[0]["chart_type"] == "scatter"
        assert charts[0]["data"][0]["x"] is not None


class TestInsights:
    def test_facts_and_deterministic_insights(self):
        df = _superstore_like()
        config = _config()
        runtime = compute_dashboard(config, df)
        facts = compute_insight_facts(df, config, runtime["kpis"])
        assert facts["primary_metric"] == "sales"
        assert facts["primary_total"] == pytest.approx(1400.0)
        insights = build_dashboard_insights(facts, config, df)
        assert insights
        assert all(item["body"] for item in insights)

    def test_filtered_view_facts_change(self):
        df = _superstore_like()
        config = _config()
        runtime = compute_dashboard(config, df, {"f_region": ["West"]})
        facts = compute_insight_facts(runtime["filtered_df"], config, runtime["kpis"])
        assert facts["row_count"] == 3


def test_filter_context_text():
    config = _config()
    assert "Region = West" in build_filter_context_text(config, {"f_region": ["West"]})
    assert build_filter_context_text(config, {}) == "None"


def test_dashboard_context_dict():
    config = _config()
    ctx = build_dashboard_context(config, {"f_region": ["East"]}, {"row_count": 3, "kpis": []})
    assert ctx["dashboard"] is True
    assert ctx["active_filters_text"] == "Region = East"
    assert ctx["row_count"] == 3
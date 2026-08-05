"""Tests for data source routing heuristics."""

from __future__ import annotations

from project.graph.nodes.planner import (
    _heuristic_route,
    _is_explicit_database_request,
    _parse_routed_source,
    _query_relates_to_past,
    _sanitize_csv_pandas_action,
    _fallback_csv_action,
)


class TestExplicitDatabaseRequest:
    def test_dashboard_not_database(self):
        assert not _is_explicit_database_request("build a dashboard from my uploaded csv")

    def test_comfortable_not_database(self):
        assert not _is_explicit_database_request("show comfortable ranges in my csv")

    def test_postgres_keyword(self):
        assert _is_explicit_database_request("query the postgres database")

    def test_sql_keyword(self):
        assert _is_explicit_database_request("run sql over the sales table")

    def test_table_phrase(self):
        assert _is_explicit_database_request("show me the database table customers")


class TestHeuristicRoute:
    def test_memory_query_routes_to_mongo(self):
        assert _heuristic_route("show my memory from earlier", has_uploaded_csv=True) == "mongo"

    def test_context_query_routes_to_mongo(self):
        assert _heuristic_route("what was our previous context", has_uploaded_csv=False) == "mongo"

    def test_csv_preferred_when_uploaded(self):
        assert _heuristic_route("analyze the data", has_uploaded_csv=True) == "csv"

    def test_csv_keyword_without_upload(self):
        assert _heuristic_route("parse my csv file", has_uploaded_csv=False) == "csv"

    def test_default_postgres(self):
        assert _heuristic_route("sum revenue by region", has_uploaded_csv=False) == "postgres"


class TestPastRelation:
    def test_memory_related(self):
        assert _query_relates_to_past("show my memory")

    def test_context_related(self):
        assert _query_relates_to_past("recall the previous context")

    def test_unrelated(self):
        assert not _query_relates_to_past("show total sales")


class TestRoutedSourceParsing:
    def test_parses_postgres(self):
        assert _parse_routed_source("Use postgres") == "postgres"

    def test_parses_csv(self):
        assert _parse_routed_source("csv") == "csv"

    def test_rejects_unknown(self):
        assert _parse_routed_source("oracle") is None


class TestCsvActionSanitization:
    COLUMNS = ["region", "sales"]

    def test_fallback_is_declarative_json(self):
        action = _fallback_csv_action("total sales", self.COLUMNS)
        assert '"operation"' in action

    def test_sum_finds_metric_column(self):
        action = _fallback_csv_action("total sales", self.COLUMNS)
        assert '"sales"' in action and '"sum"' in action

    def test_unknown_column_falls_back(self):
        action = _sanitize_csv_pandas_action(
            '{"operation": "aggregate", "column": "revenue", "function": "sum"}',
            "total sales",
            self.COLUMNS,
        )
        assert '"sales"' in action

    def test_valid_json_preserved(self):
        action = _sanitize_csv_pandas_action(
            '{"operation": "groupby", "by": ["region"], "column": "sales", "function": "mean"}',
            "average sales by region",
            self.COLUMNS,
        )
        assert '"groupby"' in action and '"region"' in action

    def test_python_code_rejected(self):
        action = _sanitize_csv_pandas_action(
            "result = df.head(20)", "show data", self.COLUMNS
        )
        assert '"operation"' in action

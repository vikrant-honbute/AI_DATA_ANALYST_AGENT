"""Tests for SQL read-only validation and guardrails."""

from __future__ import annotations

import pytest

from project.tools.sql_tool import _validate_safe_query


class TestSqlValidation:
    def test_accepts_simple_select(self):
        query = _validate_safe_query("SELECT * FROM sales LIMIT 10")
        assert query.startswith("SELECT")

    def test_accepts_trailing_semicolon(self):
        assert _validate_safe_query("SELECT 1;") == "SELECT 1"

    def test_accepts_with_cte(self):
        assert _validate_safe_query("WITH x AS (SELECT 1 AS a) SELECT a FROM x")

    def test_rejects_multiple_statements(self):
        with pytest.raises(ValueError):
            _validate_safe_query("SELECT 1; DROP TABLE sales")

    def test_rejects_insert(self):
        with pytest.raises(ValueError):
            _validate_safe_query("INSERT INTO sales VALUES (1)")

    def test_rejects_update(self):
        with pytest.raises(ValueError):
            _validate_safe_query("UPDATE sales SET price = 0")

    def test_rejects_delete(self):
        with pytest.raises(ValueError):
            _validate_safe_query("DELETE FROM sales")

    def test_rejects_drop(self):
        with pytest.raises(ValueError):
            _validate_safe_query("DROP TABLE sales")

    def test_rejects_copy(self):
        with pytest.raises(ValueError):
            _validate_safe_query("COPY sales TO '/tmp/x.csv'")

    def test_rejects_leading_comment_then_insert(self):
        with pytest.raises(ValueError):
            _validate_safe_query("-- comment\nINSERT INTO sales VALUES (1)")

    def test_rejects_pg_sleep(self):
        with pytest.raises(ValueError):
            _validate_safe_query("SELECT pg_sleep(100)")

    def test_rejects_nextval(self):
        with pytest.raises(ValueError):
            _validate_safe_query("SELECT nextval('seq')")

    def test_rejects_pg_read_file(self):
        with pytest.raises(ValueError):
            _validate_safe_query("SELECT pg_read_file('/etc/passwd')")

    def test_rejects_non_select_start(self):
        with pytest.raises(ValueError):
            _validate_safe_query("EXPLAIN SELECT 1")

    def test_rejects_empty_query(self):
        with pytest.raises(ValueError):
            _validate_safe_query("   ")

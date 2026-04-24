"""PostgreSQL query tool using SQLAlchemy and pandas."""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, inspect, text

try:
    from config import get_settings
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.config import get_settings

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_TOKENS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|EXEC|EXECUTE|COPY)\b",
    re.IGNORECASE,
)


def _validate_safe_query(query: str) -> str:
    """Allow only a single read-only SQL statement."""
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty SQL string.")

    normalized = query.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()

    if ";" in normalized:
        raise ValueError("Only one SQL statement is allowed.")
    if not _ALLOWED_START.match(normalized):
        raise ValueError("Only read-only SELECT/with queries are allowed.")
    if _FORBIDDEN_TOKENS.search(normalized):
        raise ValueError("Potentially unsafe SQL keyword detected.")

    return normalized


def query_postgres(query: str) -> pd.DataFrame:
    """Execute a safe PostgreSQL query and return the result as a DataFrame."""
    settings = get_settings()
    if not settings.postgres_url:
        raise ValueError("POSTGRES_URL is not configured.")

    safe_query = _validate_safe_query(query)
    engine = create_engine(settings.postgres_url, pool_pre_ping=True, future=True)

    try:
        with engine.connect() as connection:
            return pd.read_sql_query(text(safe_query), connection)
    finally:
        engine.dispose()


def fetch_postgres_schema() -> dict[str, list[str]]:
    """Fetch PostgreSQL table names and columns for prompt grounding."""
    settings = get_settings()
    if not settings.postgres_url:
        raise ValueError("POSTGRES_URL is not configured.")

    engine = create_engine(settings.postgres_url, pool_pre_ping=True, future=True)

    try:
        inspector = inspect(engine)
        schema_map: dict[str, list[str]] = {}

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            column_names = [
                str(column.get("name", "")).strip()
                for column in columns
                if str(column.get("name", "")).strip()
            ]
            schema_map[table_name] = column_names

        return schema_map
    finally:
        engine.dispose()

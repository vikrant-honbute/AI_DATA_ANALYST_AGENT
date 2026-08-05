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
_FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(?:pg_sleep|nextval|setval|currval|dblink|lo_import|lo_export|pg_read_file|pg_write_file)\s*\(",
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
    if _FORBIDDEN_FUNCTIONS.search(normalized):
        raise ValueError("Potentially unsafe SQL function detected.")

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
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(
                    text("SET LOCAL statement_timeout = :timeout"),
                    {"timeout": max(settings.postgres_statement_timeout_ms, 1)},
                )
                connection.execute(text("SET LOCAL lock_timeout = '2s'"))
                result = connection.execute(text(safe_query))
                rows = result.fetchmany(max(settings.postgres_max_rows, 1) + 1)
                columns = list(result.keys())
                if len(rows) > settings.postgres_max_rows:
                    raise ValueError(
                        f"Query returned more than {settings.postgres_max_rows} rows. Add a LIMIT clause."
                    )
                return pd.DataFrame(rows, columns=columns)
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

        for schema_name in settings.allowed_postgres_schemas:
            for table_name in inspector.get_table_names(schema=schema_name):
                columns = inspector.get_columns(table_name, schema=schema_name)
                qualified_name = f"{schema_name}.{table_name}"
                column_names = [
                    str(column.get("name", "")).strip()
                    for column in columns
                    if str(column.get("name", "")).strip()
                ]
                schema_map[qualified_name] = column_names

        return schema_map
    finally:
        engine.dispose()

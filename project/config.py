"""Environment-driven configuration for the AI Data Analysis Agent."""

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    groq_api_key: str
    postgres_url: str
    mongodb_url: str
    postgres_statement_timeout_ms: int
    postgres_max_rows: int
    allowed_postgres_schemas: tuple[str, ...]
    max_csv_bytes: int
    max_csv_rows: int
    max_csv_columns: int


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        postgres_url=os.getenv("POSTGRES_URL", ""),
        mongodb_url=os.getenv("MONGODB_URL", ""),
        postgres_statement_timeout_ms=int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "10000")),
        postgres_max_rows=int(os.getenv("POSTGRES_MAX_ROWS", "1000")),
        allowed_postgres_schemas=tuple(
            item.strip()
            for item in os.getenv("ALLOWED_POSTGRES_SCHEMAS", "public").split(",")
            if item.strip()
        ),
        max_csv_bytes=int(os.getenv("MAX_CSV_BYTES", str(10 * 1024 * 1024))),
        max_csv_rows=int(os.getenv("MAX_CSV_ROWS", "100000")),
        max_csv_columns=int(os.getenv("MAX_CSV_COLUMNS", "200")),
    )

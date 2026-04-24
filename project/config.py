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


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        postgres_url=os.getenv("POSTGRES_URL", ""),
        mongodb_url=os.getenv("MONGODB_URL", ""),
    )

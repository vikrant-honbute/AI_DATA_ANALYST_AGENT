"""MongoDB memory tool using pymongo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConfigurationError

try:
    from config import get_settings
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.config import get_settings

_DEFAULT_DB_NAME = "ai_agent"
_COLLECTION_NAME = "memory"


def _get_collection(client: MongoClient) -> Collection:
    """Return the target memory collection from the configured database."""
    try:
        database = client.get_default_database()
    except ConfigurationError:
        database = client[_DEFAULT_DB_NAME]
    return database[_COLLECTION_NAME]


def save_memory(session_id: str, query: str, result: Any) -> None:
    """Store one memory record with query and result."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    settings = get_settings()
    if not settings.mongodb_url:
        raise ValueError("MONGODB_URL is not configured.")

    client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
    try:
        collection = _get_collection(client)
        collection.create_index("expires_at", expireAfterSeconds=0)
        collection.create_index([("session_id", 1), ("created_at", DESCENDING)])
        collection.insert_one(
            {
                "session_id": session_id.strip(),
                "query": query.strip(),
                "result": result,
                "created_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
            }
        )
    finally:
        client.close()


def get_recent_memory(session_id: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent memory entries ordered from newest to oldest."""
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    if not session_id or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    settings = get_settings()
    if not settings.mongodb_url:
        raise ValueError("MONGODB_URL is not configured.")

    client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=5000)
    try:
        collection = _get_collection(client)
        cursor = (
            collection.find(
                {"session_id": session_id.strip()},
                {"_id": 0, "session_id": 0},
            )
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    finally:
        client.close()

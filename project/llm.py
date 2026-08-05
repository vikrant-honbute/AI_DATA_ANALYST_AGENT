"""Groq LLM factory for LangChain pipelines."""

from typing import Literal

from langchain_groq import ChatGroq

try:
    from config import Settings, get_settings
except ModuleNotFoundError:  # pragma: no cover - supports package-style execution.
    from project.config import Settings, get_settings

GroqModel = Literal[
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def get_llm(model: GroqModel = "openai/gpt-oss-20b") -> ChatGroq:
    """Return a configured Groq chat model, defaulting to openai/gpt-oss-20b."""
    settings: Settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is required for Groq LLM initialization.")

    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=model,
        temperature=0,
    )

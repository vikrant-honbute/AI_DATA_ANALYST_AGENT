"""Groq LLM factory for LangChain pipelines."""

from typing import Literal

from langchain_groq import ChatGroq

from config import Settings, get_settings

GroqModel = Literal[
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]


def get_llm(model: GroqModel = "llama-3.1-8b-instant") -> ChatGroq:
    """Return a configured Groq chat model."""
    settings: Settings = get_settings()
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is required for Groq LLM initialization.")

    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=model,
        temperature=0,
    )

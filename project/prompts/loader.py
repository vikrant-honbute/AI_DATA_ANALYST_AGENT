"""Prompt loading and rendering helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt_template(template_name: str) -> str:
    """Load one prompt template file from the prompts directory."""
    template_path = _PROMPTS_DIR / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def render_prompt(template_name: str, **kwargs: Any) -> str:
    """Render a prompt template with keyword substitutions."""
    template_text = load_prompt_template(template_name)
    values = {key: str(value) for key, value in kwargs.items()}
    return Template(template_text).safe_substitute(values)

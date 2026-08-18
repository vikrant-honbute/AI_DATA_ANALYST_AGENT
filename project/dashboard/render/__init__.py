"""Streamlit renderer for the AI-planned dynamic dashboard.

The renderer is a thin, deterministic layer: it reads a validated dashboard
configuration, renders Streamlit/Plotly widgets per filter definition, and pipes
the computed runtime payload (from ``dashboard.engine``) into KPI cards, charts
and insight cards. It never generates code and never calls the LLM itself.
"""

from .dashboard import render_ai_dashboard

__all__ = ["render_ai_dashboard"]
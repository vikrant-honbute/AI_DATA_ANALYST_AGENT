"""Tests for prompt rendering, memory scoping, and streamlit helpers."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from project.prompts import render_prompt
from project.tools.memory_tool import get_recent_memory, save_memory


class TestPromptRendering:
    def test_planner_prompt_renders(self):
        text = render_prompt(
            "planner_prompt.txt",
            query="sum sales",
            routed_data_source="csv",
            schema_text="none",
            csv_columns_text="[]",
            memory_context_text="none",
            use_memory_context="false",
            format_instructions="{}",
            dashboard_context="None",
        )
        assert "sum sales" in text

    def test_critic_prompt_renders(self):
        text = render_prompt(
            "critic_prompt.txt",
            data_source="csv",
            csv_columns="[]",
            serialized_plan="[]",
            serialized_results="[]",
            format_instructions="{}",
            execution_mode="executor",
            dashboard_context="None",
        )
        assert "csv" in text
        assert "Dashboard context" in text

    def test_router_prompt_renders(self):
        text = render_prompt(
            "router_prompt.txt", query="sum sales", has_uploaded_csv="false"
        )
        assert "sum sales" in text

    def test_insight_prompt_renders(self):
        text = render_prompt(
            "insight_prompt.txt", final_result="result", dashboard_context="None"
        )
        assert "result" in text

    def test_missing_variable_fails_fast(self):
        with pytest.raises(KeyError):
            render_prompt("planner_prompt.txt", query="sum sales")


class TestMemoryScoping:
    def test_save_requires_session_id(self):
        with pytest.raises(ValueError):
            save_memory("", "hello", {"value": 1})

    def test_get_recent_requires_session_id(self):
        with pytest.raises(ValueError):
            get_recent_memory("", limit=5)

    def test_get_recent_rejects_nonpositive_limit(self):
        with pytest.raises(ValueError):
            get_recent_memory("session-a", limit=0)


class _FakeSettings:
    postgres_statement_timeout_ms = 10000
    postgres_max_rows = 1000
    allowed_postgres_schemas = ("public",)
    max_csv_bytes = 10 * 1024 * 1024
    max_csv_rows = 100
    max_csv_columns = 50


@pytest.fixture
def fake_settings(monkeypatch):
    import project.streamlit_app as app

    monkeypatch.setattr(app, "get_settings", lambda: _FakeSettings())
    return app


def _upload_file(content: str, size: int | None = None) -> io.BytesIO:
    stream = io.BytesIO(content.encode("utf-8"))
    stream.seek(0)
    if size is not None:
        stream.size = size  # type: ignore[attr-defined]
    return stream


class TestCsvUploadLimits:
    def test_row_limit_enforced(self, fake_settings):
        content = "a,b\n" + "\n".join(f"1,2" for _ in range(150))
        with pytest.raises(ValueError, match="row limit"):
            fake_settings._read_uploaded_csv(_upload_file(content))

    def test_column_limit_enforced(self, fake_settings):
        columns = ",".join(f"col{i}" for i in range(60))
        with pytest.raises(ValueError, match="column limit"):
            fake_settings._read_uploaded_csv(_upload_file(f"{columns}\n"))

    def test_size_limit_enforced(self, fake_settings):
        stream = _upload_file("a,b\n1,2\n", size=11 * 1024 * 1024)
        with pytest.raises(ValueError, match="MB limit"):
            fake_settings._read_uploaded_csv(stream)

    def test_valid_csv_accepted(self, fake_settings):
        frame = fake_settings._read_uploaded_csv(_upload_file("a,b\n1,2\n3,4\n"))
        assert frame.shape == (2, 2)

    def test_binary_garbage_rejected(self, fake_settings):
        with pytest.raises(ValueError):
            fake_settings._read_uploaded_csv(
                io.BytesIO(b"\x00\xff\xfe\x01 not a csv at all \x80\x81")
            )


class TestTrendComputation:
    def test_rising(self):
        from project.streamlit_app import _compute_trend

        assert _compute_trend(pd.Series([1, 2, 3, 4, 5])) == "rising"

    def test_declining(self):
        from project.streamlit_app import _compute_trend

        assert _compute_trend(pd.Series([5, 4, 3, 2, 1])) == "declining"

    def test_v_shape_is_mixed(self):
        from project.streamlit_app import _compute_trend

        assert _compute_trend(pd.Series([1, 5, 1, 5, 1])) == "mixed"

    def test_flat_is_stable(self):
        from project.streamlit_app import _compute_trend

        assert _compute_trend(pd.Series([3, 3, 3, 3])) == "stable"

    def test_single_value_is_stable(self):
        from project.streamlit_app import _compute_trend

        assert _compute_trend(pd.Series([42])) == "stable"

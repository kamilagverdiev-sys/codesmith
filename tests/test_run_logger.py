"""Tests for the per-run observability log.

The run logger is intentionally lenient — an observability layer must
never break production — so most tests verify the HAPPY path plus the
"swallow errors silently" contract.
"""

from __future__ import annotations

import json
from pathlib import Path

from codesmith.run_logger import RunLogger, build_run_record


def test_record_creates_file_and_appends_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "runs.jsonl"
    logger = RunLogger(log_path)

    logger.record(
        build_run_record(
            session_id="sid-1",
            profile_key="local-fast",
            provider="ollama",
            model="qwen2.5-coder:7b",
            user_message="hello",
            steps=3,
            tokens=123,
            hit_limit=False,
            duration_ms=456,
            forced_final=False,
        )
    )
    logger.record(
        build_run_record(
            session_id="sid-2",
            profile_key="local-quality",
            provider="ollama",
            model="qwen3-coder:30b",
            user_message="build a web app",
            steps=9,
            tokens=5120,
            hit_limit=True,
            duration_ms=9999,
            forced_final=True,
            error=None,
            extra={"caller": "cli.solve"},
        )
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["session_id"] == "sid-1"
    assert first["tokens"] == 123
    assert first["forced_final"] is False
    assert second["profile"] == "local-quality"
    assert second["hit_limit"] is True
    assert second["forced_final"] is True
    assert second["extra"]["caller"] == "cli.solve"
    assert "ts" in first and "T" in first["ts"]


def test_tail_returns_last_records_in_order(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    logger = RunLogger(log_path)
    for i in range(5):
        logger.record(
            build_run_record(
                session_id=f"sid-{i}",
                profile_key="local-fast",
                provider="ollama",
                model="qwen2.5-coder:7b",
                user_message=f"msg {i}",
                steps=i,
                tokens=i * 10,
                hit_limit=False,
                duration_ms=i,
                forced_final=False,
            )
        )

    tail = logger.tail(limit=3)
    assert [r.session_id for r in tail] == ["sid-2", "sid-3", "sid-4"]
    assert tail[-1].steps == 4


def test_tail_on_missing_file_returns_empty_list(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "does-not-exist.jsonl")
    assert logger.tail(limit=10) == []


def test_record_truncates_long_user_message(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "runs.jsonl")
    very_long = "x" * 2000
    logger.record(
        build_run_record(
            session_id="sid",
            profile_key="config-default",
            provider="anthropic",
            model="claude-sonnet-4-6",
            user_message=very_long,
            steps=1,
            tokens=10,
            hit_limit=False,
            duration_ms=1,
            forced_final=False,
        )
    )
    tail = logger.tail(limit=1)
    assert len(tail) == 1
    # Truncation keeps the message bounded and marks it with an ellipsis.
    assert len(tail[0].user_message) <= 501
    assert tail[0].user_message.endswith("…")


def test_record_swallows_errors_silently(tmp_path: Path) -> None:
    """Writing to a path where the parent cannot be created must not raise."""
    # Put the log under a path where an existing FILE blocks the parent mkdir.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    logger = RunLogger(blocker / "runs.jsonl")
    # If this raises, the test fails implicitly.
    logger.record(
        build_run_record(
            session_id="sid",
            profile_key="config-default",
            provider="anthropic",
            model="claude-sonnet-4-6",
            user_message="hi",
            steps=1,
            tokens=1,
            hit_limit=False,
            duration_ms=1,
            forced_final=False,
        )
    )
    # The blocker file must be untouched.
    assert blocker.read_text(encoding="utf-8") == "not a dir"

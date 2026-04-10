"""Tests for context compaction module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from codesmith.compaction import (
    _DEFAULT_KEEP_TAIL,
    _format_message_for_summary,
    compact,
    should_compact,
)
from codesmith.config import CompactionConfig
from codesmith.llm import LLMResponse
from codesmith.session import Session


def _make_session(n_user_messages: int) -> Session:
    """Build a session with n_user_messages user+assistant pairs + system."""
    session = Session(
        session_id="test-compact",
        messages=[],
        workspace_dir=Path("/tmp/fake"),
    )
    session.add_system("You are a helpful assistant.")
    for i in range(n_user_messages):
        session.add_user(f"User message {i}: " + "x" * 100)
        session.add_assistant(f"Assistant reply {i}: " + "y" * 100)
    return session


# ---- should_compact ----


def test_should_compact_disabled() -> None:
    cfg = CompactionConfig(enabled=False)
    session = _make_session(100)
    assert should_compact(session, cfg) is False


def test_should_compact_not_enough_messages() -> None:
    cfg = CompactionConfig(enabled=True, trigger_at_messages=50, trigger_at_tokens=20000)
    session = _make_session(5)  # 10 non-system + system = 11 total
    assert should_compact(session, cfg) is False


def test_should_compact_by_message_count() -> None:
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=999999)
    session = _make_session(15)  # 30 non-system messages
    assert should_compact(session, cfg) is True


def test_should_compact_by_token_estimate() -> None:
    cfg = CompactionConfig(enabled=True, trigger_at_messages=99999, trigger_at_tokens=500)
    session = _make_session(15)  # ~30 msgs with ~100+ chars each
    assert should_compact(session, cfg) is True


def test_should_compact_just_under_threshold() -> None:
    cfg = CompactionConfig(enabled=True, trigger_at_messages=30, trigger_at_tokens=999999)
    # 14 pairs = 28 non-system messages, below 30
    session = _make_session(14)
    assert should_compact(session, cfg) is False


# ---- _format_message_for_summary ----


def test_format_user_message() -> None:
    result = _format_message_for_summary({"role": "user", "content": "hello"})
    assert result == "[user] hello"


def test_format_assistant_with_tool_calls() -> None:
    msg = {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}},
        ],
    }
    result = _format_message_for_summary(msg)
    assert "[called: read_file]" in result
    assert "Let me check." in result


def test_format_tool_result() -> None:
    msg = {"role": "tool", "tool_call_id": "call-abcdef12", "content": "file contents"}
    result = _format_message_for_summary(msg)
    assert "tool(call-abc" in result
    assert "file contents" in result


def test_format_long_content_truncated() -> None:
    msg = {"role": "user", "content": "z" * 2000}
    result = _format_message_for_summary(msg)
    assert "…(truncated)" in result
    assert len(result) < 1000


# ---- compact ----


def _mock_llm(summary: str = "Summary of conversation.") -> AsyncMock:
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=summary,
        tool_calls=[],
        finish_reason="stop",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        model="mock/test",
        raw=None,
    )
    return llm


@pytest.mark.asyncio
async def test_compact_replaces_old_messages() -> None:
    session = _make_session(20)  # system + 40 messages
    original_count = len(session.messages)
    assert original_count == 41  # 1 system + 20*2

    llm = _mock_llm("The user asked 20 questions and got answers.")
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=20000)
    removed = await compact(session, llm, cfg)

    assert removed > 0
    # System prompt preserved
    assert session.messages[0]["role"] == "system"
    assert session.messages[0]["content"] == "You are a helpful assistant."
    # Summary pair inserted
    assert session.messages[1]["role"] == "user"
    assert "[Conversation summary" in session.messages[1]["content"]
    assert "The user asked 20 questions" in session.messages[1]["content"]
    assert session.messages[2]["role"] == "assistant"
    assert "Understood" in session.messages[2]["content"]
    # Tail preserved
    assert len(session.messages) == 1 + 2 + _DEFAULT_KEEP_TAIL  # system + summary pair + tail


@pytest.mark.asyncio
async def test_compact_preserves_tail_messages() -> None:
    session = _make_session(20)
    tail_before = [
        m.get("content", "") for m in session.messages[-_DEFAULT_KEEP_TAIL:]
    ]

    llm = _mock_llm()
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=20000)
    await compact(session, llm, cfg)

    tail_after = [
        m.get("content", "") for m in session.messages[-_DEFAULT_KEEP_TAIL:]
    ]
    assert tail_before == tail_after


@pytest.mark.asyncio
async def test_compact_returns_zero_when_not_enough_messages() -> None:
    session = _make_session(3)  # only 6 non-system, below keep_tail + 2
    llm = _mock_llm()
    cfg = CompactionConfig(enabled=True, trigger_at_messages=2, trigger_at_tokens=100)
    removed = await compact(session, llm, cfg)
    assert removed == 0


@pytest.mark.asyncio
async def test_compact_handles_llm_failure() -> None:
    session = _make_session(20)
    original_messages = list(session.messages)

    llm = AsyncMock()
    llm.chat.side_effect = RuntimeError("LLM is down")
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=20000)

    removed = await compact(session, llm, cfg)
    assert removed == 0
    assert session.messages == original_messages


@pytest.mark.asyncio
async def test_compact_handles_empty_summary() -> None:
    session = _make_session(20)
    original_messages = list(session.messages)

    llm = _mock_llm("")  # empty summary
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=20000)

    removed = await compact(session, llm, cfg)
    assert removed == 0
    assert session.messages == original_messages


@pytest.mark.asyncio
async def test_compact_no_system_prompt() -> None:
    """Session without a system prompt should still compact correctly."""
    session = Session(
        session_id="test-no-sys",
        messages=[],
        workspace_dir=Path("/tmp/fake"),
    )
    for i in range(25):
        session.add_user(f"msg {i}: " + "w" * 100)
        session.add_assistant(f"reply {i}: " + "v" * 100)

    llm = _mock_llm("Summary without system prompt.")
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=99999)
    removed = await compact(session, llm, cfg)

    assert removed > 0
    assert session.messages[0]["role"] == "user"
    assert "[Conversation summary" in session.messages[0]["content"]


@pytest.mark.asyncio
async def test_compact_custom_keep_tail() -> None:
    session = _make_session(20)
    llm = _mock_llm("Short summary.")
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=99999)

    removed = await compact(session, llm, cfg, keep_tail=4)
    assert removed > 0
    # system(1) + summary_pair(2) + tail(4) = 7
    assert len(session.messages) == 7


@pytest.mark.asyncio
async def test_compact_long_block_capped() -> None:
    """Ensure very long conversation blocks are truncated for the summary prompt."""
    session = Session(
        session_id="test-long",
        messages=[],
        workspace_dir=Path("/tmp/fake"),
    )
    session.add_system("sys")
    for _i in range(30):
        session.add_user("x" * 1000)
        session.add_assistant("y" * 1000)

    llm = _mock_llm("Long conversation summarized.")
    cfg = CompactionConfig(enabled=True, trigger_at_messages=20, trigger_at_tokens=99999)
    removed = await compact(session, llm, cfg)

    assert removed > 0
    call_args = llm.chat.call_args
    messages = call_args[1].get("messages") or call_args[0][0]
    user_msg = messages[1]["content"]
    assert "…(conversation truncated for summary)" in user_msg

"""Tests for the Session dataclass — lifecycle, messages, token estimate."""

from __future__ import annotations

from pathlib import Path

import pytest

from codesmith.session import Session


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path / "workspaces"


# ---- create ----


def test_create_makes_workspace_dir(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    assert session.workspace_dir.is_dir()
    assert session.workspace_dir.parent == ws.resolve()


def test_create_uses_explicit_session_id(ws: Path) -> None:
    session = Session.create(workspace_root=ws, session_id="my-custom-id")
    assert session.session_id == "my-custom-id"
    assert session.workspace_dir.name == "my-custom-id"


def test_create_generates_uuid_by_default(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    # UUID format: 8-4-4-4-12 hex chars
    parts = session.session_id.split("-")
    assert len(parts) == 5


def test_create_starts_with_empty_messages(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    assert session.messages == []
    assert session.metadata == {}


# ---- message helpers ----


def test_add_user(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_user("hello")
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "hello"


def test_add_assistant_plain(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_assistant("response text")
    msg = session.messages[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "response text"
    assert "tool_calls" not in msg


def test_add_assistant_with_tool_calls(ws: Path) -> None:
    tc = [{"id": "tc1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    session = Session.create(workspace_root=ws)
    session.add_assistant("thinking...", tool_calls=tc)
    msg = session.messages[0]
    assert msg["tool_calls"] == tc


def test_add_assistant_empty_content_becomes_empty_string(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_assistant("")
    assert session.messages[0]["content"] == ""


def test_add_tool_result(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_tool_result("call-123", "output here")
    msg = session.messages[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-123"
    assert msg["content"] == "output here"


def test_add_system_inserts_at_front(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_user("hi")
    session.add_system("you are helpful")
    assert session.messages[0]["role"] == "system"
    assert session.messages[0]["content"] == "you are helpful"
    assert session.messages[1]["role"] == "user"


def test_add_system_replaces_existing_system(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_system("first prompt")
    session.add_user("hi")
    session.add_system("second prompt")
    # Should replace, not duplicate
    assert len(session.messages) == 2
    assert session.messages[0]["content"] == "second prompt"


# ---- token estimate ----


def test_token_estimate_empty(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    assert session.token_estimate() == 0


def test_token_estimate_proportional_to_content(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_user("x" * 400)  # ~100 tokens
    estimate = session.token_estimate()
    assert 80 <= estimate <= 120


def test_token_estimate_counts_all_messages(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.add_user("a" * 100)
    session.add_assistant("b" * 100)
    session.add_user("c" * 100)
    total = session.token_estimate()
    # 300 chars / 4 ≈ 75 tokens
    assert total == 75


# ---- destroy ----


def test_destroy_removes_workspace(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    assert session.workspace_dir.exists()
    session.destroy()
    assert not session.workspace_dir.exists()


def test_destroy_idempotent(ws: Path) -> None:
    session = Session.create(workspace_root=ws)
    session.destroy()
    # Second call should not raise
    session.destroy()


# ---- construction without create ----


def test_direct_construction(tmp_path: Path) -> None:
    session = Session(
        session_id="test-1",
        messages=[{"role": "user", "content": "hi"}],
        workspace_dir=tmp_path,
        metadata={"key": "val"},
    )
    assert session.session_id == "test-1"
    assert len(session.messages) == 1
    assert session.metadata["key"] == "val"
    assert session.created_at is not None

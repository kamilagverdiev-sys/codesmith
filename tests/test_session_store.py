"""Tests for the pluggable session store.

Covers both backends:
- InMemorySessionStore (Phase 1 behavior, dict-backed)
- SQLiteSessionStore (Phase 2 behavior, persists to a local SQLite file)

The SQLite tests exercise the full round-trip path: create → save →
simulate a process restart by throwing away the store instance →
reload from disk and verify the message history is intact.
"""

from __future__ import annotations

from pathlib import Path

from codesmith.session import Session
from codesmith.session_store import (
    InMemorySessionStore,
    SQLiteSessionStore,
    _derive_title,
)


def _seed_messages(session: Session) -> None:
    session.add_system("You are Codesmith.")
    session.add_user("Write a fibonacci function in Python.")
    session.add_assistant(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
            }
        ],
    )
    session.add_tool_result("call-1", "1")
    session.add_assistant(content="Here is the function: def fib(n): ...")


# ============================================================
# InMemorySessionStore
# ============================================================


class TestInMemorySessionStore:
    def test_create_and_get_roundtrip(self, tmp_path: Path) -> None:
        store = InMemorySessionStore(workspace_root=tmp_path / "ws")
        session = store.create(model_profile="local-fast")
        assert session.metadata["model_profile"] == "local-fast"
        assert store.get(session.session_id) is session

    def test_list_returns_created_sessions(self, tmp_path: Path) -> None:
        store = InMemorySessionStore(workspace_root=tmp_path / "ws")
        a = store.create()
        b = store.create()
        a.add_user("hello")
        store.save(a)
        rows = store.list(limit=10)
        ids = {r.session_id for r in rows}
        assert a.session_id in ids
        assert b.session_id in ids
        a_row = next(r for r in rows if r.session_id == a.session_id)
        assert a_row.message_count == 1
        assert "hello" in a_row.title

    def test_delete_removes_from_store(self, tmp_path: Path) -> None:
        store = InMemorySessionStore(workspace_root=tmp_path / "ws")
        session = store.create()
        sid = session.session_id
        assert store.delete(sid) is True
        assert store.get(sid) is None
        assert store.delete(sid) is False  # idempotent


# ============================================================
# SQLiteSessionStore
# ============================================================


class TestSQLiteSessionStore:
    def test_create_and_save_roundtrip(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            workspace_root=tmp_path / "ws",
        )
        session = store.create(model_profile="local-quality")
        _seed_messages(session)
        store.save(session)

        loaded_same = store.get(session.session_id)
        assert loaded_same is session  # live reference stays fresh

    def test_reload_after_fresh_store_instance(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sessions.db"
        ws = tmp_path / "ws"

        # First store — create, seed, save, drop the reference.
        store = SQLiteSessionStore(db_path=db_path, workspace_root=ws)
        session = store.create(model_profile="local-quality")
        _seed_messages(session)
        store.save(session)
        sid = session.session_id

        # "Process restart": a brand-new store instance must be able to
        # rematerialize the session off disk.
        del store
        revived = SQLiteSessionStore(db_path=db_path, workspace_root=ws)
        loaded = revived.get(sid)
        assert loaded is not None
        assert loaded.session_id == sid
        assert loaded.metadata.get("model_profile") == "local-quality"
        assert len(loaded.messages) == 5
        assert loaded.messages[1]["role"] == "user"
        assert "fibonacci" in loaded.messages[1]["content"].lower()
        # The tool_call message round-trips with tool_calls intact.
        assert loaded.messages[2]["role"] == "assistant"
        assert loaded.messages[2]["tool_calls"][0]["function"]["name"] == "execute_python"
        assert loaded.messages[3]["role"] == "tool"
        assert loaded.messages[3]["tool_call_id"] == "call-1"

    def test_list_returns_newest_first(self, tmp_path: Path) -> None:
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            workspace_root=tmp_path / "ws",
        )
        first = store.create()
        first.add_user("first task")
        store.save(first)
        second = store.create()
        second.add_user("second task")
        store.save(second)

        rows = store.list(limit=10)
        assert len(rows) >= 2
        assert rows[0].session_id == second.session_id
        assert rows[0].title == "second task"
        assert rows[0].message_count == 1

    def test_delete_cascades_messages(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sessions.db"
        ws = tmp_path / "ws"
        store = SQLiteSessionStore(db_path=db_path, workspace_root=ws)
        session = store.create()
        _seed_messages(session)
        store.save(session)
        sid = session.session_id

        assert store.delete(sid) is True
        assert store.delete(sid) is False

        # Reopen the DB and confirm nothing is left.
        reopened = SQLiteSessionStore(db_path=db_path, workspace_root=ws)
        assert reopened.get(sid) is None
        assert reopened.list() == []

    def test_save_is_idempotent(self, tmp_path: Path) -> None:
        """Saving the same session twice must not duplicate messages."""
        store = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            workspace_root=tmp_path / "ws",
        )
        session = store.create()
        _seed_messages(session)
        store.save(session)
        store.save(session)
        store.save(session)

        # Use a fresh store to bypass the live cache.
        fresh = SQLiteSessionStore(
            db_path=tmp_path / "sessions.db",
            workspace_root=tmp_path / "ws",
        )
        loaded = fresh.get(session.session_id)
        assert loaded is not None
        assert len(loaded.messages) == 5


def test_derive_title_uses_first_user_message(tmp_path: Path) -> None:
    session = Session.create(workspace_root=tmp_path / "ws")
    session.add_system("ignored")
    session.add_user("Help me refactor this module.")
    assert _derive_title(session) == "Help me refactor this module."


def test_derive_title_falls_back_to_session_id(tmp_path: Path) -> None:
    session = Session.create(workspace_root=tmp_path / "ws", session_id="abcdef1234")
    title = _derive_title(session)
    assert "abcdef12" in title

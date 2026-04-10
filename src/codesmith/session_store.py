"""Pluggable session store — SQLite persistence + in-memory fallback.

A SessionStore is the thing that knows how to:
  - create a new Session (with workspace dir);
  - save its message history somewhere durable;
  - load an existing session by id;
  - list recent sessions;
  - drop a session and its workspace.

Two implementations ship in the box:

    InMemorySessionStore — Phase 1 behavior. Sessions live in a dict
        and die with the process. Still useful for tests and for the
        CLI, which creates short-lived one-shot sessions.

    SQLiteSessionStore — Phase 2 behavior. Writes one row per session
        to `sessions`, one row per message to `messages`, keyed by
        ordinal so full history round-trips cleanly. Atomic per-call,
        thread-safe via a module-level lock, idempotent on `save`.

The API server picks which one to use from config.sessions.backend.

The store does NOT touch the workspace directory on disk beyond what
Session.create / Session.destroy already do — it only owns the message
history + metadata.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codesmith.session import Session

log = logging.getLogger(__name__)


# ============================================================
# Types
# ============================================================


class SessionSummaryRow:
    """Metadata row for list_sessions — not the full message history."""

    __slots__ = (
        "session_id",
        "created_at",
        "updated_at",
        "model_profile",
        "workspace",
        "title",
        "message_count",
    )

    def __init__(
        self,
        *,
        session_id: str,
        created_at: str,
        updated_at: str,
        model_profile: str | None,
        workspace: str,
        title: str,
        message_count: int,
    ) -> None:
        self.session_id = session_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.model_profile = model_profile
        self.workspace = workspace
        self.title = title
        self.message_count = message_count


def _now_iso() -> str:
    # Microsecond precision so two saves in the same second still sort
    # deterministically in list_sessions — we use updated_at as the
    # primary ORDER BY key.
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _derive_title(session: Session) -> str:
    """Use the first user message as a human-readable title. Fallback to id."""
    for msg in session.messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            title = content.strip().replace("\n", " ")
            return title[:120]
        break
    return f"session {session.session_id[:8]}"


# ============================================================
# In-memory store
# ============================================================


class InMemorySessionStore:
    """Sessions live in a plain dict; nothing is persisted to disk.

    The API server used to hold this dict directly; pulling it behind
    the same interface as the SQLite store means swapping backends is a
    one-line config change.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root
        self._sessions: dict[str, Session] = {}
        self._updated_at: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        model_profile: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        with self._lock:
            session = Session.create(
                workspace_root=self._workspace_root, session_id=session_id
            )
            if model_profile:
                session.metadata["model_profile"] = model_profile
            self._sessions[session.session_id] = session
            self._updated_at[session.session_id] = session.created_at
            return session

    def save(self, session: Session) -> None:
        # In-memory store already holds the live Session reference, so
        # there is nothing to persist — but we still record it in case
        # a freshly constructed Session (e.g. from a test) is saved
        # without first passing through create().
        with self._lock:
            self._sessions[session.session_id] = session
            self._updated_at[session.session_id] = datetime.now()

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self, limit: int = 20) -> list[SessionSummaryRow]:
        with self._lock:
            items = list(self._sessions.values())
            updated_map = dict(self._updated_at)
        # Sort by updated_at (most recently touched first), matching the
        # SQLite store behavior.
        items.sort(
            key=lambda s: updated_map.get(s.session_id, s.created_at),
            reverse=True,
        )
        rows: list[SessionSummaryRow] = []
        for s in items[:limit]:
            updated = updated_map.get(s.session_id, s.created_at)
            rows.append(
                SessionSummaryRow(
                    session_id=s.session_id,
                    created_at=s.created_at.isoformat(timespec="seconds"),
                    updated_at=updated.isoformat(timespec="seconds"),
                    model_profile=s.metadata.get("model_profile"),
                    workspace=str(s.workspace_dir),
                    title=_derive_title(s),
                    message_count=len(s.messages),
                )
            )
        return rows

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._updated_at.pop(session_id, None)
        if session is None:
            return False
        try:
            session.destroy()
        except Exception as e:  # noqa: BLE001
            log.warning("session.destroy failed for %s: %s", session_id, e)
        return True


# ============================================================
# SQLite store
# ============================================================


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    model_profile   TEXT,
    workspace_dir   TEXT NOT NULL,
    title           TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    ordinal         INTEGER NOT NULL,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_ord
    ON messages(session_id, ordinal);

CREATE INDEX IF NOT EXISTS idx_sessions_updated
    ON sessions(updated_at DESC);
"""


class SQLiteSessionStore:
    """Writes sessions + messages to a local SQLite file.

    The schema is tiny (two tables) and the hot-path operation `save`
    rewrites messages for a single session atomically inside one
    transaction — small enough that incremental append gains nothing.

    Concurrency: guarded by a process-local Lock. SQLite's own file
    lock handles cross-process, but we never intend to run more than
    one codesmith process per DB.
    """

    def __init__(self, db_path: Path, workspace_root: Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._workspace_root = Path(workspace_root).expanduser()
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        # Live Session objects, keyed by id — SQLite is the source of
        # truth on disk, but active in-flight sessions also stay in
        # memory so we don't re-materialise them for every chat turn.
        self._live: dict[str, Session] = {}

    # ---- connection helpers ----

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ---- public API ----

    def create(
        self,
        *,
        model_profile: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        session = Session.create(
            workspace_root=self._workspace_root, session_id=session_id
        )
        if model_profile:
            session.metadata["model_profile"] = model_profile
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, created_at, updated_at,
                    model_profile, workspace_dir, title, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    now,
                    now,
                    model_profile,
                    str(session.workspace_dir),
                    _derive_title(session),
                    json.dumps(session.metadata, ensure_ascii=False),
                ),
            )
        self._live[session.session_id] = session
        return session

    def save(self, session: Session) -> None:
        """Upsert the session row and fully rewrite its messages list.

        Full rewrite is cheap — message histories for one session are
        small, and the atomic transaction means readers never see a
        half-updated state. This also eliminates any drift between the
        in-memory list and the DB.
        """
        now = _now_iso()
        title = _derive_title(session)
        metadata_json = json.dumps(session.metadata, ensure_ascii=False)
        workspace_dir = str(session.workspace_dir)
        model_profile = session.metadata.get("model_profile")

        payloads: list[tuple[str, int, str, str]] = []
        for ordinal, msg in enumerate(session.messages):
            payloads.append(
                (
                    session.session_id,
                    ordinal,
                    json.dumps(msg, ensure_ascii=False, default=str),
                    now,
                )
            )

        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            try:
                cur = conn.execute(
                    "SELECT created_at FROM sessions WHERE session_id = ?",
                    (session.session_id,),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO sessions (
                            session_id, created_at, updated_at,
                            model_profile, workspace_dir, title, metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.session_id,
                            now,
                            now,
                            model_profile,
                            workspace_dir,
                            title,
                            metadata_json,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET updated_at = ?,
                            model_profile = ?,
                            workspace_dir = ?,
                            title = ?,
                            metadata_json = ?
                        WHERE session_id = ?
                        """,
                        (
                            now,
                            model_profile,
                            workspace_dir,
                            title,
                            metadata_json,
                            session.session_id,
                        ),
                    )
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session.session_id,),
                )
                if payloads:
                    conn.executemany(
                        """
                        INSERT INTO messages (session_id, ordinal, payload_json, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        payloads,
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        self._live[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        # Prefer the live in-memory copy when it exists so mutations
        # from the current chat aren't clobbered by a stale reload.
        live = self._live.get(session_id)
        if live is not None:
            return live

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT session_id, created_at, workspace_dir, metadata_json
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            messages_cur = conn.execute(
                """
                SELECT payload_json
                FROM messages
                WHERE session_id = ?
                ORDER BY ordinal ASC
                """,
                (session_id,),
            )
            message_rows = messages_cur.fetchall()

        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            created_at = datetime.fromisoformat(row["created_at"])
        except (TypeError, ValueError):
            created_at = datetime.now(UTC)

        messages: list[Any] = []
        for idx, m in enumerate(message_rows):
            try:
                messages.append(json.loads(m["payload_json"]))
            except json.JSONDecodeError:
                log.warning(
                    "corrupt message ordinal=%d in session %s, skipping",
                    idx,
                    session_id,
                )

        session = Session(
            session_id=row["session_id"],
            messages=messages,
            workspace_dir=Path(row["workspace_dir"]),
            metadata=metadata if isinstance(metadata, dict) else {},
            created_at=created_at,
        )
        self._live[session_id] = session
        return session

    def list(self, limit: int = 20) -> list[SessionSummaryRow]:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                SELECT s.session_id,
                       s.created_at,
                       s.updated_at,
                       s.model_profile,
                       s.workspace_dir,
                       s.title,
                       (SELECT COUNT(*) FROM messages m
                          WHERE m.session_id = s.session_id) AS msg_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

        return [
            SessionSummaryRow(
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                model_profile=row["model_profile"],
                workspace=row["workspace_dir"],
                title=row["title"],
                message_count=int(row["msg_count"] or 0),
            )
            for row in rows
        ]

    def delete(self, session_id: str) -> bool:
        live = self._live.pop(session_id, None)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            removed = cur.rowcount > 0
        if live is not None:
            try:
                live.destroy()
            except Exception as e:  # noqa: BLE001
                log.warning("session.destroy failed for %s: %s", session_id, e)
        return removed

"""Tests for the FastAPI Web UI / HTTP layer.

These tests don't touch the LLM or the sandbox — both are mocked.
We exercise:
  - GET /health, GET /api/info — read-only metadata
  - GET / — static UI is served
  - POST /api/sessions, GET /api/sessions, DELETE /api/sessions/{id}
  - POST /api/sessions/{id}/chat — full SSE round-trip with a stub agent

Run: pytest tests/test_api.py -q
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from codesmith.agent import AgentRun, AgentStep
from codesmith.api.main import AppState, app
from codesmith.config import (
    APIConfig,
    Config,
    FilesystemToolConfig,
    LLMConfig,
    LoggingConfig,
    LoopsConfig,
    MemoryConfig,
    SandboxConfig,
    SessionsConfig,
    ToolsConfig,
    WebSearchToolConfig,
)
from codesmith.llm import LLMResponse


def _make_config(tmp_path: Path) -> Config:
    """Build a fully synthetic Config that points at tmp_path."""
    return Config(
        llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b", api_base="http://x"),
        llm_fallbacks=[
            LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
        ],
        sandbox=SandboxConfig(image="codesmith-sandbox:test"),
        loops=LoopsConfig(),
        memory=MemoryConfig(enabled=False),
        tools=ToolsConfig(
            filesystem=FilesystemToolConfig(
                enabled=True, workspace_root=tmp_path / "ws"
            ),
            web_search=WebSearchToolConfig(enabled=False),
        ),
        sessions=SessionsConfig(path=tmp_path / "sessions.db"),
        api=APIConfig(),
        logging=LoggingConfig(),
    )


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppState]:
    """A fresh AppState scoped to one test, with a fake sandbox tool that
    is never actually invoked because the agent is mocked at run() level."""
    cfg = _make_config(tmp_path)

    # Don't let SandboxTool talk to a real Docker daemon during tests:
    # patch the constructor so it returns a no-op stub.
    class _NoopTool:
        name = "execute_python"
        description = "noop"

        def schema(self) -> dict:
            return {
                "type": "function",
                "function": {"name": "execute_python", "parameters": {}},
            }

        async def run(self, args: dict, session) -> object:  # noqa: ANN001
            from codesmith.tools.base import ToolResult

            return ToolResult(ok=True, content="noop", error=None)

    with patch("codesmith.api.main.SandboxTool", lambda _cfg: _NoopTool()):
        s = AppState(cfg)
    # Redirect the run log into tmp_path so tests don't touch the user's
    # real ~/.codesmith/logs/runs.jsonl file.
    from codesmith.run_logger import RunLogger

    s.run_logger = RunLogger(tmp_path / "runs.jsonl")
    yield s


@pytest.fixture
def client(state: AppState, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient that bypasses the lifespan loader and injects our state."""

    # Mock Ollama discovery so tests don't hang when Ollama isn't running.
    monkeypatch.setattr(
        "codesmith.api.main.list_installed_ollama_models",
        lambda _cfg: [],
    )

    # The default lifespan calls load_config(); we don't want that.
    async def _no_op_lifespan(_app):  # noqa: ANN001
        _app.state.codesmith = state
        yield

    from contextlib import asynccontextmanager

    app.router.lifespan_context = asynccontextmanager(_no_op_lifespan)

    with TestClient(app) as c:
        yield c


# ============================================================
# Read-only endpoints
# ============================================================


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_info_returns_config_summary(client: TestClient) -> None:
    r = client.get("/api/info")
    assert r.status_code == 200
    body = r.json()
    assert body["default_model_profile"] == "config-default"
    assert body["primary_provider"] == "ollama"
    assert body["primary_model"] == "qwen2.5-coder:7b"
    assert body["fallbacks"] == ["anthropic/claude-sonnet-4-6"]
    assert body["sandbox_image"] == "codesmith-sandbox:test"
    assert body["memory_enabled"] is False
    assert body["web_search_enabled"] is False


def test_models_endpoint_returns_profiles(client: TestClient) -> None:
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default_model_profile"] == "config-default"
    profile_keys = {item["key"] for item in body["profiles"]}
    assert "config-default" in profile_keys
    assert "local-auto" in profile_keys


def test_runs_endpoint_returns_empty_list_by_default(client: TestClient) -> None:
    r = client.get("/api/runs")
    assert r.status_code == 200
    body = r.json()
    assert body["runs"] == []
    assert body["path"]


def test_runs_endpoint_after_chat_contains_final_run(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "observe me"},
    ) as resp:
        assert resp.status_code == 200
        resp.read()  # drain

    r = client.get("/api/runs?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert len(body["runs"]) == 1
    run = body["runs"][0]
    assert run["session_id"] == sid
    assert run["profile"] == "config-default"
    assert run["steps"] == 2
    assert run["tokens"] == 43
    assert run["forced_final"] is False
    assert run["hit_limit"] is False
    assert run["error"] is None
    assert run["caller"] == "web"
    assert "observe me" in run["user_message"]


def test_runs_endpoint_limit_is_honored(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)
    sid = client.post("/api/sessions").json()["session_id"]
    for i in range(3):
        with client.stream(
            "POST",
            f"/api/sessions/{sid}/chat",
            json={"message": f"msg {i}"},
        ) as resp:
            resp.read()

    r = client.get("/api/runs?limit=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body["runs"]) == 2
    # Newest first: the last message posted is index 2 ("msg 2").
    assert "msg 2" in body["runs"][0]["user_message"]
    assert "msg 1" in body["runs"][1]["user_message"]


# ============================================================
# Session persistence + restore
# ============================================================


def test_get_session_messages_returns_full_history(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "persist me"},
    ) as resp:
        resp.read()

    r = client.get(f"/api/sessions/{sid}/messages")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    # System prompt is hidden from the UI-facing history.
    roles = [m["role"] for m in body["messages"]]
    assert "system" not in roles
    # User turn is preserved with its original text.
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "persist me"
    # Final assistant text is present.
    assistant_final = [m for m in body["messages"] if m["role"] == "assistant" and m["content"]]
    assert any("final answer is 42" in m["content"] for m in assistant_final)


def test_get_session_messages_unknown_returns_404(client: TestClient) -> None:
    r = client.get("/api/sessions/no-such-id/messages")
    assert r.status_code == 404


def test_list_sessions_returns_title_and_metadata(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "name this chat"},
    ) as resp:
        resp.read()

    r = client.get("/api/sessions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 1
    row = next(item for item in body if item["session_id"] == sid)
    assert row["title"] == "name this chat"
    assert row["messages"] >= 2  # user + assistant at minimum
    assert row["model_profile"] == "config-default"


def test_sqlite_session_store_survives_state_rebuild(
    tmp_path: Path,
    state: AppState,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a server restart: create+chat, then rebuild AppState on
    the same config, and confirm the history reloads from disk."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)
    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "should survive restart"},
    ) as resp:
        resp.read()

    # Rebuild a brand-new AppState against the same config (same DB
    # file). The previous live cache is gone, so get() must hit disk.
    from codesmith.api.main import AppState
    from codesmith.session_store import SQLiteSessionStore

    fresh_state = AppState(state.config)
    assert isinstance(fresh_state.session_store, SQLiteSessionStore)
    loaded = fresh_state.session_store.get(sid)
    assert loaded is not None
    roles = [m.get("role") for m in loaded.messages]
    assert "user" in roles
    assert any(
        isinstance(m.get("content"), str) and "should survive restart" in m["content"]
        for m in loaded.messages
    )


def test_abort_inflight_cancels_registered_task(
    state: AppState,
) -> None:
    """Unit-level test for the abort plumbing in AppState.

    Driving real cancellation through TestClient + SSE + threading is
    flaky because TestClient serializes the request/response round-trip
    in one portal thread; instead we hit the exact primitives the DELETE
    /chat endpoint uses (register_inflight + abort_inflight) and assert
    the cancellation actually reaches the task.
    """
    import asyncio as _asyncio

    async def _scenario() -> tuple[bool, bool]:
        cancelled = _asyncio.Event()

        async def long_running() -> None:
            try:
                await _asyncio.sleep(10)
            except _asyncio.CancelledError:
                cancelled.set()
                raise

        task = _asyncio.create_task(long_running())
        state.register_inflight("sid-abc", task)

        # Let the task actually start before aborting.
        await _asyncio.sleep(0)

        ok = state.abort_inflight("sid-abc")
        with contextlib.suppress(_asyncio.CancelledError):
            await task
        state.clear_inflight("sid-abc", task)

        return ok, cancelled.is_set()

    import contextlib

    ok, was_cancelled = _asyncio.run(_scenario())
    assert ok is True
    assert was_cancelled is True
    # After the task finishes the inflight map must be empty.
    assert state._inflight.get("sid-abc") is None


def test_abort_inflight_returns_false_when_no_task(state: AppState) -> None:
    assert state.abort_inflight("no-such-session") is False


def test_abort_chat_endpoint_returns_false_for_idle_session(
    client: TestClient,
) -> None:
    """DELETE /chat on a known session with no inflight task is a noop."""
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.delete(f"/api/sessions/{sid}/chat")
    assert r.status_code == 200
    assert r.json()["cancelled"] is False


def test_abort_chat_unknown_session_404(client: TestClient) -> None:
    r = client.delete("/api/sessions/no-such-id/chat")
    assert r.status_code == 404


# ============================================================
# Personas + /plan endpoint (5a)
# ============================================================


def test_personas_endpoint_returns_four_roles(client: TestClient) -> None:
    r = client.get("/api/personas")
    assert r.status_code == 200
    body = r.json()
    keys = [p["key"] for p in body["personas"]]
    assert keys == ["default", "architect", "coder", "reviewer"]
    architect = next(p for p in body["personas"] if p["key"] == "architect")
    assert architect["no_tools"] is True


def _fake_architect_run_factory():
    """A fake agent.run that emits a plausible architect plan as final text."""
    plan_text = (
        "## Goal\n"
        "Make the test pass.\n\n"
        "## Assumptions\n"
        "- Python repo with pytest.\n\n"
        "## Plan\n"
        "1. read the failing test\n"
        "2. fix the bug\n"
        "3. verify with pytest\n\n"
        "## Acceptance checks\n"
        "- pytest -q shows 0 failures.\n\n"
        "## Risks\n"
        "- None obvious.\n"
    )

    async def fake_run(self, session, on_step=None):
        # The architect writes its plan as plain assistant text and
        # finishes in one step. We mirror that here so the /plan
        # endpoint reads exactly what the real architect would produce.
        session.add_assistant(content=plan_text)
        step = AgentStep(
            response=LLMResponse(
                content=plan_text,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
                model="test/architect",
                raw=None,
            ),
            tool_results=[],
        )
        if on_step:
            await on_step(step, 0)
        return AgentRun(
            final_text=plan_text,
            steps=[step],
            hit_limit=False,
            total_tokens=60,
        )

    return fake_run, plan_text


def test_plan_endpoint_happy_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, plan_text = _fake_architect_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/plan",
        json={"task": "fix the failing test in tests/test_foo.py"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["persona"] == "architect"
    # Endpoint strips trailing whitespace for display.
    assert body["plan_text"] == plan_text.strip()
    assert body["steps"] == 1
    assert body["tokens"] == 60
    assert body["duration_ms"] >= 0


def test_plan_endpoint_stores_plan_on_session(
    client: TestClient,
    state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan endpoint must park the plan text on session metadata
    so /execute-plan (5b) and the Web UI can pick it up later."""
    fake_run, plan_text = _fake_architect_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    client.post(
        f"/api/sessions/{sid}/plan",
        json={"task": "build the thing"},
    )

    session = state.session_store.get(sid)
    assert session is not None
    last_plan = session.metadata.get("last_plan")
    assert last_plan is not None
    assert last_plan["task"] == "build the thing"
    assert last_plan["text"] == plan_text.strip()
    assert "ts" in last_plan


def test_plan_endpoint_does_not_pollute_main_chat(
    client: TestClient,
    state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An architect planning turn must leave the main session's
    messages list untouched — planning runs on an isolated sub-session
    so chat history stays clean."""
    fake_run, _ = _fake_architect_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    # Sanity: fresh session has no messages yet.
    before = client.get(f"/api/sessions/{sid}/messages").json()
    assert before["messages"] == []

    client.post(
        f"/api/sessions/{sid}/plan",
        json={"task": "refactor /api/sessions"},
    )

    after = client.get(f"/api/sessions/{sid}/messages").json()
    assert after["messages"] == []  # still empty — plan stayed isolated


def test_plan_endpoint_unknown_session_404(client: TestClient) -> None:
    r = client.post(
        "/api/sessions/no-such/plan",
        json={"task": "anything"},
    )
    assert r.status_code == 404


def test_plan_endpoint_empty_task_validation_error(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/plan",
        json={"task": ""},
    )
    assert r.status_code == 422


# ============================================================
# Pending changes + auto-approve (5b)
# ============================================================


def test_auto_approve_defaults_to_true_on_new_session(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.get(f"/api/sessions/{sid}/pending-changes")
    assert r.status_code == 200
    body = r.json()
    assert body["auto_approve"] is True
    assert body["changes"] == []


def test_set_auto_approve_flag(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/auto-approve",
        json={"enabled": False},
    )
    assert r.status_code == 200
    assert r.json()["auto_approve"] is False
    # Verify via list endpoint.
    r = client.get(f"/api/sessions/{sid}/pending-changes")
    assert r.json()["auto_approve"] is False
    # Flip back.
    r = client.post(
        f"/api/sessions/{sid}/auto-approve",
        json={"enabled": True},
    )
    assert r.json()["auto_approve"] is True


def test_approve_pending_change_writes_file(
    client: TestClient, state: AppState
) -> None:
    """End-to-end: queue a change on a session, hit the approve endpoint,
    verify the target file now exists on disk."""
    sid = client.post("/api/sessions").json()["session_id"]
    # Flip session to review mode.
    client.post(f"/api/sessions/{sid}/auto-approve", json={"enabled": False})
    # Queue a change by calling queue_change directly on the live session.
    from codesmith.pending_changes import queue_change

    session = state.session_store.get(sid)
    assert session is not None
    change = queue_change(
        session,
        kind="write_file",
        path="approved.py",
        before="",
        after="print('approved')\n",
    )
    state.save_session(session)
    idx = change["idx"]

    # Approve.
    r = client.post(f"/api/sessions/{sid}/pending-changes/{idx}/approve")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "applied"
    assert body["path"] == "approved.py"

    # File exists on disk.
    target = session.workspace_dir / "approved.py"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "print('approved')\n"

    # Second approve is a conflict.
    r = client.post(f"/api/sessions/{sid}/pending-changes/{idx}/approve")
    assert r.status_code == 409


def test_reject_pending_change_marks_state_and_does_not_write(
    client: TestClient, state: AppState
) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    client.post(f"/api/sessions/{sid}/auto-approve", json={"enabled": False})
    from codesmith.pending_changes import queue_change

    session = state.session_store.get(sid)
    assert session is not None
    change = queue_change(
        session,
        kind="write_file",
        path="rejected.py",
        before="",
        after="print('nope')\n",
    )
    state.save_session(session)
    idx = change["idx"]

    r = client.post(f"/api/sessions/{sid}/pending-changes/{idx}/reject")
    assert r.status_code == 200
    assert r.json()["state"] == "rejected"
    assert not (session.workspace_dir / "rejected.py").exists()


def test_approve_unknown_idx_returns_404(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/pending-changes/999/approve")
    assert r.status_code == 404


def test_pending_changes_unknown_session_returns_404(client: TestClient) -> None:
    r = client.get("/api/sessions/no-such/pending-changes")
    assert r.status_code == 404


def test_in_memory_backend_config_path(tmp_path: Path) -> None:
    """Switching sessions.backend to 'memory' must give an InMemorySessionStore."""
    from unittest.mock import patch as _patch

    from codesmith.api.main import AppState
    from codesmith.config import (
        APIConfig,
        Config,
        FilesystemToolConfig,
        LLMConfig,
        LoggingConfig,
        LoopsConfig,
        MemoryConfig,
        SandboxConfig,
        SessionsConfig,
        ToolsConfig,
        WebSearchToolConfig,
    )
    from codesmith.session_store import InMemorySessionStore

    cfg = Config(
        llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"),
        llm_fallbacks=[],
        sandbox=SandboxConfig(image="codesmith-sandbox:test"),
        loops=LoopsConfig(),
        memory=MemoryConfig(enabled=False),
        tools=ToolsConfig(
            filesystem=FilesystemToolConfig(
                enabled=True, workspace_root=tmp_path / "ws"
            ),
            web_search=WebSearchToolConfig(enabled=False),
        ),
        sessions=SessionsConfig(backend="memory", path=tmp_path / "unused.db"),
        api=APIConfig(),
        logging=LoggingConfig(),
    )

    class _NoopTool:
        name = "execute_python"
        description = "noop"

        def schema(self) -> dict:
            return {"type": "function", "function": {"name": "execute_python", "parameters": {}}}

        async def run(self, args: dict, session) -> object:  # noqa: ANN001
            from codesmith.tools.base import ToolResult

            return ToolResult(ok=True, content="noop", error=None)

    with _patch("codesmith.api.main.SandboxTool", lambda _cfg: _NoopTool()):
        state = AppState(cfg)
    assert isinstance(state.session_store, InMemorySessionStore)


def test_index_serves_static_ui(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "codesmith" in r.text.lower()


# ============================================================
# Session lifecycle
# ============================================================


def test_create_and_list_and_delete_session(client: TestClient) -> None:
    # Create
    r = client.post("/api/sessions")
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert r.json()["model_profile"] == "config-default"
    assert sid

    # List has it
    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = [s["session_id"] for s in r.json()]
    assert sid in ids

    # Delete
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # Second delete is 404
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 404


def test_chat_unknown_session_404(client: TestClient) -> None:
    r = client.post("/api/sessions/does-not-exist/chat", json={"message": "hi"})
    assert r.status_code == 404


# ============================================================
# Chat SSE round-trip with a stubbed agent
# ============================================================


def _fake_run_factory():
    """Build a fake Agent.run that fires on_step twice and returns a final.

    IMPORTANT: the real Agent.run mutates `session.messages` inside each
    step(): it appends an assistant message (and any tool results). The
    session store saves those messages. If the fake does NOT mutate the
    session, downstream tests that read the persisted history will see
    only the user message. So we mirror the real behavior here by
    appending the corresponding assistant/tool messages.
    """

    async def fake_run(self, session, on_step=None):
        # Step 1: tool call
        tool_call = {
            "id": "tc-1",
            "type": "function",
            "function": {
                "name": "execute_python",
                "arguments": json.dumps({"code": "print(1)"}),
            },
        }
        step1 = AgentStep(
            response=LLMResponse(
                content="thinking...",
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                model="test/test",
                raw=None,
            ),
            tool_results=[],
        )
        session.add_assistant(content="thinking...", tool_calls=[tool_call])
        session.add_tool_result("tc-1", "1")
        if on_step:
            await on_step(step1, 0)

        # Step 2: final answer
        step2 = AgentStep(
            response=LLMResponse(
                content="final answer is 42",
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
                model="test/test",
                raw=None,
            ),
            tool_results=[],
        )
        session.add_assistant(content="final answer is 42")
        if on_step:
            await on_step(step2, 1)

        return AgentRun(
            final_text="final answer is 42",
            steps=[step1, step2],
            hit_limit=False,
            total_tokens=43,
        )

    return fake_run


def _parse_sse_stream(raw: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE byte stream into a list of (event_name, data_dict).

    sse_starlette emits CRLF line endings, so the event separator is
    "\r\n\r\n", not "\n\n".
    """
    # Normalize CRLF to LF so we can split on the simpler separator.
    normalized = raw.replace("\r\n", "\n")
    events = []
    for chunk in normalized.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name = "message"
        data_lines = []
        for line in chunk.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        try:
            data = json.loads("".join(data_lines))
        except json.JSONDecodeError:
            data = {}
        events.append((name, data))
    return events


def test_chat_streams_step_and_final_events(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]

    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "hello"},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    names = [name for name, _ in events if name != "ping"]

    assert "start" in names
    assert names.count("step") == 1
    assert "final" in names

    final = next(d for n, d in events if n == "final")
    assert final["text"] == "final answer is 42"
    assert final["steps"] == 2
    assert final["tokens"] == 43
    assert final["hit_limit"] is False


def test_chat_emits_error_event_on_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(self, session, on_step=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("codesmith.agent.Agent.run", boom)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "hello"},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    names = [name for name, _ in events if name != "ping"]
    assert "error" in names
    err = next(d for n, d in events if n == "error")
    assert "kaboom" in err["error"]


# ============================================================
# Execute plan endpoint
# ============================================================


def test_execute_plan_from_last_plan(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /execute-plan uses session.metadata["last_plan"] when no body plan_text."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]

    # Store a plan on the session via the /plan endpoint.
    # We need to mock Agent.run for the plan call too.
    r = client.post(f"/api/sessions/{sid}/plan", json={"task": "build a CLI"})
    assert r.status_code == 200
    assert r.json()["plan_text"] == "final answer is 42"

    # Now execute it.
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/execute-plan",
        json={},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    names = [name for name, _ in events if name != "ping"]
    assert "start" in names
    assert "final" in names

    final = next(d for n, d in events if n == "final")
    assert final["text"] == "final answer is 42"


def test_execute_plan_with_explicit_plan_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /execute-plan with explicit plan_text in body."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]

    with client.stream(
        "POST",
        f"/api/sessions/{sid}/execute-plan",
        json={"plan_text": "Step 1: create hello.py\nStep 2: run it"},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    names = [name for name, _ in events if name != "ping"]
    assert "final" in names


def test_execute_plan_404_when_no_plan(client: TestClient) -> None:
    """POST /execute-plan without plan or last_plan returns 404."""
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/execute-plan", json={})
    assert r.status_code == 404
    assert "No plan found" in r.json()["detail"]


def test_execute_plan_404_unknown_session(client: TestClient) -> None:
    r = client.post("/api/sessions/nonexistent/execute-plan", json={"plan_text": "x"})
    assert r.status_code == 404


def test_final_event_includes_pending_count(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final SSE event should include pending_count field."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "test pending count"},
    ) as resp:
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    final = next(d for n, d in events if n == "final")
    assert "pending_count" in final
    assert final["pending_count"] == 0


def test_final_event_includes_compacted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final SSE event should include compacted field."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    with client.stream(
        "POST",
        f"/api/sessions/{sid}/chat",
        json={"message": "test compacted"},
    ) as resp:
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    final = next(d for n, d in events if n == "final")
    assert "compacted" in final


# ============================================================
# Review endpoint (6)
# ============================================================


def _fake_reviewer_run_factory(verdict: str = "APPROVE"):
    review_text = (
        f"{verdict}\n\n"
        "The code looks clean and well-structured.\n"
        "- No obvious bugs.\n"
        "- Tests cover the main path.\n"
    )

    async def fake_run(self, session, on_step=None):
        session.add_assistant(content=review_text)
        step = AgentStep(
            response=LLMResponse(
                content=review_text,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 30,
                    "total_tokens": 80,
                },
                model="test/reviewer",
                raw=None,
            ),
            tool_results=[],
        )
        if on_step:
            await on_step(step, 0)
        return AgentRun(
            final_text=review_text,
            steps=[step],
            hit_limit=False,
            total_tokens=80,
        )

    return fake_run, review_text


def test_review_endpoint_happy_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, review_text = _fake_reviewer_run_factory("APPROVE")
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["persona"] == "reviewer"
    assert body["verdict"] == "APPROVE"
    assert body["review_text"] == review_text.strip()
    assert body["steps"] == 1
    assert body["tokens"] == 80
    assert body["duration_ms"] >= 0


def test_review_endpoint_request_changes_verdict(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, _ = _fake_reviewer_run_factory("REQUEST CHANGES")
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b"},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] == "REQUEST CHANGES"


def test_review_endpoint_approve_with_nits_verdict(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, _ = _fake_reviewer_run_factory("APPROVE WITH NITS")
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": "+line"},
    )
    assert r.status_code == 200
    assert r.json()["verdict"] == "APPROVE WITH NITS"


def test_review_stores_last_review_on_session(
    client: TestClient,
    state: AppState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, _ = _fake_reviewer_run_factory("APPROVE")
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": "diff content"},
    )

    session = state.session_store.get(sid)
    assert session is not None
    last_review = session.metadata.get("last_review")
    assert last_review is not None
    assert last_review["verdict"] == "APPROVE"
    assert "diff content" in last_review["diff"]
    assert "ts" in last_review


def test_review_does_not_pollute_main_chat(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run, _ = _fake_reviewer_run_factory("APPROVE")
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]
    before = client.get(f"/api/sessions/{sid}/messages").json()
    assert before["messages"] == []

    client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": "+something"},
    )

    after = client.get(f"/api/sessions/{sid}/messages").json()
    assert after["messages"] == []  # review stayed isolated


def test_review_unknown_session_404(client: TestClient) -> None:
    r = client.post(
        "/api/sessions/no-such/review",
        json={"diff": "anything"},
    )
    assert r.status_code == 404


def test_review_empty_diff_validation_error(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/review",
        json={"diff": ""},
    )
    assert r.status_code == 422


# ============================================================
# Solve endpoint (6)
# ============================================================


def test_solve_endpoint_happy_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /solve streams SSE and returns a final event with attempts/solved."""
    fake_run = _fake_run_factory()
    monkeypatch.setattr("codesmith.agent.Agent.run", fake_run)

    sid = client.post("/api/sessions").json()["session_id"]

    with client.stream(
        "POST",
        f"/api/sessions/{sid}/solve",
        json={"task": "write fibonacci function", "max_attempts": 2},
    ) as resp:
        assert resp.status_code == 200
        body = resp.read().decode("utf-8")

    events = _parse_sse_stream(body)
    names = [name for name, _ in events if name != "ping"]
    assert "start" in names
    assert "final" in names

    final = next(d for n, d in events if n == "final")
    assert "text" in final
    assert "attempts" in final
    assert isinstance(final["solved"], bool)
    assert "pending_count" in final


def test_solve_unknown_session_404(client: TestClient) -> None:
    r = client.post(
        "/api/sessions/nonexistent/solve",
        json={"task": "anything"},
    )
    assert r.status_code == 404


def test_solve_empty_task_validation_error(client: TestClient) -> None:
    sid = client.post("/api/sessions").json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid}/solve",
        json={"task": ""},
    )
    assert r.status_code == 422

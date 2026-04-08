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
    yield s


@pytest.fixture
def client(state: AppState, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A TestClient that bypasses the lifespan loader and injects our state."""

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
    assert body["primary_provider"] == "ollama"
    assert body["primary_model"] == "qwen2.5-coder:7b"
    assert body["fallbacks"] == ["anthropic/claude-sonnet-4-6"]
    assert body["sandbox_image"] == "codesmith-sandbox:test"
    assert body["memory_enabled"] is False
    assert body["web_search_enabled"] is False


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
    """Build a fake Agent.run that fires on_step twice and returns a final."""

    async def fake_run(self, session, on_step=None):
        # Step 1: tool call
        step1 = AgentStep(
            response=LLMResponse(
                content="thinking...",
                tool_calls=[
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": "execute_python",
                            "arguments": json.dumps({"code": "print(1)"}),
                        },
                    }
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                model="test/test",
                raw=None,
            ),
            tool_results=[],
        )
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
    assert names.count("step") == 2
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

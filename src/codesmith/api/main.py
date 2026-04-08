"""FastAPI server hosting the Codesmith Web UI and HTTP API.

Endpoints:
    GET  /                          — Web UI (single-page chat interface)
    GET  /health                    — liveness probe
    GET  /api/info                  — config summary (no secrets)
    POST /api/sessions              — create a new chat session
    GET  /api/sessions              — list active sessions
    DELETE /api/sessions/{sid}      — drop a session and its workspace
    POST /api/sessions/{sid}/chat   — send a message, streamed via SSE

The chat endpoint streams server-sent events as the agent works:
    event: start       — agent has accepted the request
    event: step        — one LLM iteration finished (tool_calls + text)
    event: final       — final assistant text + stats
    event: error       — fatal error, stream ends

This is the Phase 3 entry point — it builds on top of the same Agent /
Session / Tool stack used by the CLI, no business logic duplicated.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from codesmith import __version__
from codesmith.agent import Agent, AgentStep
from codesmith.config import Config, load_config
from codesmith.llm import LLMRouter
from codesmith.session import Session
from codesmith.tools.filesystem import default_filesystem_tools
from codesmith.tools.registry import ToolRegistry
from codesmith.tools.sandbox import SandboxTool

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ============================================================
# App state — built once at startup, shared across requests.
# ============================================================


class AppState:
    """Process-wide state for the API server.

    Holds the loaded config, the LLM router, a tool registry, and an
    in-memory dict of active sessions. The sandbox image and Docker
    client are reused — building per-request would be wasteful.

    For Phase 3 multi-user this becomes per-tenant; for the local
    single-user UI it stays a flat dict.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.router = LLMRouter(primary=config.llm, fallbacks=config.llm_fallbacks)

        self.registry = ToolRegistry()
        self.registry.register(SandboxTool(config.sandbox))
        if config.tools.filesystem.enabled:
            for t in default_filesystem_tools():
                self.registry.register(t)

        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    def make_agent(self) -> Agent:
        return Agent(config=self.config, llm=self.router, registry=self.registry)

    async def create_session(self) -> Session:
        async with self._lock:
            session = Session.create(workspace_root=self.config.tools.filesystem.workspace_root)
            self.sessions[session.session_id] = session
            return session

    async def drop_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                return False
            try:
                session.destroy()
            except Exception as e:  # noqa: BLE001
                log.warning("session.destroy failed for %s: %s", session_id, e)
            return True

    def get_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError as e:
            raise HTTPException(status_code=404, detail="session not found") from e


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the Config + AppState once. Reload via env CODESMITH_CONFIG."""
    cfg_path = os.environ.get("CODESMITH_CONFIG", "config.yaml")
    config = load_config(Path(cfg_path))
    app.state.codesmith = AppState(config)
    log.info(
        "codesmith api ready — primary=%s/%s sessions=in-memory",
        config.llm.provider,
        config.llm.model,
    )
    try:
        yield
    finally:
        # Best-effort cleanup; we don't want shutdown to fail noisily.
        state: AppState | None = getattr(app.state, "codesmith", None)
        if state is not None:
            for sid in list(state.sessions.keys()):
                with contextlib.suppress(Exception):
                    await state.drop_session(sid)


app = FastAPI(
    title="Codesmith API",
    version=__version__,
    description="Hybrid AI coder agent — HTTP interface and Web UI.",
    lifespan=_lifespan,
)


def _state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "codesmith", None)
    if state is None:
        raise HTTPException(status_code=503, detail="codesmith state not initialised")
    return state


# ============================================================
# Schemas
# ============================================================


class CreateSessionResponse(BaseModel):
    session_id: str
    workspace: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_iterations: int | None = Field(default=None, ge=1, le=50)


class SessionSummary(BaseModel):
    session_id: str
    messages: int
    workspace: str


class InfoResponse(BaseModel):
    version: str
    primary_provider: str
    primary_model: str
    has_api_key: bool
    fallbacks: list[str]
    sandbox_image: str
    sandbox_memory_mb: int
    sandbox_cpus: float
    sandbox_timeout_s: int
    memory_enabled: bool
    web_search_enabled: bool


# ============================================================
# Endpoints
# ============================================================


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/info", response_model=InfoResponse)
async def info(request: Request) -> InfoResponse:
    cfg = _state(request).config
    return InfoResponse(
        version=__version__,
        primary_provider=cfg.llm.provider,
        primary_model=cfg.llm.model,
        has_api_key=bool(cfg.llm.api_key),
        fallbacks=[f"{f.provider}/{f.model}" for f in cfg.llm_fallbacks],
        sandbox_image=cfg.sandbox.image,
        sandbox_memory_mb=cfg.sandbox.memory_mb,
        sandbox_cpus=cfg.sandbox.cpus,
        sandbox_timeout_s=cfg.sandbox.timeout_seconds,
        memory_enabled=cfg.memory.enabled,
        web_search_enabled=cfg.tools.web_search.enabled,
    )


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(request: Request) -> CreateSessionResponse:
    state = _state(request)
    session = await state.create_session()
    return CreateSessionResponse(
        session_id=session.session_id,
        workspace=str(session.workspace_dir),
    )


@app.get("/api/sessions", response_model=list[SessionSummary])
async def list_sessions(request: Request) -> list[SessionSummary]:
    state = _state(request)
    return [
        SessionSummary(
            session_id=s.session_id,
            messages=len(s.messages),
            workspace=str(s.workspace_dir),
        )
        for s in state.sessions.values()
    ]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, bool]:
    state = _state(request)
    ok = await state.drop_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    payload: ChatRequest,
    request: Request,
) -> EventSourceResponse:
    """Send one user message; stream agent progress as Server-Sent Events.

    Event types emitted in order:
        start  → {message}
        step   → {iteration, assistant_text, tool_calls: [{name, args}],
                  tool_summaries: [str]}  (one per agent iteration)
        final  → {text, steps, tokens, hit_limit}
        error  → {error}      (terminal; stream ends)

    The connection closes after `final` or `error`.
    """
    state = _state(request)
    session = state.get_session(session_id)
    session.add_user(payload.message)

    agent = state.make_agent()
    if payload.max_iterations:
        agent.max_iterations = payload.max_iterations

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_step(step: AgentStep, iteration: int) -> None:
        tool_calls_summary = []
        for tc in step.response.tool_calls or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls_summary.append(
                {"name": tc["function"]["name"], "args_keys": list(args.keys())}
            )
        tool_summaries = [
            _summarize_tool_result(tcid, tr) for tcid, tr in step.tool_results
        ]
        await queue.put(
            {
                "event": "step",
                "data": {
                    "iteration": iteration + 1,
                    "assistant_text": step.response.content or "",
                    "tool_calls": tool_calls_summary,
                    "tool_results": tool_summaries,
                    "tokens": step.response.usage.get("total_tokens", 0),
                },
            }
        )

    async def runner() -> None:
        try:
            await queue.put(
                {"event": "start", "data": {"message": payload.message}}
            )
            run = await agent.run(session, on_step=on_step)
            await queue.put(
                {
                    "event": "final",
                    "data": {
                        "text": run.final_text,
                        "steps": len(run.steps),
                        "tokens": run.total_tokens,
                        "hit_limit": run.hit_limit,
                    },
                }
            )
        except Exception as e:  # noqa: BLE001
            log.exception("agent run failed")
            await queue.put({"event": "error", "data": {"error": f"{type(e).__name__}: {e}"}})
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(runner())

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            while True:
                if await request.is_disconnected():
                    task.cancel()
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    # Heartbeat keeps proxies happy and lets us notice disconnects.
                    yield {"event": "ping", "data": "{}"}
                    continue
                if item is None:
                    return
                yield {"event": item["event"], "data": json.dumps(item["data"])}
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(event_stream())


# ============================================================
# Static UI
# ============================================================


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"error": "Web UI not found", "expected_at": str(index_path)},
            status_code=500,
        )
    return FileResponse(index_path, media_type="text/html")


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    # Avoid 404 noise in dev console; we don't ship a favicon yet.
    return JSONResponse({}, status_code=204)


# ============================================================
# Helpers
# ============================================================


def _summarize_tool_result(tool_call_id: str, result: Any) -> dict[str, Any]:
    """Compact, JSON-safe representation of a ToolResult for the UI."""
    content = getattr(result, "to_llm_content", lambda: str(result))()
    if not isinstance(content, str):
        content = str(content)
    truncated = content if len(content) <= 2000 else content[:2000] + "\n…(truncated)"
    return {
        "tool_call_id": tool_call_id,
        "ok": getattr(result, "ok", True),
        "preview": truncated,
    }

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
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from codesmith import __version__
from codesmith.agent import Agent, AgentStep
from codesmith.config import Config, load_config
from codesmith.llm import LLMError, LLMRouter
from codesmith.model_selection import (
    UnknownModelProfileError,
    default_model_profile_key,
    list_installed_ollama_models,
    list_model_profiles,
    resolve_model_profile,
)
from codesmith.run_logger import RunLogger, build_run_record
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

        self.registry = ToolRegistry()
        self.registry.register(SandboxTool(config.sandbox))
        if config.tools.filesystem.enabled:
            for t in default_filesystem_tools():
                self.registry.register(t)

        self.sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.run_logger = RunLogger()

    def make_agent(self, model_profile: str | None = None) -> Agent:
        resolved = resolve_model_profile(
            self.config,
            model_profile,
            list_installed_ollama_models(self.config),
        )
        router = LLMRouter(primary=resolved.primary, fallbacks=resolved.fallbacks)
        return Agent(config=self.config, llm=router, registry=self.registry)

    async def create_session(self, model_profile: str | None = None) -> Session:
        async with self._lock:
            session = Session.create(workspace_root=self.config.tools.filesystem.workspace_root)
            if model_profile:
                session.metadata["model_profile"] = model_profile
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
    resolved = resolve_model_profile(
        config,
        default_model_profile_key(config),
        list_installed_ollama_models(config),
    )
    log.info(
        "codesmith api ready - default_profile=%s primary=%s/%s sessions=in-memory",
        resolved.key,
        resolved.primary.provider,
        resolved.primary.model,
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


class CreateSessionRequest(BaseModel):
    model_profile: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    workspace: str
    model_profile: str
    resolved_provider: str
    resolved_model: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_iterations: int | None = Field(default=None, ge=1, le=50)
    model_profile: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    messages: int
    workspace: str


class InfoResponse(BaseModel):
    version: str
    default_model_profile: str
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


class ModelProfileResponse(BaseModel):
    key: str
    label: str
    description: str
    primary_provider: str
    primary_model: str
    fallbacks: list[str]
    available: bool
    availability: str
    source: str
    recommended: bool = False


class ModelsResponse(BaseModel):
    default_model_profile: str
    profiles: list[ModelProfileResponse]
    installed_ollama_models: list[str]


class RunLogEntry(BaseModel):
    ts: str
    session_id: str
    profile: str
    provider: str
    model: str
    steps: int
    tokens: int
    hit_limit: bool
    forced_final: bool
    duration_ms: int
    error: str | None = None
    user_message: str = ""
    caller: str = ""


class RunsResponse(BaseModel):
    runs: list[RunLogEntry]
    path: str


# ============================================================
# Endpoints
# ============================================================


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/info", response_model=InfoResponse)
async def info(request: Request) -> InfoResponse:
    state = _state(request)
    cfg = state.config
    resolved = resolve_model_profile(
        cfg,
        default_model_profile_key(cfg),
        list_installed_ollama_models(cfg),
    )
    return InfoResponse(
        version=__version__,
        default_model_profile=resolved.key,
        primary_provider=resolved.primary.provider,
        primary_model=resolved.primary.model,
        has_api_key=bool(resolved.primary.api_key),
        fallbacks=[f"{f.provider}/{f.model}" for f in resolved.fallbacks],
        sandbox_image=cfg.sandbox.image,
        sandbox_memory_mb=cfg.sandbox.memory_mb,
        sandbox_cpus=cfg.sandbox.cpus,
        sandbox_timeout_s=cfg.sandbox.timeout_seconds,
        memory_enabled=cfg.memory.enabled,
        web_search_enabled=cfg.tools.web_search.enabled,
    )


@app.get("/api/runs", response_model=RunsResponse)
async def list_runs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> RunsResponse:
    """Return the last N completed agent runs from the JSONL log.

    Backed by the same RunLogger the CLI `codesmith logs` reads — this
    endpoint just exposes it to the Web UI observability panel.
    """
    state = _state(request)
    records = state.run_logger.tail(limit)
    entries = [
        RunLogEntry(
            ts=r.ts,
            session_id=r.session_id,
            profile=r.profile,
            provider=r.provider,
            model=r.model,
            steps=r.steps,
            tokens=r.tokens,
            hit_limit=r.hit_limit,
            forced_final=r.forced_final,
            duration_ms=r.duration_ms,
            error=r.error,
            user_message=r.user_message,
            caller=str(r.extra.get("caller", "")),
        )
        for r in records
    ]
    # Newest first — matches what a user scanning the panel expects.
    entries.reverse()
    return RunsResponse(runs=entries, path=str(state.run_logger.path))


@app.get("/api/models", response_model=ModelsResponse)
async def models(request: Request) -> ModelsResponse:
    state = _state(request)
    installed = list_installed_ollama_models(state.config)
    profiles = [
        ModelProfileResponse(**summary.__dict__)
        for summary in list_model_profiles(state.config, installed)
    ]
    return ModelsResponse(
        default_model_profile=default_model_profile_key(state.config),
        profiles=profiles,
        installed_ollama_models=installed,
    )


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(
    request: Request,
    payload: Annotated[CreateSessionRequest | None, Body()] = None,
) -> CreateSessionResponse:
    state = _state(request)
    selection = payload.model_profile if payload else None
    try:
        resolved = resolve_model_profile(
            state.config,
            selection,
            list_installed_ollama_models(state.config),
        )
    except UnknownModelProfileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    session = await state.create_session(resolved.key)
    return CreateSessionResponse(
        session_id=session.session_id,
        workspace=str(session.workspace_dir),
        model_profile=resolved.key,
        resolved_provider=resolved.primary.provider,
        resolved_model=resolved.primary.model,
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
    selected_profile = payload.model_profile or session.metadata.get("model_profile")
    try:
        resolved_profile = resolve_model_profile(
            state.config,
            selected_profile,
            list_installed_ollama_models(state.config),
        )
    except UnknownModelProfileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    session.metadata["model_profile"] = resolved_profile.key
    session.add_user(payload.message)

    agent = state.make_agent(resolved_profile.key)
    if payload.max_iterations:
        agent.max_iterations = payload.max_iterations

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_step(step: AgentStep, iteration: int) -> None:
        if not step.response.tool_calls and not step.tool_results:
            # Plain assistant text is surfaced by the terminal `final` event;
            # skip the duplicate debug step in the Web UI.
            return
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
        started_at = time.monotonic()
        log_error: str | None = None
        run = None
        try:
            await queue.put(
                {
                    "event": "start",
                    "data": {
                        "message": payload.message,
                        "model_profile": resolved_profile.key,
                        "provider": resolved_profile.primary.provider,
                        "model": resolved_profile.primary.model,
                    },
                }
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
                        "forced_final": run.forced_final,
                    },
                }
            )
        except LLMError as e:
            # All providers failed. Surface a friendly hint instead of the
            # full multi-provider traceback.
            log.warning("LLM router exhausted: %s", e)
            log_error = f"LLMError: {e}"
            await queue.put(
                {
                    "event": "error",
                    "data": {
                        "error": (
                            "LLM backend unavailable.\n\n"
                            "Run `codesmith info` for live health checks, or "
                            "`codesmith pull-model` if Ollama is missing the "
                            "configured model."
                        ),
                        "details": str(e),
                    },
                }
            )
        except Exception as e:  # noqa: BLE001
            log.exception("agent run failed")
            log_error = f"{type(e).__name__}: {e}"
            await queue.put(
                {
                    "event": "error",
                    "data": {"error": log_error},
                }
            )
        finally:
            duration_ms = int((time.monotonic() - started_at) * 1000)
            with contextlib.suppress(Exception):
                state.run_logger.record(
                    build_run_record(
                        session_id=session.session_id,
                        profile_key=resolved_profile.key,
                        provider=resolved_profile.primary.provider,
                        model=resolved_profile.primary.model,
                        user_message=payload.message,
                        steps=len(run.steps) if run else 0,
                        tokens=run.total_tokens if run else 0,
                        hit_limit=bool(run.hit_limit) if run else False,
                        duration_ms=duration_ms,
                        forced_final=bool(run.forced_final) if run else False,
                        error=log_error,
                        extra={"caller": "web"},
                    )
                )
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
async def favicon() -> Response:
    # Avoid 404 noise in dev console; we don't ship a favicon yet.
    # 204 No Content REQUIRES an empty body, otherwise uvicorn raises
    # "Response content longer than Content-Length".
    return Response(status_code=204)


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

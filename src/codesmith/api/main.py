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
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from codesmith import __version__
from codesmith.agent import DEFAULT_SYSTEM_PROMPT, Agent, AgentStep
from codesmith.config import Config, load_config
from codesmith.llm import LLMError, LLMRouter
from codesmith.model_selection import (
    UnknownModelProfileError,
    default_model_profile_key,
    list_installed_ollama_models,
    list_model_profiles,
    resolve_model_profile,
)
from codesmith.pending_changes import (
    apply_change,
    is_auto_approve,
    list_pending,
    reject_change,
    set_auto_approve,
)
from codesmith.pending_changes import list_pending as list_pending_changes
from codesmith.personas import ARCHITECT, CODER, Persona, list_personas
from codesmith.run_logger import RunLogger, build_run_record
from codesmith.session import Session
from codesmith.session_store import (
    InMemorySessionStore,
    SQLiteSessionStore,
)
from codesmith.tools.filesystem import default_filesystem_tools
from codesmith.tools.registry import ToolRegistry
from codesmith.tools.sandbox import SandboxTool
from codesmith.workspace_snapshot import bootstrap_system_prompt_addition

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ============================================================
# App state — built once at startup, shared across requests.
# ============================================================


class AppState:
    """Process-wide state for the API server.

    Holds the loaded config, the tool registry, the pluggable session
    store (SQLite or in-memory per config.sessions.backend), the run
    logger, and a mapping of inflight agent tasks so the abort endpoint
    can actually cancel a running chat.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

        self.registry = ToolRegistry()
        self.registry.register(SandboxTool(config.sandbox))
        if config.tools.filesystem.enabled:
            for t in default_filesystem_tools():
                self.registry.register(t)

        workspace_root = config.tools.filesystem.workspace_root
        if config.sessions.backend == "sqlite":
            self.session_store: InMemorySessionStore | SQLiteSessionStore = (
                SQLiteSessionStore(
                    db_path=config.sessions.path,
                    workspace_root=workspace_root,
                )
            )
        else:
            self.session_store = InMemorySessionStore(workspace_root=workspace_root)

        self._lock = asyncio.Lock()
        self.run_logger = RunLogger()
        # session_id -> asyncio.Task currently running agent.run() for that
        # session. Populated on chat start, cleared on chat finish/error.
        # Abort endpoint calls .cancel() on the task.
        self._inflight: dict[str, asyncio.Task[Any]] = {}

    def make_agent(
        self,
        model_profile: str | None = None,
        persona: Persona | None = None,
    ) -> Agent:
        resolved = resolve_model_profile(
            self.config,
            model_profile,
            list_installed_ollama_models(self.config),
        )
        router = LLMRouter(primary=resolved.primary, fallbacks=resolved.fallbacks)
        # When the caller passed a tool-less persona (e.g. ARCHITECT),
        # give the agent an EMPTY registry so the LLM cannot see any
        # tools in its prompt — much cheaper than patching the agent
        # loop to ignore tools on the fly.
        registry = self.registry
        if persona is not None and persona.no_tools:
            registry = ToolRegistry()
        prompt = persona.system_prompt if persona is not None else DEFAULT_SYSTEM_PROMPT
        return Agent(
            config=self.config,
            llm=router,
            registry=registry,
            system_prompt=prompt,
        )

    async def create_session(self, model_profile: str | None = None) -> Session:
        async with self._lock:
            return self.session_store.create(model_profile=model_profile)

    def save_session(self, session: Session) -> None:
        try:
            self.session_store.save(session)
        except Exception as e:  # noqa: BLE001
            log.warning("session persist failed for %s: %s", session.session_id, e)

    async def drop_session(self, session_id: str) -> bool:
        async with self._lock:
            # Cancel any in-flight chat for this session before dropping
            # its state — otherwise the cancelled task would try to save
            # back into a non-existent row.
            task = self._inflight.pop(session_id, None)
            if task is not None and not task.done():
                task.cancel()
            return self.session_store.delete(session_id)

    def get_session(self, session_id: str) -> Session:
        session = self.session_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    def register_inflight(self, session_id: str, task: asyncio.Task[Any]) -> None:
        self._inflight[session_id] = task

    def clear_inflight(self, session_id: str, task: asyncio.Task[Any]) -> None:
        current = self._inflight.get(session_id)
        if current is task:
            self._inflight.pop(session_id, None)

    def abort_inflight(self, session_id: str) -> bool:
        task = self._inflight.get(session_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True


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
            for sid, task in list(state._inflight.items()):
                if not task.done():
                    task.cancel()
                state._inflight.pop(sid, None)


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
    title: str = ""
    model_profile: str | None = None
    updated_at: str = ""


class SessionMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    workspace: str
    model_profile: str | None
    messages: list[SessionMessage]


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


class PendingChangeEntry(BaseModel):
    idx: int
    kind: str
    path: str
    state: str
    created_at: str
    applied_at: str | None = None
    summary: str = ""
    diff: str = ""
    before: str = ""
    after: str = ""
    files: list[dict[str, Any]] | None = None
    error: str | None = None


class PendingChangesResponse(BaseModel):
    session_id: str
    auto_approve: bool
    changes: list[PendingChangeEntry]


class AutoApproveRequest(BaseModel):
    enabled: bool


class AutoApproveResponse(BaseModel):
    session_id: str
    auto_approve: bool


class PersonaResponse(BaseModel):
    key: str
    label: str
    description: str
    no_tools: bool


class PersonasListResponse(BaseModel):
    personas: list[PersonaResponse]


class PlanRequest(BaseModel):
    task: str = Field(..., min_length=1)
    model_profile: str | None = None


class PlanResponse(BaseModel):
    session_id: str
    persona: str
    provider: str
    model: str
    plan_text: str
    steps: int
    tokens: int
    duration_ms: int


class ExecutePlanRequest(BaseModel):
    plan_text: str | None = None
    model_profile: str | None = None
    max_iterations: int | None = Field(default=None, ge=1, le=50)


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


@app.get("/api/personas", response_model=PersonasListResponse)
async def personas_list() -> PersonasListResponse:
    """Return all registered personas — used by the UI palette."""
    items = [
        PersonaResponse(
            key=p.key,
            label=p.label,
            description=p.description,
            no_tools=p.no_tools,
        )
        for p in list_personas()
    ]
    return PersonasListResponse(personas=items)


@app.post("/api/sessions/{session_id}/plan", response_model=PlanResponse)
async def plan(
    session_id: str,
    payload: PlanRequest,
    request: Request,
) -> PlanResponse:
    """Run the ARCHITECT persona on a task and return a plan.

    This is a SHORT, one-shot call (no SSE): the architect has no tools
    and is capped to a tight iteration budget, so it finishes in one or
    two LLM calls. The resulting plan is stored on the session under
    metadata["last_plan"] so the UI / execute-plan endpoint can pick it
    up later, but it is NOT added to the main chat history — planning
    and chatting stay on separate tracks.
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

    # Build an ISOLATED session for planning so the architect's
    # conversation does not pollute the main chat history.
    plan_session = Session(
        session_id=session.session_id + "::plan",
        messages=[],
        workspace_dir=session.workspace_dir,
        metadata={"model_profile": resolved_profile.key, "role": "planner"},
    )
    plan_session.add_system(ARCHITECT.system_prompt)
    plan_session.add_user(payload.task)

    agent = state.make_agent(resolved_profile.key, persona=ARCHITECT)
    # Plans are short: two iterations is usually one assistant turn.
    agent.max_iterations = 3

    started_at = time.monotonic()
    log_error: str | None = None
    run = None
    try:
        run = await agent.run(plan_session)
    except LLMError as e:
        log.warning("plan LLM error: %s", e)
        raise HTTPException(
            status_code=503,
            detail="LLM backend unavailable for planning. Run codesmith info.",
        ) from e
    except Exception as e:  # noqa: BLE001
        log.exception("plan failed")
        log_error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=log_error) from e
    finally:
        duration_ms = int((time.monotonic() - started_at) * 1000)
        with contextlib.suppress(Exception):
            state.run_logger.record(
                build_run_record(
                    session_id=session.session_id,
                    profile_key=resolved_profile.key,
                    provider=resolved_profile.primary.provider,
                    model=resolved_profile.primary.model,
                    user_message=payload.task,
                    steps=len(run.steps) if run else 0,
                    tokens=run.total_tokens if run else 0,
                    hit_limit=bool(run.hit_limit) if run else False,
                    duration_ms=duration_ms,
                    forced_final=bool(run.forced_final) if run else False,
                    error=log_error,
                    extra={"caller": "web.plan", "persona": "architect"},
                )
            )

    plan_text = (run.final_text or "").strip()
    session.metadata["last_plan"] = {
        "task": payload.task,
        "text": plan_text,
        "provider": resolved_profile.primary.provider,
        "model": resolved_profile.primary.model,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    state.save_session(session)

    return PlanResponse(
        session_id=session.session_id,
        persona=ARCHITECT.key,
        provider=resolved_profile.primary.provider,
        model=resolved_profile.primary.model,
        plan_text=plan_text,
        steps=len(run.steps),
        tokens=run.total_tokens,
        duration_ms=int((time.monotonic() - started_at) * 1000),
    )


@app.post("/api/sessions/{session_id}/execute-plan")
async def execute_plan(
    session_id: str,
    request: Request,
    payload: Annotated[ExecutePlanRequest | None, Body()] = None,
) -> EventSourceResponse:
    """Execute a plan using the CODER persona, streamed via SSE.

    If ``plan_text`` is not provided in the request body, falls back to
    ``session.metadata["last_plan"]["text"]`` (set by the /plan endpoint).
    Uses the same SSE protocol as /chat so the UI can render it
    identically.
    """
    state = _state(request)
    session = state.get_session(session_id)

    # Resolve plan text.
    plan_text: str | None = None
    if payload and payload.plan_text:
        plan_text = payload.plan_text
    else:
        last_plan = session.metadata.get("last_plan")
        if isinstance(last_plan, dict):
            plan_text = last_plan.get("text")
    if not plan_text or not plan_text.strip():
        raise HTTPException(
            status_code=404,
            detail="No plan found. Run /plan first or provide plan_text.",
        )

    # Resolve model profile.
    selected_profile = (
        (payload.model_profile if payload else None)
        or session.metadata.get("model_profile")
    )
    try:
        resolved_profile = resolve_model_profile(
            state.config,
            selected_profile,
            list_installed_ollama_models(state.config),
        )
    except UnknownModelProfileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Bootstrap workspace snapshot on first turn (same as /chat).
    is_first_turn = not any(m.get("role") == "user" for m in session.messages)
    if is_first_turn:
        snapshot_block = bootstrap_system_prompt_addition(session.workspace_dir)
        base_prompt = CODER.system_prompt
        if snapshot_block:
            base_prompt = base_prompt + "\n\n" + snapshot_block
        session.add_system(base_prompt)

    # Add the plan as a user message for the coder to execute.
    user_message = (
        "Execute this plan step by step. Do not deviate. "
        "Use tools to implement each step.\n\n" + plan_text
    )
    session.add_user(user_message)

    agent = state.make_agent(resolved_profile.key, persona=CODER)
    if payload and payload.max_iterations:
        agent.max_iterations = payload.max_iterations

    return _build_sse_response(
        state=state,
        session=session,
        agent=agent,
        request=request,
        user_message=user_message,
        profile_key=resolved_profile.key,
        provider=resolved_profile.primary.provider,
        model=resolved_profile.primary.model,
        caller="web.execute",
        extra_log={"persona": "coder"},
    )


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
async def list_sessions(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[SessionSummary]:
    state = _state(request)
    rows = state.session_store.list(limit=limit)
    return [
        SessionSummary(
            session_id=row.session_id,
            messages=row.message_count,
            workspace=row.workspace,
            title=row.title,
            model_profile=row.model_profile,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


@app.get("/api/sessions/{session_id}/messages", response_model=SessionHistoryResponse)
async def get_session_messages(
    session_id: str, request: Request
) -> SessionHistoryResponse:
    """Return the full stored message history for a session.

    Used by the Web UI on page load to restore the chat after a
    refresh, and by anything that needs to render a past transcript.
    """
    state = _state(request)
    session = state.get_session(session_id)
    messages: list[SessionMessage] = []
    for raw in session.messages:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "")
        if role == "system":
            # Hide the framework system prompt from the UI.
            continue
        content_field = raw.get("content")
        content = content_field if isinstance(content_field, str) else ""
        tool_calls_field = raw.get("tool_calls")
        tool_calls_value = (
            tool_calls_field
            if isinstance(tool_calls_field, list) and tool_calls_field
            else None
        )
        tool_call_id_field = raw.get("tool_call_id")
        tool_call_id = (
            tool_call_id_field if isinstance(tool_call_id_field, str) else None
        )
        messages.append(
            SessionMessage(
                role=role,
                content=content,
                tool_calls=tool_calls_value,
                tool_call_id=tool_call_id,
            )
        )
    return SessionHistoryResponse(
        session_id=session.session_id,
        workspace=str(session.workspace_dir),
        model_profile=session.metadata.get("model_profile"),
        messages=messages,
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, bool]:
    state = _state(request)
    ok = await state.drop_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="session not found")
    return {"deleted": True}


@app.get(
    "/api/sessions/{session_id}/pending-changes",
    response_model=PendingChangesResponse,
)
async def get_pending_changes(
    session_id: str, request: Request
) -> PendingChangesResponse:
    """List every pending change parked on this session.

    Includes applied / rejected / failed entries in addition to live
    pending ones so the UI can render a full timeline, not just the
    queue head.
    """
    state = _state(request)
    session = state.get_session(session_id)
    entries = [
        PendingChangeEntry(
            idx=int(item.get("idx", 0)),
            kind=str(item.get("kind", "")),
            path=str(item.get("path", "")),
            state=str(item.get("state", "")),
            created_at=str(item.get("created_at", "")),
            applied_at=item.get("applied_at"),
            summary=str(item.get("summary", "")),
            diff=str(item.get("diff", "")),
            before=str(item.get("before", "")),
            after=str(item.get("after", "")),
            files=item.get("files") if isinstance(item.get("files"), list) else None,
            error=item.get("error") if isinstance(item.get("error"), str) else None,
        )
        for item in list_pending(session)
    ]
    return PendingChangesResponse(
        session_id=session.session_id,
        auto_approve=is_auto_approve(session),
        changes=entries,
    )


@app.post(
    "/api/sessions/{session_id}/pending-changes/{idx}/approve",
    response_model=PendingChangeEntry,
)
async def approve_pending_change(
    session_id: str, idx: int, request: Request
) -> PendingChangeEntry:
    state = _state(request)
    session = state.get_session(session_id)
    try:
        updated = apply_change(session, idx, workspace_dir=session.workspace_dir)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"apply failed: {e}") from e
    state.save_session(session)
    return PendingChangeEntry(
        idx=int(updated.get("idx", 0)),
        kind=str(updated.get("kind", "")),
        path=str(updated.get("path", "")),
        state=str(updated.get("state", "")),
        created_at=str(updated.get("created_at", "")),
        applied_at=updated.get("applied_at"),
        summary=str(updated.get("summary", "")),
        diff=str(updated.get("diff", "")),
        before=str(updated.get("before", "")),
        after=str(updated.get("after", "")),
        files=updated.get("files") if isinstance(updated.get("files"), list) else None,
        error=updated.get("error") if isinstance(updated.get("error"), str) else None,
    )


@app.post(
    "/api/sessions/{session_id}/pending-changes/{idx}/reject",
    response_model=PendingChangeEntry,
)
async def reject_pending_change(
    session_id: str, idx: int, request: Request
) -> PendingChangeEntry:
    state = _state(request)
    session = state.get_session(session_id)
    updated = reject_change(session, idx)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"pending change {idx} not found")
    state.save_session(session)
    return PendingChangeEntry(
        idx=int(updated.get("idx", 0)),
        kind=str(updated.get("kind", "")),
        path=str(updated.get("path", "")),
        state=str(updated.get("state", "")),
        created_at=str(updated.get("created_at", "")),
        applied_at=updated.get("applied_at"),
        summary=str(updated.get("summary", "")),
        diff=str(updated.get("diff", "")),
        before=str(updated.get("before", "")),
        after=str(updated.get("after", "")),
        files=updated.get("files") if isinstance(updated.get("files"), list) else None,
        error=updated.get("error") if isinstance(updated.get("error"), str) else None,
    )


@app.post(
    "/api/sessions/{session_id}/auto-approve",
    response_model=AutoApproveResponse,
)
async def set_session_auto_approve(
    session_id: str,
    payload: AutoApproveRequest,
    request: Request,
) -> AutoApproveResponse:
    state = _state(request)
    session = state.get_session(session_id)
    set_auto_approve(session, payload.enabled)
    state.save_session(session)
    return AutoApproveResponse(
        session_id=session.session_id,
        auto_approve=is_auto_approve(session),
    )


@app.delete("/api/sessions/{session_id}/chat")
async def abort_chat(session_id: str, request: Request) -> dict[str, bool]:
    """Cancel the in-flight chat run for this session, if any.

    Called by the Web UI when the user hits Esc mid-response. Leaves
    the session and its workspace untouched so the conversation can
    continue with a new prompt.
    """
    state = _state(request)
    if state.session_store.get(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    cancelled = state.abort_inflight(session_id)
    return {"cancelled": cancelled}


def _build_sse_response(
    *,
    state: AppState,
    session: Session,
    agent: Agent,
    request: Request,
    user_message: str,
    profile_key: str,
    provider: str,
    model: str,
    caller: str = "web",
    extra_log: dict[str, Any] | None = None,
) -> EventSourceResponse:
    """Shared SSE streaming logic for /chat and /execute-plan.

    Runs the agent loop in a background task, streams step/final/error
    events through an asyncio.Queue, and handles cancellation, logging,
    and session persistence.
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_step(step: AgentStep, iteration: int) -> None:
        if not step.response.tool_calls and not step.tool_results:
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
        # Count pending changes so the UI knows when to render diff cards.
        pending_count = sum(
            1
            for c in list_pending_changes(session)
            if c.get("state") == "pending"
        )
        await queue.put(
            {
                "event": "step",
                "data": {
                    "iteration": iteration + 1,
                    "assistant_text": step.response.content or "",
                    "tool_calls": tool_calls_summary,
                    "tool_results": tool_summaries,
                    "tokens": step.response.usage.get("total_tokens", 0),
                    "pending_count": pending_count,
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
                        "message": user_message,
                        "model_profile": profile_key,
                        "provider": provider,
                        "model": model,
                    },
                }
            )
            run = await agent.run(session, on_step=on_step)
            pending_count = sum(
                1
                for c in list_pending_changes(session)
                if c.get("state") == "pending"
            )
            await queue.put(
                {
                    "event": "final",
                    "data": {
                        "text": run.final_text,
                        "steps": len(run.steps),
                        "tokens": run.total_tokens,
                        "hit_limit": run.hit_limit,
                        "forced_final": run.forced_final,
                        "compacted": run.compacted_count,
                        "pending_count": pending_count,
                    },
                }
            )
        except asyncio.CancelledError:
            log.info("chat cancelled for session %s", session.session_id)
            log_error = "cancelled"
            await queue.put(
                {
                    "event": "error",
                    "data": {"error": "Chat cancelled by user.", "cancelled": True},
                }
            )
            raise
        except LLMError as e:
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
                state.save_session(session)
            with contextlib.suppress(Exception):
                state.run_logger.record(
                    build_run_record(
                        session_id=session.session_id,
                        profile_key=profile_key,
                        provider=provider,
                        model=model,
                        user_message=user_message,
                        steps=len(run.steps) if run else 0,
                        tokens=run.total_tokens if run else 0,
                        hit_limit=bool(run.hit_limit) if run else False,
                        duration_ms=duration_ms,
                        forced_final=bool(run.forced_final) if run else False,
                        error=log_error,
                        extra={"caller": caller, **(extra_log or {})},
                    )
                )
            await queue.put(None)

    task = asyncio.create_task(runner())
    state.register_inflight(session.session_id, task)
    task.add_done_callback(lambda t: state.clear_inflight(session.session_id, t))

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            while True:
                if await request.is_disconnected():
                    task.cancel()
                    return
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                if item is None:
                    return
                yield {"event": item["event"], "data": json.dumps(item["data"])}
        finally:
            if not task.done():
                task.cancel()

    return EventSourceResponse(event_stream())


@app.post("/api/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    payload: ChatRequest,
    request: Request,
) -> EventSourceResponse:
    """Send one user message; stream agent progress as Server-Sent Events.

    Event types emitted in order:
        start  → {message}
        step   → {iteration, assistant_text, tool_calls, tool_results,
                  tokens, pending_count}  (one per agent iteration)
        final  → {text, steps, tokens, hit_limit, compacted, pending_count}
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

    is_first_turn = not any(m.get("role") == "user" for m in session.messages)
    if is_first_turn:
        snapshot_block = bootstrap_system_prompt_addition(session.workspace_dir)
        if snapshot_block:
            session.add_system(DEFAULT_SYSTEM_PROMPT + "\n\n" + snapshot_block)

    session.add_user(payload.message)

    agent = state.make_agent(resolved_profile.key)
    if payload.max_iterations:
        agent.max_iterations = payload.max_iterations

    return _build_sse_response(
        state=state,
        session=session,
        agent=agent,
        request=request,
        user_message=payload.message,
        profile_key=resolved_profile.key,
        provider=resolved_profile.primary.provider,
        model=resolved_profile.primary.model,
    )


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

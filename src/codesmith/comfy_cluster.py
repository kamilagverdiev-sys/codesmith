"""Small ComfyUI prompt router for two or more GPU PCs.

The router does not split one ComfyUI workflow across GPUs. Instead, it exposes
one ComfyUI-compatible endpoint and sends each browser/API client to a healthy
backend ComfyUI instance. Browser sessions are sticky by ``clientId`` so the
ComfyUI WebSocket and prompts land on the same GPU PC.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field, HttpUrl
from websockets.exceptions import ConnectionClosed


class BackendConfig(BaseModel):
    """Configuration for one ComfyUI node."""

    name: str
    url: HttpUrl = Field(description="Base URL, for example http://192.168.1.20:8188")


@dataclass
class BackendState:
    config: BackendConfig
    in_flight: int = 0
    queue_running: int = 0
    queue_pending: int = 0
    ok: bool = True
    last_error: str | None = None
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def base_url(self) -> str:
        return str(self.config.url).rstrip("/")

    @property
    def ws_url(self) -> str:
        base = self.base_url
        if base.startswith("https://"):
            return "wss://" + base.removeprefix("https://")
        return "ws://" + base.removeprefix("http://")

    @property
    def load(self) -> int:
        return self.in_flight + self.queue_running + self.queue_pending


@dataclass
class RouterState:
    backends: list[BackendState]
    prompt_to_backend: dict[str, str] = field(default_factory=dict)
    client_to_backend: dict[str, str] = field(default_factory=dict)
    _round_robin: itertools.count = field(default_factory=itertools.count)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def choose_backend(self) -> BackendState:
        async with self._lock:
            candidates = [backend for backend in self.backends if backend.ok]
            if not candidates:
                candidates = self.backends
            if not candidates:
                raise HTTPException(status_code=503, detail="No ComfyUI backends configured")
            offset = next(self._round_robin)
            return min(
                candidates,
                key=lambda backend: (backend.load, (self.backends.index(backend) - offset) % len(self.backends)),
            )

    async def mark_started(self, backend: BackendState) -> None:
        async with self._lock:
            backend.in_flight += 1

    async def mark_finished(self, backend: BackendState, error: str | None = None) -> None:
        async with self._lock:
            backend.in_flight = max(0, backend.in_flight - 1)
            backend.ok = error is None
            backend.last_error = error
            backend.last_seen = time.monotonic()

    async def update_queue(self, backend: BackendState, running: int, pending: int) -> None:
        async with self._lock:
            backend.queue_running = running
            backend.queue_pending = pending
            backend.ok = True
            backend.last_error = None
            backend.last_seen = time.monotonic()

    async def mark_unhealthy(self, backend: BackendState, error: str) -> None:
        async with self._lock:
            backend.ok = False
            backend.last_error = error
            backend.last_seen = time.monotonic()

    async def remember_prompt(self, prompt_id: str, backend: BackendState) -> None:
        async with self._lock:
            self.prompt_to_backend[prompt_id] = backend.config.name

    async def remember_client(self, client_id: str, backend: BackendState) -> None:
        async with self._lock:
            self.client_to_backend[client_id] = backend.config.name

    def backend_for_prompt(self, prompt_id: str) -> BackendState | None:
        name = self.prompt_to_backend.get(prompt_id)
        return self.backend_by_name(name)

    def backend_for_client(self, client_id: str | None) -> BackendState | None:
        if not client_id:
            return None
        return self.backend_by_name(self.client_to_backend.get(client_id))

    def backend_by_name(self, name: str | None) -> BackendState | None:
        return next((backend for backend in self.backends if backend.config.name == name), None)


def create_app(backends: list[BackendConfig], timeout_seconds: float = 300.0) -> FastAPI:
    """Build a FastAPI app that routes ComfyUI HTTP and WebSocket traffic."""

    state = RouterState([BackendState(config=backend) for backend in backends])
    app = FastAPI(title="ComfyUI GPU Router", version="0.2.0")
    client = httpx.AsyncClient(timeout=timeout_seconds)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    @app.get("/cluster/backends")
    async def list_backends() -> dict[str, Any]:
        await _refresh_all_queues(state, client)
        return {
            "backends": [
                {
                    "name": backend.config.name,
                    "url": backend.base_url,
                    "in_flight": backend.in_flight,
                    "queue_running": backend.queue_running,
                    "queue_pending": backend.queue_pending,
                    "load": backend.load,
                    "ok": backend.ok,
                    "last_error": backend.last_error,
                }
                for backend in state.backends
            ]
        }

    @app.websocket("/ws")
    async def websocket_proxy(websocket: WebSocket) -> None:
        client_id = websocket.query_params.get("clientId")
        await _refresh_all_queues(state, client)
        backend = state.backend_for_client(client_id) or await state.choose_backend()
        if client_id:
            await state.remember_client(client_id, backend)
        await _proxy_websocket(websocket, backend)

    @app.post("/prompt")
    async def prompt(request: Request) -> Response:
        body = await request.body()
        client_id = _client_id_from_prompt_body(body)
        await _refresh_all_queues(state, client)
        backend = state.backend_for_client(client_id) or await state.choose_backend()
        if client_id:
            await state.remember_client(client_id, backend)
        await state.mark_started(backend)
        try:
            upstream = await client.post(
                f"{backend.base_url}/prompt",
                content=body,
                headers=_forward_headers(request),
            )
            data = _response(upstream)
            if upstream.is_success:
                with contextlib.suppress(ValueError):
                    payload = upstream.json()
                    prompt_id = payload.get("prompt_id")
                    if isinstance(prompt_id, str):
                        await state.remember_prompt(prompt_id, backend)
            await state.mark_finished(backend)
            return data
        except httpx.HTTPError as e:
            await state.mark_finished(backend, error=str(e))
            raise HTTPException(status_code=502, detail=f"{backend.config.name}: {e}") from e

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        backend = _select_backend_for_path(state, path, request) or state.backends[0]
        try:
            upstream = await client.request(
                request.method,
                f"{backend.base_url}/{path}",
                params=request.query_params,
                content=await request.body(),
                headers=_forward_headers(request),
            )
            return _response(upstream)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"{backend.config.name}: {e}") from e

    return app


async def _refresh_all_queues(state: RouterState, client: httpx.AsyncClient) -> None:
    await asyncio.gather(*(_refresh_queue(state, client, backend) for backend in state.backends))


async def _refresh_queue(state: RouterState, client: httpx.AsyncClient, backend: BackendState) -> None:
    try:
        response = await client.get(f"{backend.base_url}/queue", timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        running = len(payload.get("queue_running") or [])
        pending = len(payload.get("queue_pending") or [])
        await state.update_queue(backend, running=running, pending=pending)
    except (httpx.HTTPError, ValueError) as e:
        await state.mark_unhealthy(backend, error=str(e))


def _select_backend_for_path(state: RouterState, path: str, request: Request) -> BackendState | None:
    if path.startswith("history/"):
        return state.backend_for_prompt(path.split("/", 1)[1])
    if path in {"queue", "system_stats"}:
        return state.backend_for_client(request.query_params.get("clientId"))
    return None


async def _proxy_websocket(websocket: WebSocket, backend: BackendState) -> None:
    upstream_url = f"{backend.ws_url}/ws"
    if websocket.query_params:
        upstream_url += f"?{websocket.query_params}"
    await websocket.accept()
    try:
        async with websockets.connect(upstream_url) as upstream:
            client_to_backend = asyncio.create_task(_copy_ws_client_to_backend(websocket, upstream))
            backend_to_client = asyncio.create_task(_copy_ws_backend_to_client(upstream, websocket))
            done, pending = await asyncio.wait(
                {client_to_backend, backend_to_client}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
    except (OSError, ConnectionClosed) as e:
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=1011, reason=f"{backend.config.name}: {e}")


async def _copy_ws_client_to_backend(websocket: WebSocket, upstream: Any) -> None:
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                await upstream.send(message["text"])
            elif "bytes" in message:
                await upstream.send(message["bytes"])
    except WebSocketDisconnect:
        await upstream.close()


async def _copy_ws_backend_to_client(upstream: Any, websocket: WebSocket) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(message)


def _client_id_from_prompt_body(body: bytes) -> str | None:
    with contextlib.suppress(ValueError, TypeError):
        payload = json.loads(body)
        client_id = payload.get("client_id")
        if isinstance(client_id, str):
            return client_id
    return None


def _forward_headers(request: Request) -> dict[str, str]:
    ignored = {"host", "content-length", "connection", "upgrade"}
    return {key: value for key, value in request.headers.items() if key.lower() not in ignored}


def _response(upstream: httpx.Response) -> Response:
    headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
    }
    return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)


def parse_backend(value: str) -> BackendConfig:
    """Parse NAME=URL or URL into a backend config."""

    if "=" in value:
        name, url = value.split("=", 1)
    else:
        url = value
        name = f"gpu-{uuid4().hex[:6]}"
    return BackendConfig(name=name.strip(), url=url.strip())

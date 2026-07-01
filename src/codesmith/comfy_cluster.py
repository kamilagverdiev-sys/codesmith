"""Small ComfyUI prompt router for two or more GPU PCs.

The router does not split one ComfyUI workflow across GPUs. Instead, it exposes
one ComfyUI-compatible ``/prompt`` endpoint and sends each new job to the least
busy backend ComfyUI instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, HttpUrl


class BackendConfig(BaseModel):
    """Configuration for one ComfyUI node."""

    name: str
    url: HttpUrl = Field(description="Base URL, for example http://192.168.1.20:8188")


@dataclass
class BackendState:
    config: BackendConfig
    in_flight: int = 0
    ok: bool = True
    last_error: str | None = None
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def base_url(self) -> str:
        return str(self.config.url).rstrip("/")


@dataclass
class RouterState:
    backends: list[BackendState]
    prompt_to_backend: dict[str, str] = field(default_factory=dict)
    _round_robin: itertools.count = field(default_factory=itertools.count)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def choose_backend(self) -> BackendState:
        async with self._lock:
            candidates = [backend for backend in self.backends if backend.ok]
            if not candidates:
                candidates = self.backends
            if not candidates:
                raise HTTPException(status_code=503, detail="No ComfyUI backends configured")
            # Least busy first; counter keeps ties stable without starving peers.
            offset = next(self._round_robin)
            return min(
                candidates,
                key=lambda backend: (backend.in_flight, (self.backends.index(backend) - offset) % len(self.backends)),
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

    async def remember_prompt(self, prompt_id: str, backend: BackendState) -> None:
        async with self._lock:
            self.prompt_to_backend[prompt_id] = backend.config.name

    def backend_for_prompt(self, prompt_id: str) -> BackendState | None:
        name = self.prompt_to_backend.get(prompt_id)
        return next((backend for backend in self.backends if backend.config.name == name), None)


def create_app(backends: list[BackendConfig], timeout_seconds: float = 300.0) -> FastAPI:
    """Build a FastAPI app that routes ComfyUI HTTP prompt traffic."""

    state = RouterState([BackendState(config=backend) for backend in backends])
    app = FastAPI(title="ComfyUI GPU Router", version="0.1.0")
    client = httpx.AsyncClient(timeout=timeout_seconds)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    @app.get("/cluster/backends")
    async def list_backends() -> dict[str, Any]:
        return {
            "backends": [
                {
                    "name": backend.config.name,
                    "url": backend.base_url,
                    "in_flight": backend.in_flight,
                    "ok": backend.ok,
                    "last_error": backend.last_error,
                }
                for backend in state.backends
            ]
        }

    @app.post("/prompt")
    async def prompt(request: Request) -> Response:
        backend = await state.choose_backend()
        await state.mark_started(backend)
        try:
            upstream = await client.post(
                f"{backend.base_url}/prompt",
                content=await request.body(),
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
        backend = _select_backend_for_path(state, path) or await state.choose_backend()
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


def _select_backend_for_path(state: RouterState, path: str) -> BackendState | None:
    # ComfyUI stores generated data under prompt ids for history lookups.
    if path.startswith("history/"):
        return state.backend_for_prompt(path.split("/", 1)[1])
    return None


def _forward_headers(request: Request) -> dict[str, str]:
    ignored = {"host", "content-length", "connection"}
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

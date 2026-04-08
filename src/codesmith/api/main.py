"""FastAPI server — Phase 3 placeholder.

This module will host the HTTP API once Phase 3 begins (see
docs/ROADMAP.md §3). For now it's a stub with the /health endpoint
only, so infrastructure-level smoke tests can start running early.
"""

from __future__ import annotations

from fastapi import FastAPI

from codesmith import __version__

app = FastAPI(
    title="Codesmith API",
    version=__version__,
    description="Hybrid AI coder agent — HTTP interface.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


# Endpoints coming in Phase 3:
#   POST   /sessions
#   GET    /sessions
#   GET    /sessions/{id}
#   DELETE /sessions/{id}
#   POST   /sessions/{id}/messages
#   POST   /sessions/{id}/stream     (Server-Sent Events)

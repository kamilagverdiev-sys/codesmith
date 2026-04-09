"""Per-run observability log.

Appends one JSON line per completed agent run to a local file:

    {"ts": "...", "session_id": "...", "profile": "...",
     "provider": "...", "model": "...", "user_message": "...",
     "steps": 5, "tokens": 1234, "hit_limit": false,
     "forced_final": true, "duration_ms": 12456, "error": null}

This is the base for debugging ("why did that run take 20 iterations?")
and for future self-repair telemetry. Intentionally schema-light and
append-only so we can rotate / grep it like any other log file.

Not involved in the hot path — callers must opt in by constructing a
RunLogger and calling .record(...). Failures in logging NEVER propagate
back to the agent; an observability layer must not break production.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = Path("~/.codesmith/logs/runs.jsonl").expanduser()
_USER_MESSAGE_MAX = 500


@dataclass
class RunRecord:
    """One completed run's metadata, ready for JSONL serialization."""

    ts: str
    session_id: str
    profile: str
    provider: str
    model: str
    steps: int
    tokens: int
    hit_limit: bool
    duration_ms: int
    forced_final: bool = False
    user_message: str = ""
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _truncate(text: str, limit: int = _USER_MESSAGE_MAX) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class RunLogger:
    """Thin append-only JSONL writer.

    Silent no-op on any IO error (including missing parent dir if we
    cannot create it): an observability log that crashes the agent is
    worse than no log at all.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or _DEFAULT_LOG_PATH).expanduser()

    def _ensure_parent(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # noqa: BLE001
            log.warning("run log parent dir not writable: %s", e)
            return False
        return True

    def record(self, run: RunRecord) -> None:
        """Append one run record as a single JSON line. Never raises."""
        try:
            if not self._ensure_parent():
                return
            line = json.dumps(asdict(run), ensure_ascii=False, default=str)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
        except Exception as e:  # noqa: BLE001
            log.warning("run log write failed: %s", e)

    def tail(self, limit: int = 20) -> list[RunRecord]:
        """Return the last `limit` records (oldest→newest). Empty list on error."""
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        records: list[RunRecord] = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                records.append(
                    RunRecord(
                        ts=str(payload.get("ts", "")),
                        session_id=str(payload.get("session_id", "")),
                        profile=str(payload.get("profile", "")),
                        provider=str(payload.get("provider", "")),
                        model=str(payload.get("model", "")),
                        steps=int(payload.get("steps", 0) or 0),
                        tokens=int(payload.get("tokens", 0) or 0),
                        hit_limit=bool(payload.get("hit_limit", False)),
                        duration_ms=int(payload.get("duration_ms", 0) or 0),
                        forced_final=bool(payload.get("forced_final", False)),
                        user_message=str(payload.get("user_message", "")),
                        error=payload.get("error"),
                        extra=dict(payload.get("extra") or {}),
                    )
                )
            except (TypeError, ValueError):
                continue
        return records


def build_run_record(
    *,
    session_id: str,
    profile_key: str,
    provider: str,
    model: str,
    user_message: str,
    steps: int,
    tokens: int,
    hit_limit: bool,
    duration_ms: int,
    forced_final: bool,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RunRecord:
    """Helper: construct a RunRecord with a UTC timestamp applied."""
    return RunRecord(
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        session_id=session_id,
        profile=profile_key,
        provider=provider,
        model=model,
        steps=steps,
        tokens=tokens,
        hit_limit=hit_limit,
        duration_ms=duration_ms,
        forced_final=forced_final,
        user_message=_truncate(user_message),
        error=error,
        extra=extra or {},
    )

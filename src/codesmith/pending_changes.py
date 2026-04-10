"""Pending-change queue for human-in-the-loop file edits.

Local coder models are unpredictable on real repos. Forcing a human
approval step between "the agent wants to change X" and "X is on
disk" converts the agent from a risky auto-writer into a structured
proposal generator that the user can accept or reject per change.

The flow:

1. Every mutating filesystem tool (write_file, edit_file, apply_patch)
   checks `session.metadata.get("auto_approve", True)`.
2. When auto_approve is False, the tool computes the proposed new
   content, DOES NOT touch the filesystem, and instead enqueues a
   PendingChange onto `session.metadata["pending_changes"]`, then
   returns a ToolResult whose body tells the model "PENDING: change
   #N queued — waiting for human review".
3. The API exposes:
     GET    /api/sessions/{id}/pending-changes
     POST   /api/sessions/{id}/pending-changes/{idx}/approve
     POST   /api/sessions/{id}/pending-changes/{idx}/reject
     POST   /api/sessions/{id}/auto-approve
4. apply_change() does the actual filesystem write when approved.

PendingChange is stored as a plain dict on session.metadata so the
whole thing round-trips through SQLite session persistence for free.
"""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codesmith.session import Session

# Keys in session.metadata
_PENDING_KEY = "pending_changes"
_AUTO_APPROVE_KEY = "auto_approve"
_NEXT_IDX_KEY = "pending_changes_next_idx"

STATE_PENDING = "pending"
STATE_APPROVED = "approved"  # not used yet; reserved for two-phase flows
STATE_APPLIED = "applied"
STATE_REJECTED = "rejected"
STATE_FAILED = "failed"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def is_auto_approve(session: Session) -> bool:
    """Default is True — existing behaviour for legacy sessions.

    The Web UI and the /auto-approve endpoint flip this to False
    when the user wants human-in-the-loop review for every write.
    """
    value = session.metadata.get(_AUTO_APPROVE_KEY, True)
    return bool(value)


def set_auto_approve(session: Session, enabled: bool) -> None:
    session.metadata[_AUTO_APPROVE_KEY] = bool(enabled)


def list_pending(session: Session) -> list[dict[str, Any]]:
    """Return pending changes (all states) as plain dicts."""
    raw = session.metadata.get(_PENDING_KEY)
    return list(raw) if isinstance(raw, list) else []


def _next_idx(session: Session) -> int:
    current = int(session.metadata.get(_NEXT_IDX_KEY, 0) or 0)
    session.metadata[_NEXT_IDX_KEY] = current + 1
    return current


def _build_diff(before: str, after: str, path: str) -> str:
    before_lines = before.splitlines(keepends=True) or [""]
    after_lines = after.splitlines(keepends=True) or [""]
    diff_lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff_lines)


def queue_change(
    session: Session,
    *,
    kind: str,
    path: str,
    before: str,
    after: str,
    summary: str | None = None,
    files: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enqueue a pending change. Returns the stored dict (with idx)."""
    idx = _next_idx(session)
    change: dict[str, Any] = {
        "idx": idx,
        "kind": kind,
        "path": path,
        "before": before,
        "after": after,
        "diff": _build_diff(before, after, path),
        "state": STATE_PENDING,
        "created_at": _now_iso(),
        "applied_at": None,
        "summary": summary or "",
    }
    # For multi-file patches we additionally store the per-file list
    # so the UI can render a nested diff viewer without re-parsing.
    if files is not None:
        change["files"] = files
    pending = list_pending(session)
    pending.append(change)
    session.metadata[_PENDING_KEY] = pending
    return change


def get_change(session: Session, idx: int) -> dict[str, Any] | None:
    for item in list_pending(session):
        if int(item.get("idx", -1)) == idx:
            return item
    return None


def _update_change(session: Session, idx: int, **fields: Any) -> dict[str, Any] | None:
    pending = list_pending(session)
    for item in pending:
        if int(item.get("idx", -1)) == idx:
            item.update(fields)
            session.metadata[_PENDING_KEY] = pending
            return item
    return None


def reject_change(session: Session, idx: int) -> dict[str, Any] | None:
    return _update_change(
        session,
        idx,
        state=STATE_REJECTED,
        applied_at=_now_iso(),
    )


def apply_change(
    session: Session,
    idx: int,
    *,
    workspace_dir: Path,
) -> dict[str, Any]:
    """Write the approved content to disk.

    Raises:
        KeyError: if the change doesn't exist.
        ValueError: if the change is in a terminal state already.
        OSError: if the write fails (leaves the change marked FAILED).

    Returns the updated change dict.
    """
    change = get_change(session, idx)
    if change is None:
        raise KeyError(f"pending change {idx} not found")
    state = change.get("state")
    if state in (STATE_APPLIED, STATE_REJECTED, STATE_FAILED):
        raise ValueError(f"pending change {idx} already in terminal state: {state}")

    kind = change.get("kind")
    if kind == "apply_patch":
        files = change.get("files") or []
        try:
            _write_many_atomic(workspace_dir, files)
        except OSError as e:
            _update_change(
                session,
                idx,
                state=STATE_FAILED,
                applied_at=_now_iso(),
                error=str(e),
            )
            raise
    else:
        target_rel = change.get("path") or ""
        after = change.get("after", "")
        target = (workspace_dir / target_rel).resolve()
        # Resolve-safety: target must stay inside the workspace.
        try:
            target.relative_to(workspace_dir.resolve())
        except ValueError as e:
            _update_change(
                session,
                idx,
                state=STATE_FAILED,
                applied_at=_now_iso(),
                error=f"path escapes workspace: {target}",
            )
            raise OSError(f"path escapes workspace: {target}") from e
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(after, encoding="utf-8")
        except OSError as e:
            _update_change(
                session,
                idx,
                state=STATE_FAILED,
                applied_at=_now_iso(),
                error=str(e),
            )
            raise

    updated = _update_change(
        session,
        idx,
        state=STATE_APPLIED,
        applied_at=_now_iso(),
    )
    assert updated is not None
    return updated


def _write_many_atomic(
    workspace_dir: Path,
    files: list[dict[str, Any]],
) -> None:
    """Write several files in one atomic-ish batch.

    Two-pass strategy:
      1. Resolve every target path and verify it stays inside the
         workspace; raise early on any escape.
      2. Write all files. If step 2 fails partway through, we don't
         roll back (file systems are not transactional) — but the
         validation pass catches every preventable error up front.
    """
    workspace_abs = workspace_dir.resolve()
    resolved: list[tuple[Path, str]] = []
    for item in files:
        rel = str(item.get("path") or "")
        after = item.get("after", "")
        if not isinstance(after, str):
            raise OSError(f"bad after content for {rel}: not a string")
        target = (workspace_abs / rel).resolve()
        try:
            target.relative_to(workspace_abs)
        except ValueError as e:
            raise OSError(f"path escapes workspace: {rel}") from e
        resolved.append((target, after))

    for target, after in resolved:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(after, encoding="utf-8")


def format_pending_preview(change: dict[str, Any]) -> str:
    """Short human-readable tool-result string that an agent sees.

    The agent MUST see enough to understand the edit was queued, not
    applied — otherwise it'll claim the task is done and finish.
    """
    idx = change.get("idx")
    kind = change.get("kind", "change")
    path = change.get("path", "?")
    return (
        f"PENDING: {kind} on {path} queued as change #{idx}. "
        "Waiting for human review — the file has NOT been modified yet. "
        "Do not assume the change is applied; if a later step needs the "
        "new content, stop and summarize the pending changes for the user."
    )

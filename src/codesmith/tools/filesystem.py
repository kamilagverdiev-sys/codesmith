"""Filesystem tools scoped to the session workspace.

Every path argument is resolved against session.workspace_dir and then
checked to still be inside it. Absolute paths, symlinks, and `..`
escapes are rejected. This means the LLM can't exfiltrate files from
the host via the filesystem tool — only the sandbox sees the host
filesystem, and even the sandbox only sees /work.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from codesmith.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from codesmith.session import Session


MAX_READ_BYTES = 256 * 1024  # 256 KiB — enough for source files, not log dumps
MAX_WRITE_BYTES = 1 * 1024 * 1024  # 1 MiB


class PathEscapeError(Exception):
    """Path tried to escape the workspace."""


def _resolve_safe(workspace: Path, user_path: str) -> Path:
    """Resolve user-supplied path against workspace, reject escapes.

    Strategy: build the candidate, resolve it (follows symlinks and
    normalizes `..`), then check it's still under workspace.resolve().
    """
    workspace_abs = workspace.resolve()
    raw = Path(user_path)
    # Absolute paths are allowed ONLY if they point inside the workspace.
    candidate = raw if raw.is_absolute() else workspace_abs / raw
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError) as e:
        raise PathEscapeError(f"cannot resolve path: {e}") from e
    try:
        resolved.relative_to(workspace_abs)
    except ValueError:
        raise PathEscapeError(
            f"path escapes workspace: {user_path} → {resolved}"
        ) from None
    return resolved


# ============================================================
# read_file
# ============================================================


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a text file from the workspace. "
        "Paths are relative to the workspace root. Max 256 KiB."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path inside workspace"},
        },
        "required": ["path"],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        path_arg = kwargs.get("path", "")
        try:
            p = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        if not p.exists():
            return ToolResult(ok=False, content="", error=f"not found: {path_arg}")
        if not p.is_file():
            return ToolResult(ok=False, content="", error=f"not a file: {path_arg}")

        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            return ToolResult(
                ok=False,
                content="",
                error=f"file too large: {size} bytes > {MAX_READ_BYTES}",
            )
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(ok=False, content="", error=f"read failed: {e}")

        return ToolResult(
            ok=True,
            content=text,
            metadata={"size": size, "path": str(p.relative_to(session.workspace_dir))},
        )


# ============================================================
# write_file
# ============================================================


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write (or overwrite) a text file in the workspace. "
        "Creates parent directories if needed. Max 1 MiB."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path inside workspace"},
            "content": {"type": "string", "description": "Text content to write"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        path_arg = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolResult(
                ok=False, content="", error=f"content too large: > {MAX_WRITE_BYTES} bytes"
            )

        try:
            p = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult(ok=False, content="", error=f"write failed: {e}")

        return ToolResult(
            ok=True,
            content=f"wrote {len(content)} chars to {p.relative_to(session.workspace_dir)}",
            metadata={"path": str(p.relative_to(session.workspace_dir))},
        )


# ============================================================
# list_directory
# ============================================================


class ListDirTool(BaseTool):
    name = "list_directory"
    description = "List files and subdirectories in a workspace directory."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path inside workspace. Use '.' for workspace root.",
                "default": ".",
            },
        },
        "required": [],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        path_arg = kwargs.get("path", ".")
        try:
            p = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        if not p.exists():
            return ToolResult(ok=False, content="", error=f"not found: {path_arg}")
        if not p.is_dir():
            return ToolResult(ok=False, content="", error=f"not a directory: {path_arg}")

        entries: list[str] = []
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            try:
                size = child.stat().st_size if child.is_file() else 0
            except OSError:
                size = 0
            entries.append(f"{child.name}{suffix} ({size}B)" if suffix == "" else f"{child.name}{suffix}")

        content = "\n".join(entries) if entries else "(empty)"
        return ToolResult(
            ok=True,
            content=content,
            metadata={"count": len(entries)},
        )


def default_filesystem_tools() -> list[BaseTool]:
    """Convenience bundle — returns one instance of each FS tool."""
    return [ReadFileTool(), WriteFileTool(), ListDirTool()]

"""Filesystem tools scoped to the session workspace.

Every path argument is resolved against session.workspace_dir and then
checked to still be inside it. Absolute paths, symlinks, and `..`
escapes are rejected. This means the LLM can't exfiltrate files from
the host via the filesystem tool — only the sandbox sees the host
filesystem, and even the sandbox only sees /work.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codesmith.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from codesmith.session import Session


MAX_READ_BYTES = 256 * 1024  # 256 KiB — enough for source files, not log dumps
MAX_WRITE_BYTES = 1 * 1024 * 1024  # 1 MiB

# Files we NEVER walk through when doing grep/glob. These are either huge
# binary blobs, caches, or virtual env dumps that nuke the model's context
# window with no useful signal.
_SEARCH_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".idea",
        ".vscode",
    }
)

_SEARCH_BINARY_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".exe",
        ".bin",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".7z",
        ".rar",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".pdf",
        ".mp3",
        ".mp4",
        ".woff",
        ".woff2",
        ".ttf",
    }
)

_SEARCH_MAX_FILE_BYTES = 256 * 1024
_SEARCH_MAX_FILES_WALKED = 2000
_SEARCH_MAX_MATCHES = 80
_GLOB_MAX_RESULTS = 200


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


# ============================================================
# edit_file
# ============================================================


class EditFileTool(BaseTool):
    """Literal string replacement in an existing workspace file.

    Safer and far cheaper than repeated full write_file for models that tend
    to re-serialize the whole file on every tiny change. Follows Claude's
    Edit semantics:

    * find must occur exactly once in the file (otherwise the model has to
      disambiguate by passing more surrounding context);
    * find and replace must differ;
    * the file must already exist (use write_file to create new files).
    """

    name = "edit_file"
    description = (
        "Edit an existing file by replacing one exact occurrence of `find` with "
        "`replace`. Use this instead of write_file for small, targeted changes. "
        "`find` must appear exactly once — include surrounding context if needed."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path inside workspace"},
            "find": {
                "type": "string",
                "description": (
                    "Exact text to replace. Must occur exactly once in the file."
                ),
            },
            "replace": {
                "type": "string",
                "description": "Replacement text. Must differ from `find`.",
            },
        },
        "required": ["path", "find", "replace"],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        path_arg = kwargs.get("path", "")
        find = kwargs.get("find", "")
        replace = kwargs.get("replace", "")

        if not isinstance(find, str) or not find:
            return ToolResult(ok=False, content="", error="`find` must be a non-empty string")
        if not isinstance(replace, str):
            return ToolResult(ok=False, content="", error="`replace` must be a string")
        if find == replace:
            return ToolResult(
                ok=False, content="", error="`find` and `replace` must differ"
            )

        try:
            p = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        if not p.exists():
            return ToolResult(
                ok=False,
                content="",
                error=f"not found: {path_arg} (use write_file to create new files)",
            )
        if not p.is_file():
            return ToolResult(ok=False, content="", error=f"not a file: {path_arg}")

        try:
            original = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=f"read failed: {e}")

        occurrences = original.count(find)
        if occurrences == 0:
            return ToolResult(
                ok=False,
                content="",
                error="find string not present in file — pass a longer, unique snippet",
            )
        if occurrences > 1:
            return ToolResult(
                ok=False,
                content="",
                error=(
                    f"find string matches {occurrences} places — include more "
                    "surrounding context so it matches exactly once"
                ),
            )

        updated = original.replace(find, replace, 1)

        if len(updated.encode("utf-8")) > MAX_WRITE_BYTES:
            return ToolResult(
                ok=False, content="", error=f"result too large: > {MAX_WRITE_BYTES} bytes"
            )

        try:
            p.write_text(updated, encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            return ToolResult(ok=False, content="", error=f"write failed: {e}")

        rel = p.relative_to(session.workspace_dir)
        delta = len(updated) - len(original)
        sign = "+" if delta >= 0 else ""
        return ToolResult(
            ok=True,
            content=f"edited {rel} ({sign}{delta} chars)",
            metadata={
                "path": str(rel),
                "delta_chars": delta,
                "occurrences_matched": 1,
            },
        )


# ============================================================
# grep_workspace
# ============================================================


def _is_searchable_file(path: Path) -> bool:
    if path.suffix.lower() in _SEARCH_BINARY_SUFFIXES:
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return size <= _SEARCH_MAX_FILE_BYTES


def _iter_searchable_files(
    root: Path,
    *,
    include_glob: str | None = None,
) -> list[Path]:
    """Walk `root`, honoring skip-dirs / binary skips / size cap / file cap."""
    matches: list[Path] = []
    walked = 0
    for current, dirs, files in _scandir_walk(root):
        # Prune skip dirs in-place so os.walk / our walker doesn't descend.
        dirs[:] = [d for d in dirs if d not in _SEARCH_SKIP_DIRS]
        for name in files:
            walked += 1
            if walked > _SEARCH_MAX_FILES_WALKED:
                return matches
            candidate = current / name
            if not _is_searchable_file(candidate):
                continue
            if include_glob is not None:
                rel = candidate.relative_to(root).as_posix()
                if not (
                    fnmatch.fnmatch(candidate.name, include_glob)
                    or fnmatch.fnmatch(rel, include_glob)
                ):
                    continue
            matches.append(candidate)
    return matches


def _scandir_walk(root: Path):
    """Tiny os.walk lookalike that yields Path objects and is easy to test."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        dirs: list[str] = []
        files: list[str] = []
        for entry in entries:
            try:
                if entry.is_dir() and not entry.is_symlink():
                    dirs.append(entry.name)
                elif entry.is_file():
                    files.append(entry.name)
            except OSError:
                continue
        yield current, dirs, files
        # Push children in deterministic order for reproducible results.
        for d in sorted(dirs, reverse=True):
            if d in _SEARCH_SKIP_DIRS:
                continue
            stack.append(current / d)


class GrepWorkspaceTool(BaseTool):
    """Regex search across workspace files with strict caps.

    Replaces the painful 'read every file and hope' pattern that small
    models love. Skips .git / .venv / node_modules and other usual
    suspects, skips binaries, caps per-file read at 256 KiB, total files
    walked at 2000, and total matches at 80.
    """

    name = "grep_workspace"
    description = (
        "Search file contents with a regular expression, workspace-scoped. "
        "Returns up to 80 matches as `path:line: text`. Use the `include` "
        "glob to narrow the file set (e.g. '*.py', 'src/**/*.ts'). Prefer "
        "this over reading files one-by-one when exploring unknown code."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Python regex. Case-sensitive by default.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Subdirectory inside workspace to search. '.' for the "
                    "workspace root (default)."
                ),
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": (
                    "Optional filename glob filter, e.g. '*.py', '*.{js,ts}' "
                    "or 'src/**/*.py'. Omit to match all text files."
                ),
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Set true to match case-insensitively.",
                "default": False,
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        path_arg = kwargs.get("path", ".") or "."
        include = kwargs.get("include")
        case_insensitive = bool(kwargs.get("case_insensitive"))

        if not isinstance(pattern, str) or not pattern:
            return ToolResult(
                ok=False, content="", error="`pattern` must be a non-empty string"
            )
        try:
            regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
        except re.error as e:
            return ToolResult(ok=False, content="", error=f"bad regex: {e}")

        try:
            root = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        if not root.exists():
            return ToolResult(ok=False, content="", error=f"not found: {path_arg}")
        if not root.is_dir():
            return ToolResult(ok=False, content="", error=f"not a directory: {path_arg}")

        matches: list[str] = []
        files_hit = 0
        truncated = False
        workspace_abs = session.workspace_dir.resolve()

        for file_path in _iter_searchable_files(root, include_glob=include):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            hit_in_file = False
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hit_in_file = True
                    rel = file_path.relative_to(workspace_abs).as_posix()
                    snippet = line.strip()
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "…"
                    matches.append(f"{rel}:{lineno}: {snippet}")
                    if len(matches) >= _SEARCH_MAX_MATCHES:
                        truncated = True
                        break
            if hit_in_file:
                files_hit += 1
            if truncated:
                break

        if not matches:
            return ToolResult(
                ok=True,
                content="(no matches)",
                metadata={"files_with_matches": 0, "matches": 0},
            )

        header = f"{len(matches)} match(es) across {files_hit} file(s)"
        if truncated:
            header += f" (truncated at {_SEARCH_MAX_MATCHES})"
        body = "\n".join(matches)
        return ToolResult(
            ok=True,
            content=f"{header}\n{body}",
            metadata={
                "files_with_matches": files_hit,
                "matches": len(matches),
                "truncated": truncated,
            },
        )


# ============================================================
# glob_workspace
# ============================================================


class GlobWorkspaceTool(BaseTool):
    """Find files by filename pattern, recursively, workspace-scoped.

    Uses `Path.rglob` when the pattern contains `**`, otherwise a plain
    `Path.glob`. Honors the same skip-dir list as grep_workspace so the
    LLM doesn't get drowned in node_modules.
    """

    name = "glob_workspace"
    description = (
        "List files matching a glob pattern, recursively, workspace-scoped. "
        "Examples: '**/*.py', 'src/**/*.ts', '*.md'. Returns up to 200 paths. "
        "Use this before grep_workspace when you know the file-type but not "
        "the content."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern. Supports '*' and '**'. Paths are relative "
                    "to the workspace root."
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Subdirectory inside workspace to search from. '.' for "
                    "the workspace root (default)."
                ),
                "default": ".",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        path_arg = kwargs.get("path", ".") or "."

        if not isinstance(pattern, str) or not pattern:
            return ToolResult(
                ok=False, content="", error="`pattern` must be a non-empty string"
            )

        try:
            root = _resolve_safe(session.workspace_dir, path_arg)
        except PathEscapeError as e:
            return ToolResult(ok=False, content="", error=str(e))

        if not root.exists():
            return ToolResult(ok=False, content="", error=f"not found: {path_arg}")
        if not root.is_dir():
            return ToolResult(ok=False, content="", error=f"not a directory: {path_arg}")

        workspace_abs = session.workspace_dir.resolve()
        try:
            raw = (
                root.rglob(pattern.replace("**/", ""))
                if "**" in pattern
                else root.glob(pattern)
            )
            iterable = iter(raw)
        except (ValueError, OSError) as e:
            return ToolResult(ok=False, content="", error=f"glob failed: {e}")

        results: list[str] = []
        truncated = False
        for candidate in iterable:
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            # Skip anything inside our skip-dirs.
            try:
                rel_parts = candidate.relative_to(workspace_abs).parts
            except ValueError:
                continue
            if any(part in _SEARCH_SKIP_DIRS for part in rel_parts):
                continue
            results.append("/".join(rel_parts))
            if len(results) >= _GLOB_MAX_RESULTS:
                truncated = True
                break

        results.sort()
        if not results:
            return ToolResult(
                ok=True,
                content="(no matches)",
                metadata={"matches": 0},
            )

        header = f"{len(results)} file(s)"
        if truncated:
            header += f" (truncated at {_GLOB_MAX_RESULTS})"
        return ToolResult(
            ok=True,
            content=f"{header}\n" + "\n".join(results),
            metadata={"matches": len(results), "truncated": truncated},
        )


def default_filesystem_tools() -> list[BaseTool]:
    """Convenience bundle — returns one instance of each FS tool."""
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        GrepWorkspaceTool(),
        GlobWorkspaceTool(),
    ]

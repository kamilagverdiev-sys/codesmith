"""Auto-build a short workspace snapshot for agent bootstrap.

At the very first turn of a session the agent has no idea what's in
its workspace — local coder models usually waste 2-3 tool calls just
running list_directory / read_file to get oriented. Feeding them a
short, bounded tree of the workspace up front as part of the system
prompt lets them skip that warm-up entirely.

The snapshot is DELIBERATELY small:
  - at most ~30 files,
  - only the top two directory levels, plus a handful of "shallow-
    interesting" entries (README, pyproject.toml, package.json, etc.),
  - skip the same junk dirs (.git / .venv / node_modules) that
    grep_workspace already ignores.

Empty workspace → empty string. The API caller is expected to fall
back to the default system prompt in that case.
"""

from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = frozenset(
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
        ".claude",
        ".next",
        ".cache",
        "_inspect",
        "coverage",
        ".coverage",
        "target",
        "~",
    }
)


def _is_skippable_dir(name: str) -> bool:
    """Skip well-known junk dirs PLUS any hidden / underscore dir that
    didn't make the interesting whitelist."""
    if name in _SKIP_DIRS:
        return True
    # Hidden or private-looking directories default to "skip".
    return name.startswith(".") or name.startswith("_")

_INTERESTING_ROOT_FILES = frozenset(
    {
        "README.md",
        "README.rst",
        "README",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "poetry.lock",
        "package.json",
        "tsconfig.json",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "Makefile",
        "CMakeLists.txt",
        "Cargo.toml",
        "go.mod",
    }
)

_MAX_FILES_LISTED = 30


def _safe_iter(path: Path) -> list[Path]:
    try:
        return sorted(path.iterdir())
    except (OSError, PermissionError):
        return []


def build_workspace_snapshot(workspace_dir: Path, *, max_files: int = _MAX_FILES_LISTED) -> str:
    """Return a short plain-text description of the workspace, or ''.

    Format (example):

        /work
        |- README.md
        |- pyproject.toml
        |- src/
        |   |- app.py
        |   |- utils.py
        |- tests/
        |   |- test_app.py
        |- (3 more files)
    """
    root = Path(workspace_dir)
    if not root.exists() or not root.is_dir():
        return ""

    entries = [
        e for e in _safe_iter(root)
        if not (e.is_dir() and _is_skippable_dir(e.name))
    ]
    if not entries:
        return ""

    lines: list[str] = []
    files_listed = 0

    def _add(prefix: str, name: str) -> bool:
        nonlocal files_listed
        if files_listed >= max_files:
            return False
        lines.append(f"{prefix}{name}")
        files_listed += 1
        return True

    # Root level
    interesting = [e for e in entries if e.is_file() and e.name in _INTERESTING_ROOT_FILES]
    other_files = [
        e for e in entries
        if e.is_file() and e.name not in _INTERESTING_ROOT_FILES and not e.name.startswith(".")
    ]
    dirs = [e for e in entries if e.is_dir() and not _is_skippable_dir(e.name)]

    for e in interesting:
        if not _add("|- ", e.name):
            break
    for e in other_files[:5]:  # cap unknown root files
        if not _add("|- ", e.name):
            break

    # Top-level dirs with their immediate children (cap per dir).
    for d in dirs:
        if files_listed >= max_files:
            break
        _add("|- ", d.name + "/")
        children = [
            c for c in _safe_iter(d)
            if not (c.is_dir() and _is_skippable_dir(c.name))
            and not (c.is_file() and c.name.startswith("."))
        ]
        children_files = [c for c in children if c.is_file()]
        children_dirs = [c for c in children if c.is_dir()]
        for c in children_files[:4]:
            if not _add("|   |- ", c.name):
                break
        for c in children_dirs[:3]:
            if not _add("|   |- ", c.name + "/"):
                break
        if len(children_files) + len(children_dirs) > 7:
            extra = len(children_files) + len(children_dirs) - 7
            lines.append(f"|   |- ({extra} more entries)")

    if files_listed >= max_files:
        lines.append(f"|- (truncated at {max_files} entries)")

    if not lines:
        return ""

    header = str(root)
    return header + "\n" + "\n".join(lines)


def bootstrap_system_prompt_addition(workspace_dir: Path) -> str:
    """Wrap build_workspace_snapshot into a labelled block, or '' if empty.

    The API chat endpoint prepends this to DEFAULT_SYSTEM_PROMPT on
    the first user turn so the agent starts with a grounded view of
    the workspace instead of having to tool-call its way in.
    """
    snapshot = build_workspace_snapshot(workspace_dir)
    if not snapshot:
        return ""
    return (
        "## Workspace snapshot\n"
        "This is the current state of your workspace (truncated). Use it as\n"
        "a starting point; call list_directory / glob_workspace / read_file\n"
        "for deeper reads when needed.\n\n"
        "```\n"
        f"{snapshot}\n"
        "```\n"
    )

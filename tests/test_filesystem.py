"""Tests for codesmith.tools.filesystem.

The path-escape tests are security-critical. If any of them starts
failing, STOP and investigate — it means the workspace isolation is
broken and the agent can now read/write arbitrary host files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from codesmith.tools.filesystem import (
    EditFileTool,
    GlobWorkspaceTool,
    GrepWorkspaceTool,
    ListDirTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
    _resolve_safe,
    default_filesystem_tools,
)


@dataclass
class _FakeSession:
    """Minimal session stand-in — filesystem tools only need workspace_dir."""

    workspace_dir: Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.txt").write_text("hi")
    (ws / "sub").mkdir()
    (ws / "sub" / "nested.txt").write_text("nested")
    return ws


@pytest.fixture
def session(workspace: Path) -> _FakeSession:
    return _FakeSession(workspace_dir=workspace)


# ============================================================
# _resolve_safe — the single most important function here
# ============================================================


class TestResolveSafe:
    def test_relative_path_inside_workspace(self, workspace: Path) -> None:
        assert _resolve_safe(workspace, "hello.txt").name == "hello.txt"

    def test_subdirectory_path(self, workspace: Path) -> None:
        assert _resolve_safe(workspace, "sub/nested.txt").name == "nested.txt"

    def test_dot_resolves_to_workspace(self, workspace: Path) -> None:
        assert _resolve_safe(workspace, ".").resolve() == workspace.resolve()

    @pytest.mark.parametrize(
        "bad_path",
        [
            "../etc/passwd",
            "../../../../etc/passwd",
            "/etc/passwd",
            "/tmp/evil",
            "sub/../../outside",
            "./sub/../../../outside",
        ],
    )
    def test_escapes_rejected(self, workspace: Path, bad_path: str) -> None:
        """Every common path-escape pattern must raise."""
        with pytest.raises(PathEscapeError):
            _resolve_safe(workspace, bad_path)


# ============================================================
# ReadFileTool
# ============================================================


class TestReadFile:
    async def test_read_existing(self, session: _FakeSession) -> None:
        tool = ReadFileTool()
        result = await tool.execute(session, path="hello.txt")
        assert result.ok
        assert result.content == "hi"

    async def test_read_missing(self, session: _FakeSession) -> None:
        tool = ReadFileTool()
        result = await tool.execute(session, path="nope.txt")
        assert not result.ok
        assert "not found" in (result.error or "")

    async def test_read_escape_attempt(self, session: _FakeSession) -> None:
        tool = ReadFileTool()
        result = await tool.execute(session, path="../../../etc/passwd")
        assert not result.ok
        assert "escape" in (result.error or "")


# ============================================================
# WriteFileTool
# ============================================================


class TestWriteFile:
    async def test_write_creates_file(
        self, session: _FakeSession, workspace: Path
    ) -> None:
        tool = WriteFileTool()
        result = await tool.execute(session, path="new.py", content="print(42)")
        assert result.ok
        assert (workspace / "new.py").read_text() == "print(42)"

    async def test_write_creates_parent_dirs(
        self, session: _FakeSession, workspace: Path
    ) -> None:
        tool = WriteFileTool()
        result = await tool.execute(
            session, path="deep/nested/dir/file.txt", content="ok"
        )
        assert result.ok
        assert (workspace / "deep" / "nested" / "dir" / "file.txt").exists()

    async def test_write_escape_rejected(self, session: _FakeSession) -> None:
        tool = WriteFileTool()
        result = await tool.execute(
            session, path="../evil.py", content="pwn"
        )
        assert not result.ok
        assert "escape" in (result.error or "")


# ============================================================
# ListDirTool
# ============================================================


class TestListDir:
    async def test_list_root(self, session: _FakeSession) -> None:
        tool = ListDirTool()
        result = await tool.execute(session, path=".")
        assert result.ok
        assert "hello.txt" in result.content
        assert "sub/" in result.content

    async def test_list_subdirectory(self, session: _FakeSession) -> None:
        tool = ListDirTool()
        result = await tool.execute(session, path="sub")
        assert result.ok
        assert "nested.txt" in result.content

    async def test_list_missing(self, session: _FakeSession) -> None:
        tool = ListDirTool()
        result = await tool.execute(session, path="nowhere")
        assert not result.ok


# ============================================================
# EditFileTool
# ============================================================


class TestEditFile:
    async def test_edit_replaces_unique_snippet(
        self, session: _FakeSession, workspace: Path
    ) -> None:
        target = workspace / "main.py"
        target.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="main.py",
            find="return 'hi'",
            replace="return 'hello'",
        )
        assert result.ok
        assert target.read_text(encoding="utf-8") == "def greet():\n    return 'hello'\n"
        assert result.metadata["occurrences_matched"] == 1

    async def test_edit_rejects_multiple_matches(
        self, session: _FakeSession, workspace: Path
    ) -> None:
        (workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="dup.py",
            find="x = 1",
            replace="x = 2",
        )
        assert not result.ok
        assert "2 places" in (result.error or "")

    async def test_edit_rejects_missing_snippet(
        self, session: _FakeSession
    ) -> None:
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="hello.txt",
            find="nope",
            replace="yup",
        )
        assert not result.ok
        assert "not present" in (result.error or "")

    async def test_edit_rejects_noop(
        self, session: _FakeSession
    ) -> None:
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="hello.txt",
            find="hi",
            replace="hi",
        )
        assert not result.ok
        assert "differ" in (result.error or "")

    async def test_edit_rejects_missing_file(
        self, session: _FakeSession
    ) -> None:
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="nope.py",
            find="x",
            replace="y",
        )
        assert not result.ok
        assert "not found" in (result.error or "")

    async def test_edit_rejects_path_escape(
        self, session: _FakeSession
    ) -> None:
        tool = EditFileTool()
        result = await tool.execute(
            session,
            path="../../../etc/passwd",
            find="root",
            replace="pwn",
        )
        assert not result.ok
        assert "escape" in (result.error or "")


def test_default_filesystem_tools_includes_edit_file() -> None:
    names = {t.name for t in default_filesystem_tools()}
    assert {
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "grep_workspace",
        "glob_workspace",
    } <= names


# ============================================================
# GrepWorkspaceTool
# ============================================================


@pytest.fixture
def search_workspace(tmp_path: Path) -> Path:
    """Small tree with Python files, a markdown file, a big binary, and
    a node_modules dir that must be skipped."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text(
        "def greet(name):\n    return 'hello ' + name\n\n# TODO: unit test\n",
        encoding="utf-8",
    )
    (ws / "src" / "utils.py").write_text(
        "def helper():\n    pass  # TODO: rewrite helper\n",
        encoding="utf-8",
    )
    (ws / "README.md").write_text(
        "# Project\n\nSome notes. TODO: write docs.\n",
        encoding="utf-8",
    )
    # A path we expect grep/glob to SKIP:
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "bad.js").write_text("TODO: should be skipped\n", encoding="utf-8")
    # A binary-by-suffix file we expect to SKIP:
    (ws / "src" / "image.png").write_bytes(b"\x89PNG\r\nTODO stuff\n")
    return ws


@pytest.fixture
def search_session(search_workspace: Path) -> _FakeSession:
    return _FakeSession(workspace_dir=search_workspace)


class TestGrepWorkspace:
    async def test_finds_matches_and_skips_noise(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(search_session, pattern="TODO")
        assert result.ok
        content = result.content
        assert "src/app.py" in content
        assert "src/utils.py" in content
        assert "README.md" in content
        # Skip dirs / binary suffixes must NOT appear.
        assert "node_modules/bad.js" not in content
        assert "image.png" not in content
        assert result.metadata["matches"] >= 3

    async def test_include_glob_narrows_scope(
        self, search_session: _FakeSession
    ) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(
            search_session,
            pattern="TODO",
            include="*.py",
        )
        assert result.ok
        assert "src/app.py" in result.content
        assert "src/utils.py" in result.content
        assert "README.md" not in result.content

    async def test_case_insensitive_flag(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(
            search_session,
            pattern="hello",
            case_insensitive=True,
        )
        assert result.ok
        assert "src/app.py" in result.content

    async def test_no_matches(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(search_session, pattern="nothing-here-at-all")
        assert result.ok
        assert "(no matches)" in result.content
        assert result.metadata["matches"] == 0

    async def test_bad_regex_is_rejected(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(search_session, pattern="(")
        assert not result.ok
        assert "bad regex" in (result.error or "")

    async def test_empty_pattern_is_rejected(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(search_session, pattern="")
        assert not result.ok

    async def test_escape_rejected(self, search_session: _FakeSession) -> None:
        tool = GrepWorkspaceTool()
        result = await tool.execute(
            search_session, pattern="TODO", path="../../../etc"
        )
        assert not result.ok
        assert "escape" in (result.error or "")


# ============================================================
# GlobWorkspaceTool
# ============================================================


class TestGlobWorkspace:
    async def test_finds_python_files_recursively(
        self, search_session: _FakeSession
    ) -> None:
        tool = GlobWorkspaceTool()
        result = await tool.execute(search_session, pattern="**/*.py")
        assert result.ok
        assert "src/app.py" in result.content
        assert "src/utils.py" in result.content
        # Not a .py — must not appear.
        assert "README.md" not in result.content
        # Skip dir must not appear.
        assert "node_modules" not in result.content

    async def test_flat_glob_in_root(self, search_session: _FakeSession) -> None:
        tool = GlobWorkspaceTool()
        result = await tool.execute(search_session, pattern="*.md")
        assert result.ok
        assert "README.md" in result.content

    async def test_subdirectory_scope(self, search_session: _FakeSession) -> None:
        tool = GlobWorkspaceTool()
        result = await tool.execute(
            search_session, pattern="*.py", path="src"
        )
        assert result.ok
        assert "src/app.py" in result.content

    async def test_empty_pattern_is_rejected(
        self, search_session: _FakeSession
    ) -> None:
        tool = GlobWorkspaceTool()
        result = await tool.execute(search_session, pattern="")
        assert not result.ok

    async def test_escape_rejected(self, search_session: _FakeSession) -> None:
        tool = GlobWorkspaceTool()
        result = await tool.execute(
            search_session, pattern="*.py", path="../../.."
        )
        assert not result.ok
        assert "escape" in (result.error or "")

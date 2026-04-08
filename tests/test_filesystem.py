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
    ListDirTool,
    PathEscapeError,
    ReadFileTool,
    WriteFileTool,
    _resolve_safe,
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

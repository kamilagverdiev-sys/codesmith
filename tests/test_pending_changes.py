"""Tests for the pending-change queue + tool rewire + apply_patch.

Covers:
- pending_changes module primitives (queue / list / reject / apply).
- write_file and edit_file, in both auto-approve and review modes.
- ApplyPatchTool validation, atomicity, and the review-mode single
  combined pending entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codesmith.pending_changes import (
    STATE_APPLIED,
    STATE_PENDING,
    STATE_REJECTED,
    _build_diff,
    apply_change,
    is_auto_approve,
    list_pending,
    queue_change,
    reject_change,
    set_auto_approve,
)
from codesmith.tools.filesystem import (
    ApplyPatchTool,
    EditFileTool,
    WriteFileTool,
)


@dataclass
class _FakeSession:
    workspace_dir: Path
    metadata: dict = field(default_factory=dict)


@pytest.fixture
def review_session(tmp_path: Path) -> _FakeSession:
    ws = tmp_path / "ws"
    ws.mkdir()
    return _FakeSession(workspace_dir=ws, metadata={"auto_approve": False})


@pytest.fixture
def auto_session(tmp_path: Path) -> _FakeSession:
    ws = tmp_path / "ws"
    ws.mkdir()
    return _FakeSession(workspace_dir=ws, metadata={"auto_approve": True})


# ============================================================
# pending_changes module primitives
# ============================================================


class TestPendingChangesModule:
    def test_default_is_auto_approve(self, tmp_path: Path) -> None:
        session = _FakeSession(workspace_dir=tmp_path / "ws", metadata={})
        assert is_auto_approve(session) is True

    def test_set_and_read_auto_approve(self, tmp_path: Path) -> None:
        session = _FakeSession(workspace_dir=tmp_path / "ws", metadata={})
        set_auto_approve(session, False)
        assert is_auto_approve(session) is False
        set_auto_approve(session, True)
        assert is_auto_approve(session) is True

    def test_queue_change_assigns_monotonic_idx(self, tmp_path: Path) -> None:
        session = _FakeSession(workspace_dir=tmp_path / "ws", metadata={})
        a = queue_change(session, kind="write_file", path="a.py", before="", after="x=1")
        b = queue_change(session, kind="write_file", path="b.py", before="", after="y=2")
        assert a["idx"] == 0
        assert b["idx"] == 1
        assert [c["state"] for c in list_pending(session)] == [STATE_PENDING, STATE_PENDING]

    def test_reject_change_transitions_state(self, tmp_path: Path) -> None:
        session = _FakeSession(workspace_dir=tmp_path / "ws", metadata={})
        change = queue_change(
            session, kind="write_file", path="a.py", before="", after="x=1"
        )
        updated = reject_change(session, change["idx"])
        assert updated is not None
        assert updated["state"] == STATE_REJECTED
        assert updated["applied_at"] is not None

    def test_apply_change_writes_to_disk(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _FakeSession(workspace_dir=ws, metadata={})
        change = queue_change(
            session,
            kind="write_file",
            path="hello.py",
            before="",
            after="print('hi')",
        )
        apply_change(session, change["idx"], workspace_dir=ws)
        assert (ws / "hello.py").read_text(encoding="utf-8") == "print('hi')"
        # State flips to applied.
        live = list_pending(session)[0]
        assert live["state"] == STATE_APPLIED
        assert live["applied_at"] is not None

    def test_apply_change_twice_raises_value_error(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _FakeSession(workspace_dir=ws, metadata={})
        change = queue_change(
            session, kind="write_file", path="x.py", before="", after="x=1"
        )
        apply_change(session, change["idx"], workspace_dir=ws)
        with pytest.raises(ValueError, match="already in terminal"):
            apply_change(session, change["idx"], workspace_dir=ws)

    def test_apply_change_missing_idx_raises_key_error(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        session = _FakeSession(workspace_dir=ws, metadata={})
        with pytest.raises(KeyError):
            apply_change(session, 99, workspace_dir=ws)

    def test_build_diff_has_unified_markers(self) -> None:
        text = _build_diff("a\nb\n", "a\nc\n", "file.py")
        assert "--- a/file.py" in text
        assert "+++ b/file.py" in text
        assert "-b" in text
        assert "+c" in text


# ============================================================
# write_file / edit_file in review mode
# ============================================================


class TestWriteFileReviewMode:
    async def test_new_file_goes_to_queue_not_disk(
        self, review_session: _FakeSession
    ) -> None:
        tool = WriteFileTool()
        result = await tool.execute(
            review_session,
            path="new.py",
            content="print('hi')",
        )
        assert result.ok
        assert "PENDING" in result.content
        assert result.metadata.get("pending") is True
        # File must NOT exist on disk.
        assert not (review_session.workspace_dir / "new.py").exists()
        # Change must be queued.
        pending = list_pending(review_session)
        assert len(pending) == 1
        assert pending[0]["kind"] == "write_file"
        assert pending[0]["before"] == ""
        assert pending[0]["after"] == "print('hi')"

    async def test_overwriting_existing_file_captures_before(
        self, review_session: _FakeSession
    ) -> None:
        target = review_session.workspace_dir / "x.py"
        target.write_text("old", encoding="utf-8")
        tool = WriteFileTool()
        await tool.execute(review_session, path="x.py", content="new")
        # Disk untouched.
        assert target.read_text(encoding="utf-8") == "old"
        pending = list_pending(review_session)[0]
        assert pending["before"] == "old"
        assert pending["after"] == "new"
        assert "-old" in pending["diff"]
        assert "+new" in pending["diff"]


class TestEditFileReviewMode:
    async def test_edit_queues_pending(
        self, review_session: _FakeSession
    ) -> None:
        target = review_session.workspace_dir / "main.py"
        target.write_text("x = 1\n", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            review_session, path="main.py", find="x = 1", replace="x = 2"
        )
        assert result.ok
        assert "PENDING" in result.content
        assert target.read_text(encoding="utf-8") == "x = 1\n"  # untouched
        pending = list_pending(review_session)
        assert len(pending) == 1
        assert pending[0]["kind"] == "edit_file"
        assert pending[0]["after"] == "x = 2\n"

    async def test_approval_applies_edit(
        self, review_session: _FakeSession
    ) -> None:
        target = review_session.workspace_dir / "main.py"
        target.write_text("x = 1\n", encoding="utf-8")
        tool = EditFileTool()
        result = await tool.execute(
            review_session, path="main.py", find="x = 1", replace="x = 2"
        )
        idx = result.metadata["pending_idx"]
        apply_change(
            review_session, idx, workspace_dir=review_session.workspace_dir
        )
        assert target.read_text(encoding="utf-8") == "x = 2\n"


# ============================================================
# apply_patch
# ============================================================


class TestApplyPatchTool:
    async def test_auto_approve_writes_all_files(
        self, auto_session: _FakeSession
    ) -> None:
        ws = auto_session.workspace_dir
        (ws / "a.py").write_text("foo = 1\n", encoding="utf-8")
        (ws / "b.py").write_text("foo = 2\n", encoding="utf-8")
        tool = ApplyPatchTool()
        result = await tool.execute(
            auto_session,
            edits=[
                {"path": "a.py", "find": "foo = 1", "replace": "bar = 1"},
                {"path": "b.py", "find": "foo = 2", "replace": "bar = 2"},
            ],
        )
        assert result.ok
        assert "2 edit" in result.content
        assert (ws / "a.py").read_text(encoding="utf-8") == "bar = 1\n"
        assert (ws / "b.py").read_text(encoding="utf-8") == "bar = 2\n"
        assert result.metadata["edits_applied"] == 2

    async def test_auto_approve_rolls_back_on_validation_fail(
        self, auto_session: _FakeSession
    ) -> None:
        """A bad edit (missing find) must abort the whole batch BEFORE
        any file is touched — even the valid edits."""
        ws = auto_session.workspace_dir
        (ws / "a.py").write_text("original\n", encoding="utf-8")
        (ws / "b.py").write_text("hello world\n", encoding="utf-8")
        tool = ApplyPatchTool()
        result = await tool.execute(
            auto_session,
            edits=[
                {"path": "a.py", "find": "original", "replace": "changed"},
                {"path": "b.py", "find": "NOT PRESENT", "replace": "whatever"},
            ],
        )
        assert not result.ok
        assert "find not present" in (result.error or "")
        # a.py was valid on its own, but the batch failed before any
        # write, so it must still hold the original content.
        assert (ws / "a.py").read_text(encoding="utf-8") == "original\n"
        assert (ws / "b.py").read_text(encoding="utf-8") == "hello world\n"

    async def test_review_mode_queues_single_batched_pending(
        self, review_session: _FakeSession
    ) -> None:
        ws = review_session.workspace_dir
        (ws / "a.py").write_text("foo = 1\n", encoding="utf-8")
        (ws / "b.py").write_text("foo = 2\n", encoding="utf-8")
        tool = ApplyPatchTool()
        result = await tool.execute(
            review_session,
            edits=[
                {"path": "a.py", "find": "foo = 1", "replace": "bar = 1"},
                {"path": "b.py", "find": "foo = 2", "replace": "bar = 2"},
            ],
        )
        assert result.ok
        assert "PENDING" in result.content
        # Files untouched.
        assert (ws / "a.py").read_text(encoding="utf-8") == "foo = 1\n"
        assert (ws / "b.py").read_text(encoding="utf-8") == "foo = 2\n"
        # One pending change that represents the whole batch.
        pending = list_pending(review_session)
        assert len(pending) == 1
        entry = pending[0]
        assert entry["kind"] == "apply_patch"
        assert len(entry["files"]) == 2
        assert {f["path"] for f in entry["files"]} == {"a.py", "b.py"}

    async def test_review_mode_approval_writes_all_files_atomically(
        self, review_session: _FakeSession
    ) -> None:
        ws = review_session.workspace_dir
        (ws / "a.py").write_text("foo = 1\n", encoding="utf-8")
        (ws / "b.py").write_text("foo = 2\n", encoding="utf-8")
        tool = ApplyPatchTool()
        result = await tool.execute(
            review_session,
            edits=[
                {"path": "a.py", "find": "foo = 1", "replace": "bar = 1"},
                {"path": "b.py", "find": "foo = 2", "replace": "bar = 2"},
            ],
        )
        idx = result.metadata["pending_idx"]
        apply_change(review_session, idx, workspace_dir=ws)
        assert (ws / "a.py").read_text(encoding="utf-8") == "bar = 1\n"
        assert (ws / "b.py").read_text(encoding="utf-8") == "bar = 2\n"

    async def test_empty_edits_list_rejected(
        self, auto_session: _FakeSession
    ) -> None:
        tool = ApplyPatchTool()
        result = await tool.execute(auto_session, edits=[])
        assert not result.ok

    async def test_noop_edit_rejected(
        self, auto_session: _FakeSession
    ) -> None:
        (auto_session.workspace_dir / "a.py").write_text("x = 1\n", encoding="utf-8")
        tool = ApplyPatchTool()
        result = await tool.execute(
            auto_session,
            edits=[{"path": "a.py", "find": "x = 1", "replace": "x = 1"}],
        )
        assert not result.ok
        assert "differ" in (result.error or "")

    async def test_ambiguous_match_rejected(
        self, auto_session: _FakeSession
    ) -> None:
        (auto_session.workspace_dir / "a.py").write_text(
            "foo\nfoo\n", encoding="utf-8"
        )
        tool = ApplyPatchTool()
        result = await tool.execute(
            auto_session,
            edits=[{"path": "a.py", "find": "foo", "replace": "bar"}],
        )
        assert not result.ok
        assert "matches 2 places" in (result.error or "")

    async def test_path_escape_rejected(
        self, auto_session: _FakeSession
    ) -> None:
        tool = ApplyPatchTool()
        result = await tool.execute(
            auto_session,
            edits=[{"path": "../escape.py", "find": "x", "replace": "y"}],
        )
        assert not result.ok
        assert "escape" in (result.error or "")

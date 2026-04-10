"""Tests for the first-turn workspace snapshot bootstrap.

The snapshot string is what gets prepended to the system prompt on
the very first user turn, so the agent starts with a grounded view
of the workspace rather than burning tool calls on list_directory.
"""

from __future__ import annotations

from pathlib import Path

from codesmith.workspace_snapshot import (
    bootstrap_system_prompt_addition,
    build_workspace_snapshot,
)


def _build_repo(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("# hi", encoding="utf-8")
    (ws / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
    (ws / "src" / "helpers.py").write_text("x = 1\n", encoding="utf-8")
    (ws / "tests").mkdir()
    (ws / "tests" / "test_app.py").write_text("def test_x(): pass\n", encoding="utf-8")
    # Junk that must not appear in the snapshot.
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "huge.js").write_text("// huge", encoding="utf-8")
    (ws / "__pycache__").mkdir()
    (ws / "_inspect").mkdir()
    (ws / ".claude").mkdir()
    return ws


def test_snapshot_lists_interesting_files_and_dirs(tmp_path: Path) -> None:
    ws = _build_repo(tmp_path)
    snapshot = build_workspace_snapshot(ws)
    assert str(ws) in snapshot
    assert "README.md" in snapshot
    assert "pyproject.toml" in snapshot
    assert "src/" in snapshot
    assert "app.py" in snapshot
    assert "tests/" in snapshot
    assert "test_app.py" in snapshot


def test_snapshot_skips_junk_and_hidden_dirs(tmp_path: Path) -> None:
    ws = _build_repo(tmp_path)
    snapshot = build_workspace_snapshot(ws)
    assert ".git" not in snapshot
    assert "node_modules" not in snapshot
    assert "__pycache__" not in snapshot
    assert "_inspect" not in snapshot
    assert ".claude" not in snapshot


def test_snapshot_empty_workspace_returns_empty_string(tmp_path: Path) -> None:
    ws = tmp_path / "empty"
    ws.mkdir()
    assert build_workspace_snapshot(ws) == ""


def test_snapshot_missing_workspace_returns_empty_string(tmp_path: Path) -> None:
    assert build_workspace_snapshot(tmp_path / "does-not-exist") == ""


def test_snapshot_respects_max_files_cap(tmp_path: Path) -> None:
    """Big repos must never produce unbounded snapshots."""
    ws = tmp_path / "ws"
    ws.mkdir()
    # Mix root files with nested dirs so both caps (root slice + per-dir
    # slice + max_files) get exercised.
    for i in range(30):
        (ws / f"root{i:03d}.py").write_text("x", encoding="utf-8")
    for i in range(20):
        d = ws / f"dir{i:03d}"
        d.mkdir()
        for j in range(20):
            (d / f"nested{j:03d}.py").write_text("x", encoding="utf-8")
    snapshot = build_workspace_snapshot(ws, max_files=15)
    # Total lines listed should stay near max_files.
    line_count = snapshot.count("\n")
    assert line_count <= 25
    # And it must clearly never contain all 30+400 entries.
    assert snapshot.count("root") + snapshot.count("nested") < 400


def test_bootstrap_system_prompt_addition_wraps_snapshot(tmp_path: Path) -> None:
    ws = _build_repo(tmp_path)
    block = bootstrap_system_prompt_addition(ws)
    assert "## Workspace snapshot" in block
    assert "```" in block
    assert "README.md" in block


def test_bootstrap_system_prompt_addition_empty_for_empty_ws(tmp_path: Path) -> None:
    assert bootstrap_system_prompt_addition(tmp_path / "nothing-here") == ""

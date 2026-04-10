"""Tests for the self-repair loop — retry logic, success detection, nudges."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from codesmith.agent import Agent, AgentRun, AgentStep
from codesmith.llm import LLMResponse
from codesmith.loops.self_repair import RepairResult, SelfRepairLoop
from codesmith.session import Session
from codesmith.tools.base import ToolResult


def _make_session(tmp_path: Path) -> Session:
    return Session(
        session_id="repair-test",
        messages=[],
        workspace_dir=tmp_path,
    )


def _dummy_llm_response() -> LLMResponse:
    """Minimal LLMResponse for tests that don't inspect it."""
    return LLMResponse(
        content="",
        tool_calls=[],
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        model="test/model",
        raw=None,
    )


def _make_step(
    *,
    ok: bool = True,
    exit_code: int | None = 0,
    content: str = "output",
    error: str | None = None,
) -> AgentStep:
    """Build an AgentStep with a single tool result."""
    metadata: dict = {}
    if exit_code is not None:
        metadata["exit_code"] = exit_code
    result = ToolResult(ok=ok, content=content, error=error, metadata=metadata)
    return AgentStep(
        response=_dummy_llm_response(),
        tool_results=[("tc-1", result)],
    )


def _make_run(
    *,
    steps: list[AgentStep] | None = None,
    hit_limit: bool = False,
    final_text: str = "done",
    total_tokens: int = 20,
) -> AgentRun:
    return AgentRun(
        steps=steps or [],
        final_text=final_text,
        total_tokens=total_tokens,
        hit_limit=hit_limit,
        forced_final=False,
    )


# ============================================================
# _check_success
# ============================================================


class TestCheckSuccess:
    def test_success_when_last_exec_ok(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=True)
        step = _make_step(ok=True, exit_code=0)
        run = _make_run(steps=[step])
        assert loop._check_success(run) is True

    def test_failure_when_last_exec_nonzero(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=True)
        step = _make_step(ok=False, exit_code=1, error="NameError")
        run = _make_run(steps=[step])
        assert loop._check_success(run) is False

    def test_failure_when_no_exec(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=True)
        # Step with no exit_code in metadata (e.g. read_file tool)
        result = ToolResult(ok=True, content="file contents")
        step = AgentStep(
            response=_dummy_llm_response(),
            tool_results=[("tc-1", result)],
        )
        run = _make_run(steps=[step])
        assert loop._check_success(run) is False

    def test_success_without_require_exec_when_not_hit_limit(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=False)
        run = _make_run(hit_limit=False)
        assert loop._check_success(run) is True

    def test_failure_without_require_exec_when_hit_limit(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=False)
        run = _make_run(hit_limit=True)
        assert loop._check_success(run) is False

    def test_last_exec_wins(self, tmp_path: Path) -> None:
        """If there are multiple exec steps, only the LAST one's exit code matters."""
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent, require_success_exec=True)
        step_fail = _make_step(ok=False, exit_code=1)
        step_ok = _make_step(ok=True, exit_code=0)
        run = _make_run(steps=[step_fail, step_ok])
        assert loop._check_success(run) is True


# ============================================================
# _build_nudge
# ============================================================


class TestBuildNudge:
    def test_nudge_on_hit_limit(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent)
        run = _make_run(hit_limit=True)
        nudge = loop._build_nudge(run)
        assert "iteration limit" in nudge.lower()

    def test_nudge_includes_last_error(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent)
        step = _make_step(ok=False, error="NameError: name 'foo' is not defined")
        run = _make_run(steps=[step])
        nudge = loop._build_nudge(run)
        assert "NameError" in nudge

    def test_nudge_generic_when_no_error(self, tmp_path: Path) -> None:
        agent = AsyncMock(spec=Agent)
        loop = SelfRepairLoop(agent)
        run = _make_run(steps=[])
        nudge = loop._build_nudge(run)
        assert "execute_python" in nudge


# ============================================================
# solve() — integration with mock agent
# ============================================================


class TestSolve:
    @pytest.mark.asyncio
    async def test_solve_succeeds_on_first_attempt(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        agent = AsyncMock(spec=Agent)
        step = _make_step(ok=True, exit_code=0)
        agent.run.return_value = _make_run(steps=[step], total_tokens=30)

        loop = SelfRepairLoop(agent, max_attempts=3)
        result = await loop.solve(session, "write hello world")

        assert isinstance(result, RepairResult)
        assert result.ok is True
        assert result.attempts == 1
        assert result.total_tokens == 30
        assert len(result.runs) == 1
        # Task was added as user message
        assert any(m.get("content") == "write hello world" for m in session.messages)

    @pytest.mark.asyncio
    async def test_solve_retries_on_failure(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        agent = AsyncMock(spec=Agent)

        fail_step = _make_step(ok=False, exit_code=1, error="SyntaxError")
        ok_step = _make_step(ok=True, exit_code=0)

        agent.run.side_effect = [
            _make_run(steps=[fail_step], total_tokens=20),
            _make_run(steps=[ok_step], total_tokens=25),
        ]

        loop = SelfRepairLoop(agent, max_attempts=3)
        result = await loop.solve(session, "fix the bug")

        assert result.ok is True
        assert result.attempts == 2
        assert result.total_tokens == 45
        assert len(result.runs) == 2

    @pytest.mark.asyncio
    async def test_solve_exhausts_all_attempts(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        agent = AsyncMock(spec=Agent)

        fail_step = _make_step(ok=False, exit_code=1, error="always fails")
        agent.run.return_value = _make_run(steps=[fail_step], total_tokens=10)

        loop = SelfRepairLoop(agent, max_attempts=2)
        result = await loop.solve(session, "impossible task")

        assert result.ok is False
        assert result.attempts == 2
        assert result.total_tokens == 20
        assert len(result.runs) == 2

    @pytest.mark.asyncio
    async def test_solve_adds_nudge_between_attempts(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        agent = AsyncMock(spec=Agent)

        fail_step = _make_step(ok=False, exit_code=1, error="NameError")
        ok_step = _make_step(ok=True, exit_code=0)
        agent.run.side_effect = [
            _make_run(steps=[fail_step], total_tokens=10),
            _make_run(steps=[ok_step], total_tokens=10),
        ]

        loop = SelfRepairLoop(agent, max_attempts=3)
        await loop.solve(session, "fix it")

        # There should be a nudge message between attempts
        user_messages = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_messages) >= 2  # original task + nudge
        # The nudge should mention the error
        nudge = user_messages[1]["content"]
        assert "NameError" in nudge

    @pytest.mark.asyncio
    async def test_solve_single_attempt_no_nudge(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        agent = AsyncMock(spec=Agent)

        fail_step = _make_step(ok=False, exit_code=1)
        agent.run.return_value = _make_run(steps=[fail_step])

        loop = SelfRepairLoop(agent, max_attempts=1)
        result = await loop.solve(session, "one shot")

        assert result.ok is False
        assert result.attempts == 1
        # Only the original task, no nudge
        user_messages = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_messages) == 1


# ============================================================
# RepairResult dataclass
# ============================================================


def test_repair_result_fields() -> None:
    r = RepairResult(ok=True, attempts=2, final_text="done", runs=[], total_tokens=100)
    assert r.ok is True
    assert r.attempts == 2
    assert r.final_text == "done"
    assert r.total_tokens == 100

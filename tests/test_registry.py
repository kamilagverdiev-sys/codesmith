"""Tests for codesmith.tools.base and codesmith.tools.registry.

These verify that the tool layer handles the three failure modes the
agent will actually hit in production:
  1. LLM requests a tool that doesn't exist
  2. LLM calls a tool with wrong/missing arguments
  3. Tool raises an unexpected exception
In all three cases the registry must return a ToolResult(ok=False, ...)
so the agent loop can keep running and the LLM can see the error.
"""

from __future__ import annotations

import pytest

from codesmith.tools.base import BaseTool, ToolResult, tool
from codesmith.tools.registry import ToolRegistry


class _DummySession:
    """Bare stand-in — tools in these tests don't touch the session."""


# ============================================================
# ToolResult
# ============================================================


class TestToolResult:
    def test_ok_result_serializes_plain(self) -> None:
        r = ToolResult(ok=True, content="42")
        assert r.to_llm_content() == "42"

    def test_error_result_prefixed(self) -> None:
        r = ToolResult(
            ok=False, content="traceback line", error="ValueError: bad input"
        )
        out = r.to_llm_content()
        assert out.startswith("[error]")
        assert "ValueError" in out
        assert "traceback line" in out


# ============================================================
# @tool decorator
# ============================================================


def _make_echo_tool() -> BaseTool:
    @tool(
        name="echo",
        description="echo back the msg",
        parameters={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    async def _echo(msg: str, session: _DummySession) -> ToolResult:
        return ToolResult(ok=True, content=msg)

    return _echo  # type: ignore[return-value]


class TestDecorator:
    def test_produces_basetool(self) -> None:
        t = _make_echo_tool()
        assert isinstance(t, BaseTool)
        assert t.name == "echo"

    def test_schema_matches_openai_format(self) -> None:
        schema = _make_echo_tool().schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert "msg" in schema["function"]["parameters"]["properties"]


# ============================================================
# ToolRegistry
# ============================================================


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_make_echo_tool())
    return reg


class TestRegistry:
    def test_names_lists_registered(self, registry: ToolRegistry) -> None:
        assert registry.names() == ["echo"]

    def test_schemas_single_entry(self, registry: ToolRegistry) -> None:
        schemas = registry.schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "echo"

    def test_register_many_bulk_insert(self) -> None:
        reg = ToolRegistry()
        reg.register_many([_make_echo_tool(), _make_echo_tool()])
        # Second register overrides — same name → still one tool
        assert reg.names() == ["echo"]

    async def test_dispatch_happy_path(self, registry: ToolRegistry) -> None:
        result = await registry.dispatch(
            "echo", {"msg": "hello"}, _DummySession()
        )
        assert result.ok
        assert result.content == "hello"

    async def test_dispatch_unknown_tool_returns_error(
        self, registry: ToolRegistry
    ) -> None:
        """Never raise on unknown tool — return a failed ToolResult instead."""
        result = await registry.dispatch("nonexistent", {}, _DummySession())
        assert not result.ok
        assert "unknown" in (result.error or "").lower()

    async def test_dispatch_bad_args_returns_error(
        self, registry: ToolRegistry
    ) -> None:
        """Wrong argument name → TypeError → failed ToolResult, not a crash."""
        result = await registry.dispatch(
            "echo", {"wrong_param": 1}, _DummySession()
        )
        assert not result.ok
        assert "bad arguments" in (result.error or "").lower()

    async def test_dispatch_tool_exception_becomes_error(self) -> None:
        """A tool that raises must not blow up the agent loop."""
        reg = ToolRegistry()

        @tool(
            name="boom",
            description="always crashes",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        async def _boom(session: _DummySession) -> ToolResult:
            raise RuntimeError("kaboom")

        reg.register(_boom)  # type: ignore[arg-type]

        result = await reg.dispatch("boom", {}, _DummySession())
        assert not result.ok
        assert "RuntimeError" in (result.error or "")
        assert "kaboom" in (result.error or "")

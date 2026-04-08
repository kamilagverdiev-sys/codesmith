"""Base tool interface.

A Tool is anything the agent can invoke: execute_python, read_file,
web_search, memory_search, etc. All tools share one shape:

  - name + description + JSON Schema for parameters (used by the LLM)
  - async execute(session, **kwargs) -> ToolResult

ToolResult carries both the content the LLM will see AND structured
metadata for logging and telemetry (token usage, exit codes, etc.).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesmith.session import Session


@dataclass
class ToolResult:
    """What a tool returns after execution."""

    ok: bool                           # did it succeed?
    content: str                       # what the LLM sees in the tool_result message
    metadata: dict[str, Any] = field(default_factory=dict)  # logs/telemetry
    error: str | None = None           # human-readable error if ok=False

    def to_llm_content(self) -> str:
        """Serialize for the LLM. Prefix errors so the model notices."""
        if self.ok:
            return self.content
        return f"[error] {self.error or 'unknown'}\n{self.content}".rstrip()


class BaseTool(ABC):
    """Abstract tool. Subclass OR use the @tool decorator for functions."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema, OpenAI function-calling format

    @abstractmethod
    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        """Run the tool. Must be async (wrap sync code with asyncio.to_thread)."""

    def schema(self) -> dict[str, Any]:
        """OpenAI/LiteLLM function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


ToolFunc = Callable[..., Awaitable[ToolResult] | ToolResult]


class FunctionTool(BaseTool):
    """A BaseTool wrapping a plain function. Used by @tool decorator."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: ToolFunc,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._func = func

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        result = self._func(session=session, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        if not isinstance(result, ToolResult):
            # Allow plain strings for convenience in tests/stubs
            result = ToolResult(ok=True, content=str(result))
        return result


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable[[ToolFunc], FunctionTool]:
    """Decorator to turn a function into a Tool.

    Usage:
        @tool(
            name="list_dir",
            description="List files in a directory",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        async def list_dir(path: str, session: Session) -> ToolResult:
            ...
    """
    def wrap(func: ToolFunc) -> FunctionTool:
        return FunctionTool(name, description, parameters, func)
    return wrap

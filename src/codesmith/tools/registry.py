"""Tool registry.

Holds a set of tools by name, exports their JSON schemas to give to the
LLM, and dispatches tool calls by name. Dispatch catches exceptions so
one broken tool can't crash the agent loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from codesmith.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from codesmith.session import Session

log = logging.getLogger(__name__)


class ToolRegistry:
    """Name → BaseTool mapping."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            log.warning("overriding existing tool: %s", tool.name)
        self._tools[tool.name] = tool

    def register_many(self, tools: list[BaseTool]) -> None:
        for t in tools:
            self.register(t)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """JSON schemas for every registered tool — feed to LLM."""
        return [t.schema() for t in self._tools.values()]

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        session: Session,
    ) -> ToolResult:
        """Look up tool by name and execute it.

        Exceptions become failed ToolResults — the agent keeps going,
        the LLM sees the error in the next turn and can react to it.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                content="",
                error=f"unknown tool: {name}. Known: {self.names()}",
            )
        try:
            result = await tool.execute(session, **arguments)
            log.debug("tool=%s ok=%s", name, result.ok)
            return result
        except TypeError as e:
            return ToolResult(
                ok=False,
                content="",
                error=f"bad arguments to {name}: {e}",
            )
        except Exception as e:
            log.exception("tool %s crashed", name)
            return ToolResult(
                ok=False,
                content="",
                error=f"{type(e).__name__}: {e}",
            )

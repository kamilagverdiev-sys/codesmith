"""Web search tool — Tavily and Brave Search API integration.

Gives the agent the ability to look up recent information, documentation,
error messages, and API references. Rate-limited per session via
``config.tools.web_search.max_calls_per_session``.

Requires one of:
  - TAVILY_API_KEY (default provider)
  - BRAVE_SEARCH_API_KEY

Install: ``pip install codesmith[web]``   (adds httpx)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from codesmith.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from codesmith.config import WebSearchToolConfig
    from codesmith.session import Session

log = logging.getLogger(__name__)

_USAGE_KEY = "_web_search_calls"


class WebSearchTool(BaseTool):
    """Search the web and return a concise summary of top results."""

    name = "web_search"
    description = (
        "Search the web for recent information, documentation, error messages, "
        "or API references. Returns the top results with titles, URLs, and "
        "content snippets. Use when you need up-to-date information that "
        "isn't in the workspace."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, config: WebSearchToolConfig) -> None:
        self._provider = config.provider
        self._max_calls = config.max_calls_per_session

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        if not query or not query.strip():
            return ToolResult(ok=False, content="", error="Empty search query.")

        max_results = min(int(kwargs.get("max_results", 5)), 10)

        # Rate limiting per session.
        used = session.metadata.get(_USAGE_KEY, 0)
        if used >= self._max_calls:
            return ToolResult(
                ok=False,
                content="",
                error=(
                    f"Web search limit reached ({self._max_calls} calls per session). "
                    "Use the results you already have."
                ),
            )
        session.metadata[_USAGE_KEY] = used + 1

        try:
            if self._provider == "brave":
                return await self._search_brave(query, max_results)
            return await self._search_tavily(query, max_results)
        except ImportError:
            return ToolResult(
                ok=False,
                content="",
                error="httpx not installed. Run: pip install codesmith[web]",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("web search failed: %s: %s", type(e).__name__, e)
            return ToolResult(
                ok=False,
                content="",
                error=f"Search failed: {type(e).__name__}: {e}",
            )

    async def _search_tavily(self, query: str, max_results: int) -> ToolResult:
        """Tavily Search API — optimized for LLM consumption."""
        import httpx

        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return ToolResult(
                ok=False,
                content="",
                error="TAVILY_API_KEY not set. Add it to .env.",
            )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        parts: list[str] = []
        answer = data.get("answer")
        if answer:
            parts.append(f"**Quick answer:** {answer}\n")

        results = data.get("results", [])
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            # Trim content to keep context reasonable.
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"{i}. **{title}**\n   {url}\n   {content}\n")

        if not parts:
            return ToolResult(ok=True, content="No results found for this query.")

        return ToolResult(
            ok=True,
            content="\n".join(parts),
            metadata={"query": query, "result_count": len(results)},
        )

    async def _search_brave(self, query: str, max_results: int) -> ToolResult:
        """Brave Search API."""
        import httpx

        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            return ToolResult(
                ok=False,
                content="",
                error="BRAVE_SEARCH_API_KEY not set. Add it to .env.",
            )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                params={"q": query, "count": max_results},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("web", {}).get("results", [])
        parts: list[str] = []
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "")
            url = r.get("url", "")
            description = r.get("description", "")
            if len(description) > 500:
                description = description[:500] + "..."
            parts.append(f"{i}. **{title}**\n   {url}\n   {description}\n")

        if not parts:
            return ToolResult(ok=True, content="No results found for this query.")

        return ToolResult(
            ok=True,
            content="\n".join(parts),
            metadata={"query": query, "result_count": len(results)},
        )

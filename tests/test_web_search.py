"""Tests for the web_search tool module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codesmith.config import WebSearchToolConfig
from codesmith.session import Session
from codesmith.tools.web_search import WebSearchTool


def _make_session() -> Session:
    return Session(
        session_id="test-ws",
        messages=[],
        workspace_dir=Path("/tmp/fake"),
    )


def _make_tool(
    provider: str = "tavily",
    max_calls: int = 10,
) -> WebSearchTool:
    cfg = WebSearchToolConfig(
        enabled=True,
        provider=provider,
        max_calls_per_session=max_calls,
    )
    return WebSearchTool(cfg)


def _mock_httpx_client(response_data: dict, method: str = "post") -> MagicMock:
    """Build a mock httpx.AsyncClient that returns *response_data* from the
    given HTTP method (post or get). The mock is usable as an async context
    manager, matching ``async with httpx.AsyncClient() as client:``.

    httpx.Response.json() and .raise_for_status() are synchronous, so the
    response object is a regular MagicMock — NOT an AsyncMock.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_data

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    setattr(mock_client, method, AsyncMock(return_value=mock_response))
    return mock_client


# ---- schema ----


def test_tool_name_and_description() -> None:
    tool = _make_tool()
    assert tool.name == "web_search"
    assert "search" in tool.description.lower()


def test_parameters_require_query() -> None:
    tool = _make_tool()
    assert "query" in tool.parameters["required"]


# ---- empty query ----


@pytest.mark.asyncio
async def test_empty_query_returns_error() -> None:
    tool = _make_tool()
    session = _make_session()
    result = await tool.execute(session, query="")
    assert result.ok is False
    assert "Empty" in (result.error or "")


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_error() -> None:
    tool = _make_tool()
    session = _make_session()
    result = await tool.execute(session, query="   ")
    assert result.ok is False
    assert "Empty" in (result.error or "")


# ---- rate limiting ----


@pytest.mark.asyncio
async def test_rate_limit_blocks_after_max_calls() -> None:
    tool = _make_tool(max_calls=2)
    session = _make_session()

    mock_client = _mock_httpx_client(
        {"answer": "test", "results": [{"title": "A", "url": "https://a.com", "content": "A"}]},
    )

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        r1 = await tool.execute(session, query="first")
        assert r1.ok is True
        r2 = await tool.execute(session, query="second")
        assert r2.ok is True

    # Third call should fail with rate limit — doesn't even hit the API.
    r3 = await tool.execute(session, query="third")
    assert r3.ok is False
    assert "limit reached" in (r3.error or "").lower()


@pytest.mark.asyncio
async def test_rate_limit_counter_persists_in_session_metadata() -> None:
    tool = _make_tool(max_calls=5)
    session = _make_session()

    mock_client = _mock_httpx_client({"results": []})

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        await tool.execute(session, query="q1")

    assert session.metadata.get("_web_search_calls") == 1


# ---- Tavily provider ----


@pytest.mark.asyncio
async def test_tavily_missing_api_key_returns_error() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    with patch.dict("os.environ", {}, clear=True):
        result = await tool.execute(session, query="python docs")

    assert result.ok is False
    assert "TAVILY_API_KEY" in (result.error or "")


@pytest.mark.asyncio
async def test_tavily_returns_formatted_results() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    mock_client = _mock_httpx_client({
        "answer": "Python is a programming language.",
        "results": [
            {"title": "Python.org", "url": "https://python.org", "content": "Welcome to Python"},
            {"title": "PyPI", "url": "https://pypi.org", "content": "Package index for Python"},
        ],
    })

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await tool.execute(session, query="python", max_results=2)

    assert result.ok is True
    assert "Python is a programming language" in result.content
    assert "Python.org" in result.content
    assert "PyPI" in result.content
    assert result.metadata is not None
    assert result.metadata["query"] == "python"
    assert result.metadata["result_count"] == 2


@pytest.mark.asyncio
async def test_tavily_no_results() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    mock_client = _mock_httpx_client({"results": []})

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await tool.execute(session, query="xyznonexistent")

    assert result.ok is True
    assert "No results" in result.content


# ---- Brave provider ----


@pytest.mark.asyncio
async def test_brave_missing_api_key_returns_error() -> None:
    tool = _make_tool(provider="brave")
    session = _make_session()

    with patch.dict("os.environ", {}, clear=True):
        result = await tool.execute(session, query="python docs")

    assert result.ok is False
    assert "BRAVE_SEARCH_API_KEY" in (result.error or "")


@pytest.mark.asyncio
async def test_brave_returns_formatted_results() -> None:
    tool = _make_tool(provider="brave")
    session = _make_session()

    mock_client = _mock_httpx_client(
        {
            "web": {
                "results": [
                    {
                        "title": "MDN Web Docs",
                        "url": "https://developer.mozilla.org",
                        "description": "Resources for developers",
                    },
                ],
            },
        },
        method="get",
    )

    with (
        patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": "test-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await tool.execute(session, query="javascript docs", max_results=3)

    assert result.ok is True
    assert "MDN Web Docs" in result.content
    assert "developer.mozilla.org" in result.content
    assert result.metadata is not None
    assert result.metadata["query"] == "javascript docs"


# ---- httpx not installed ----


@pytest.mark.asyncio
async def test_httpx_not_installed_error() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    import builtins
    import sys

    real_import = builtins.__import__

    # Remove httpx from sys.modules temporarily so the lazy import triggers.
    saved_httpx = sys.modules.pop("httpx", None)

    def mock_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    try:
        with (
            patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            result = await tool.execute(session, query="test")
    finally:
        # Restore httpx in sys.modules.
        if saved_httpx is not None:
            sys.modules["httpx"] = saved_httpx

    assert result.ok is False
    assert "httpx" in (result.error or "").lower()


# ---- max_results clamped ----


@pytest.mark.asyncio
async def test_max_results_clamped_to_10() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    mock_client = _mock_httpx_client({"results": []})

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        await tool.execute(session, query="test", max_results=50)

    # Verify the API was called with max_results=10 (clamped).
    call_args = mock_client.post.call_args
    request_body = call_args.kwargs.get("json") or call_args[1].get("json")
    assert request_body["max_results"] == 10


# ---- content truncation ----


@pytest.mark.asyncio
async def test_tavily_long_content_truncated() -> None:
    tool = _make_tool(provider="tavily")
    session = _make_session()

    long_content = "x" * 1000

    mock_client = _mock_httpx_client({
        "results": [
            {"title": "Long", "url": "https://long.com", "content": long_content},
        ],
    })

    with (
        patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}),
        patch("httpx.AsyncClient", return_value=mock_client),
    ):
        result = await tool.execute(session, query="test")

    assert result.ok is True
    # Content should be truncated (500 chars + "...")
    assert "..." in result.content
    assert len(result.content) < len(long_content) + 200

"""Tests for the LLM layer — LLMProvider, LLMRouter, LLMResponse.

These tests mock litellm.acompletion so they never hit a real LLM.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from codesmith.config import LLMConfig
from codesmith.llm import LLMError, LLMProvider, LLMResponse, LLMRouter

# ---- helpers ----


def _make_config(**overrides: Any) -> LLMConfig:
    defaults = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _fake_response(
    content: str = "Hello!",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    """Build a fake litellm response matching the real structure."""
    tc_objs = []
    for tc in tool_calls or []:
        tc_objs.append(
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
        )

    message = SimpleNamespace(
        content=content,
        tool_calls=tc_objs or None,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# ============================================================
# LLMProvider
# ============================================================


class TestLLMProvider:
    def test_model_string_prefixed_when_no_slash(self) -> None:
        cfg = _make_config(provider="anthropic", model="claude-sonnet-4-6")
        p = LLMProvider(cfg)
        assert p.model == "anthropic/claude-sonnet-4-6"

    def test_model_string_unchanged_when_slash_present(self) -> None:
        cfg = _make_config(provider="openai", model="openai/gpt-4o")
        p = LLMProvider(cfg)
        assert p.model == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_chat_returns_llm_response(self) -> None:
        cfg = _make_config()
        provider = LLMProvider(cfg)
        fake = _fake_response(content="Hi there")

        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, return_value=fake):
            resp = await provider.chat([{"role": "user", "content": "hello"}])

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hi there"
        assert resp.finish_reason == "stop"
        assert resp.usage["total_tokens"] == 15
        assert resp.tool_calls == []
        assert resp.raw is fake

    @pytest.mark.asyncio
    async def test_chat_normalizes_tool_calls(self) -> None:
        cfg = _make_config()
        provider = LLMProvider(cfg)
        tc = [{"id": "tc1", "function": {"name": "foo", "arguments": '{"x": 1}'}}]
        fake = _fake_response(content="", tool_calls=tc)

        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, return_value=fake):
            resp = await provider.chat(
                [{"role": "user", "content": "call foo"}],
                tools=[{"type": "function", "function": {"name": "foo", "parameters": {}}}],
            )

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["id"] == "tc1"
        assert resp.tool_calls[0]["type"] == "function"
        assert resp.tool_calls[0]["function"]["name"] == "foo"

    @pytest.mark.asyncio
    async def test_chat_passes_api_key_and_base(self) -> None:
        cfg = _make_config(api_key="sk-secret", api_base="http://localhost:8080")
        provider = LLMProvider(cfg)
        fake = _fake_response()

        mock_acomp = AsyncMock(return_value=fake)
        with patch("codesmith.llm.acompletion", mock_acomp):
            await provider.chat([{"role": "user", "content": "hi"}])

        call_kwargs = mock_acomp.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-secret"
        assert call_kwargs["api_base"] == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_chat_handles_none_usage(self) -> None:
        cfg = _make_config()
        provider = LLMProvider(cfg)
        fake = _fake_response()
        fake.usage = None

        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, return_value=fake):
            resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert resp.usage == {}

    @pytest.mark.asyncio
    async def test_chat_handles_none_content(self) -> None:
        cfg = _make_config()
        provider = LLMProvider(cfg)
        fake = _fake_response()
        fake.choices[0].message.content = None

        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, return_value=fake):
            resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert resp.content == ""


# ============================================================
# LLMRouter
# ============================================================


class TestLLMRouter:
    @pytest.mark.asyncio
    async def test_primary_success(self) -> None:
        primary = _make_config(provider="openai", model="gpt-4o")
        router = LLMRouter(primary=primary)

        fake = _fake_response(content="from primary")
        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, return_value=fake):
            resp = await router.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "from primary"

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self) -> None:
        primary = _make_config(provider="openai", model="gpt-4o")
        fallback = _make_config(provider="anthropic", model="claude-sonnet-4-6")
        router = LLMRouter(primary=primary, fallbacks=[fallback])

        call_count = 0
        fake = _fake_response(content="from fallback")

        async def side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("primary down")
            return fake

        with patch("codesmith.llm.acompletion", new_callable=AsyncMock, side_effect=side_effect):
            resp = await router.chat([{"role": "user", "content": "hi"}])

        assert resp.content == "from fallback"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_llm_error(self) -> None:
        primary = _make_config(provider="openai", model="gpt-4o")
        fallback = _make_config(provider="anthropic", model="claude-sonnet-4-6")
        router = LLMRouter(primary=primary, fallbacks=[fallback])

        with patch(
            "codesmith.llm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ), pytest.raises(LLMError, match="All LLM providers failed"):
            await router.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_error_message_includes_all_failures(self) -> None:
        primary = _make_config(provider="a", model="m1")
        fallback = _make_config(provider="b", model="m2")
        router = LLMRouter(primary=primary, fallbacks=[fallback])

        call_count = 0

        async def side_effect(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first error")
            raise TimeoutError("second error")

        with (
            patch("codesmith.llm.acompletion", new_callable=AsyncMock, side_effect=side_effect),
            pytest.raises(LLMError) as exc_info,
        ):
            await router.chat([{"role": "user", "content": "hi"}])

        error_text = str(exc_info.value)
        assert "first error" in error_text
        assert "second error" in error_text

    def test_router_with_no_fallbacks(self) -> None:
        primary = _make_config()
        router = LLMRouter(primary=primary)
        assert len(router.providers) == 1

    def test_router_with_multiple_fallbacks(self) -> None:
        primary = _make_config()
        fb1 = _make_config(provider="a", model="m1")
        fb2 = _make_config(provider="b", model="m2")
        router = LLMRouter(primary=primary, fallbacks=[fb1, fb2])
        assert len(router.providers) == 3

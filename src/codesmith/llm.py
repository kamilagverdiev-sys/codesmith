"""LLM provider abstraction using LiteLLM.

LiteLLM handles the 100+ provider translations; we add:
  - config-driven instantiation
  - a simple fallback chain
  - uniform tool-calling interface
  - logging of token usage

Do NOT add provider-specific branches here. If something only works
for Anthropic or only for OpenAI, route it through LiteLLM parameters,
not if/else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import acompletion
from litellm.types.completion import ChatCompletionMessageParam as Message

from codesmith.config import LLMConfig

log = logging.getLogger(__name__)

# Tell LiteLLM not to print its own noisy logs
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Normalized LLM response."""

    content: str
    tool_calls: list[dict[str, Any]]  # OpenAI-format tool calls
    finish_reason: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    model: str
    raw: Any  # original litellm response for advanced use


class LLMError(Exception):
    """LLM call failed after all fallbacks exhausted."""


class LLMProvider:
    """Single-provider LLM caller."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Build LiteLLM-style model string. LiteLLM accepts both
        # "claude-opus-4-6" and "anthropic/claude-opus-4-6". The prefixed
        # form is safer when the same model name exists across providers.
        if "/" in config.model:
            self.model = config.model
        else:
            self.model = f"{config.provider}/{config.model}"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: list of messages in OpenAI/LiteLLM format
            tools: optional list of tool schemas (OpenAI function-calling format)
            **extra: passed through to litellm.acompletion

        Returns:
            Normalized LLMResponse.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        kwargs.update(extra)

        log.debug("llm.chat model=%s msgs=%d tools=%d", self.model, len(messages), len(tools or []))

        response = await acompletion(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        # Normalize tool calls into plain dicts (LiteLLM returns pydantic-ish objects)
        tool_calls: list[dict[str, Any]] = []
        raw_tc = getattr(msg, "tool_calls", None) or []
        for tc in raw_tc:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # JSON string
                    },
                }
            )

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=self.model,
            raw=response,
        )


class LLMRouter:
    """LLM caller with a fallback chain.

    Tries the primary provider first. On failure, iterates through fallbacks.
    This is how the hybrid online/offline story becomes real:
        primary=claude → fallback=gpt → fallback=ollama(local)
    """

    def __init__(self, primary: LLMConfig, fallbacks: list[LLMConfig] | None = None) -> None:
        self.providers: list[LLMProvider] = [LLMProvider(primary)]
        for f in fallbacks or []:
            self.providers.append(LLMProvider(f))

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> LLMResponse:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return await provider.chat(messages, tools=tools, **extra)
            except Exception as e:
                msg = f"{provider.model}: {type(e).__name__}: {e}"
                log.warning("llm provider failed, trying fallback: %s", msg)
                errors.append(msg)
                continue
        raise LLMError(
            "All LLM providers failed:\n  " + "\n  ".join(errors)
        )

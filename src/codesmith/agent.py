"""Agent — the main agentic loop.

Responsibilities:
  1. Own an LLMRouter and a ToolRegistry.
  2. Run a loop: LLM call → if tool calls, execute them → feed results
     back → repeat, until the LLM stops calling tools or we hit a limit.
  3. Stay dumb. Anything fancier (self-repair, multi-agent, planning)
     lives in codesmith/loops/*, not here.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from codesmith.llm import LLMResponse, LLMRouter
from codesmith.personas import DEFAULT as DEFAULT_PERSONA
from codesmith.tools.base import ToolResult
from codesmith.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from codesmith.config import Config
    from codesmith.session import Session

StepCallback = Callable[["AgentStep", int], Awaitable[None]]

log = logging.getLogger(__name__)

_SMALL_TALK_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "how are you",
    "who are you",
    "what can you do",
    "how do you work",
    "привет",
    "здравствуй",
    "здравствуйте",
    "доброе утро",
    "добрый день",
    "добрый вечер",
    "спасибо",
    "как дела",
    "кто ты",
    "что ты умеешь",
    "что ты можешь",
    "как ты работаешь",
}

_SMALL_TALK_PREFIXES = ("а ", "ну ")

_TOOL_HINT_PATTERNS = (
    r"[/\\]",
    r"```",
    r"\b[a-zA-Z0-9_.-]+\.(py|js|ts|tsx|json|yaml|yml|toml|md|txt|csv|html|css|sql|sh|ps1)\b",
    r"\b(fix|debug|run|test|write|create|edit|refactor|implement|inspect|analyze|read|open|update|install|configure|build|execute|search|find|list|rename|delete|move|generate|solve|review|check)\b",
    r"(исправ|почин|запуст|протест|напиш|созда|отредакт|рефактор|реализ|посмотр|проанализ|прочит|открой|обнови|установ|настро|собер|выполн|найд|спис|переимен|удал|перемест|сгенер|реши|проверь)",
    r"\b(file|folder|directory|repo|repository|project|workspace|code|script|function|class|module|package|api|endpoint|bug|error|test|docker|container|config|python|javascript|typescript|html|css|sql|git|branch|commit|pull request)\b",
    r"(файл|папк|директор|репо|репозитор|проект|воркспейс|код|скрипт|функц|класс|модул|пакет|api|эндпоинт|баг|ошибк|тест|докер|контейнер|конфиг|питон|ветк|коммит)",
)

_SYNTHETIC_ANSWER_TOOL_NAMES = {
    "answer",
    "final",
    "final_answer",
    "respond",
    "response",
    "reply",
}

_ANSWER_ARGUMENT_KEYS = ("text", "answer", "response", "content", "message")

_MAX_IDENTICAL_STEP_REPEATS = 3

# Window-based "no progress" detector: look at the last N tool-calling steps,
# and if there are fewer than K distinct tool-call signatures among them, the
# model is thrashing even though individual steps are not byte-identical.
_NO_PROGRESS_WINDOW = 6
_MIN_UNIQUE_SIGNATURES_IN_WINDOW = 3

# Soft limit: if we're about to hit max_iterations, stop burning budget on
# another half-useful tool call and ask the model for a plain final answer
# instead. Triggers when remaining_iterations <= _SOFT_LIMIT_TAIL.
_SOFT_LIMIT_TAIL = 2

_FORCED_FINALIZATION_PROMPT = """\
You are finishing a run because the model is repeating itself or hit a guardrail.

Do not call any tools.
Give the user the best final answer you can from the verified tool results already in the conversation.
If the task is incomplete, say exactly what succeeded, what failed, and the next concrete step.
"""


# The default system prompt is owned by codesmith/personas.py. Keep a
# module-level alias so existing callers and tests that reference
# `DEFAULT_SYSTEM_PROMPT` from codesmith.agent keep working unchanged.
DEFAULT_SYSTEM_PROMPT = DEFAULT_PERSONA.system_prompt


def _normalize_user_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s]+", " ", text.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for prefix in _SMALL_TALK_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):].strip()
    return normalized


def should_offer_tools_for_text(text: str) -> bool:
    """Return True only when the prompt likely needs workspace actions."""
    normalized = _normalize_user_text(text)
    if not normalized:
        return True
    if normalized in _SMALL_TALK_MESSAGES:
        return False
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _TOOL_HINT_PATTERNS
    )


def _parse_tool_arguments(raw_args: object) -> dict[str, object]:
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {}
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_answer_text(payload: dict[str, object]) -> str:
    for key in _ANSWER_ARGUMENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_synthetic_answer_from_content(content: str) -> str:
    candidate = content.strip()
    if not candidate:
        return ""
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        candidate = candidate.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    function = payload.get("function")
    if not isinstance(function, dict):
        return ""
    name = str(function.get("name") or "").strip().casefold()
    if name not in _SYNTHETIC_ANSWER_TOOL_NAMES:
        return ""
    args = _parse_tool_arguments(function.get("arguments"))
    return _extract_answer_text(args)


def normalize_response_for_model_quirks(response: LLMResponse) -> LLMResponse:
    """Coerce pseudo-tools like `answer(text=...)` into plain assistant text."""
    if len(response.tool_calls) == 1:
        tool_call = response.tool_calls[0]
        function = tool_call.get("function", {})
        name = str(function.get("name") or "").strip().casefold()
        if name in _SYNTHETIC_ANSWER_TOOL_NAMES:
            text = _extract_answer_text(_parse_tool_arguments(function.get("arguments")))
            if text:
                return replace(
                    response,
                    content=text,
                    tool_calls=[],
                    finish_reason="stop",
                )

    text = _extract_synthetic_answer_from_content(response.content)
    if text:
        return replace(
            response,
            content=text,
            tool_calls=[],
            finish_reason="stop",
        )

    return response


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def _tool_call_signature(tool_call: dict[str, Any]) -> tuple[str, str]:
    function = tool_call.get("function", {})
    name = str(function.get("name") or "")
    args = _parse_tool_arguments(function.get("arguments"))
    return (name, _stable_json(args))


def _tool_result_signature(result: ToolResult) -> tuple[bool, str, str, int | None]:
    preview = (result.to_llm_content() or "")[:240]
    exit_code = result.metadata.get("exit_code")
    return (result.ok, result.error or "", preview, exit_code)


def step_signature(step: AgentStep) -> tuple[Any, ...]:
    return (
        step.response.content.strip(),
        tuple(_tool_call_signature(tc) for tc in step.response.tool_calls),
        tuple(_tool_result_signature(result) for _tcid, result in step.tool_results),
    )


@dataclass
class AgentStep:
    """One iteration of the agent loop. Used for logging and debugging."""

    response: LLMResponse
    tool_results: list[tuple[str, ToolResult]]  # (tool_call_id, result)


@dataclass
class AgentRun:
    """Result of Agent.run(). The final response is the last LLM message."""

    final_text: str
    steps: list[AgentStep]
    hit_limit: bool
    total_tokens: int
    forced_final: bool = False


class Agent:
    """One generic agent — LLM + tools + simple loop."""

    def __init__(
        self,
        config: Config,
        llm: LLMRouter,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 20,
    ) -> None:
        self.config = config
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    def _should_offer_tools(self, session: Session) -> bool:
        schemas = self.registry.schemas()
        if not schemas:
            return False

        for message in reversed(session.messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return should_offer_tools_for_text(content)
            break
        return True

    async def _force_final_answer(
        self,
        session: Session,
        reason: str,
    ) -> LLMResponse | None:
        """Ask the model for a plain final answer with tools disabled."""
        messages = list(session.messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_FORCED_FINALIZATION_PROMPT}\n\n"
                    f"Guardrail reason: {reason}"
                ),
            }
        )
        try:
            response = await self.llm.chat(
                messages=messages,  # type: ignore[arg-type]
                tools=None,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("forced final answer failed: %s: %s", type(e).__name__, e)
            return None
        return normalize_response_for_model_quirks(response)

    async def step(self, session: Session) -> AgentStep:
        """One LLM call + dispatch of any tool calls it makes."""
        # Ensure the system prompt is at the top.
        if not session.messages or session.messages[0].get("role") != "system":
            session.add_system(self.system_prompt)

        response = await self.llm.chat(
            messages=session.messages,  # type: ignore[arg-type]
            tools=self.registry.schemas() if self._should_offer_tools(session) else None,
        )
        response = normalize_response_for_model_quirks(response)

        # Append assistant message (content + tool_calls)
        session.add_assistant(
            content=response.content,
            tool_calls=response.tool_calls or None,
        )

        # Dispatch any tool calls
        tool_results: list[tuple[str, ToolResult]] = []
        for tc in response.tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as e:
                log.warning("bad json args for %s: %s", name, e)
                args = {}

            log.info("tool_call: %s(%s)", name, list(args.keys()))
            result = await self.registry.dispatch(name, args, session)
            tool_results.append((tc["id"], result))
            session.add_tool_result(tc["id"], result.to_llm_content())

        return AgentStep(response=response, tool_results=tool_results)

    async def run(
        self,
        session: Session,
        on_step: StepCallback | None = None,
    ) -> AgentRun:
        """Loop step() until the LLM stops calling tools or we hit max_iterations.

        The loop has three guardrails layered on top of the base max_iterations
        cap, designed specifically for small local models (Qwen 7B / 30B) that
        tend to thrash on tool-calling:

        1. Identical-repeat detector: if the exact same tool call + result
           signature happens N=_MAX_IDENTICAL_STEP_REPEATS times in a row,
           abort the tool loop and ask the model for a plain final answer.
        2. No-progress window: look at the last _NO_PROGRESS_WINDOW tool-using
           steps — if they contain fewer than _MIN_UNIQUE_SIGNATURES_IN_WINDOW
           distinct signatures, the model is cycling through 1-2 actions over
           and over without real progress. Same forced-final exit.
        3. Soft limit: when there are <= _SOFT_LIMIT_TAIL iterations remaining
           and the model still wants tools, stop burning budget on half-useful
           calls — ask for a plain final answer instead of hitting the raw
           max_iterations wall with no text.

        Args:
            session: conversation state, mutated in place.
            on_step: optional async callback invoked after each AgentStep with
                (step, iteration_index). Used by the API server to stream
                progress events to the UI without duplicating loop logic.
        """
        steps: list[AgentStep] = []
        total_tokens = 0
        hit_limit = False
        forced_final = False
        repeated_step_count = 0
        last_signature: tuple[Any, ...] | None = None
        signature_window: list[tuple[Any, ...]] = []

        async def _emit_forced_final(reason: str, iteration: int) -> None:
            nonlocal total_tokens, forced_final
            final_response = await self._force_final_answer(session, reason)
            if final_response is None:
                return
            final_step = AgentStep(response=final_response, tool_results=[])
            steps.append(final_step)
            total_tokens += final_response.usage.get("total_tokens", 0)
            forced_final = True
            if on_step is not None:
                await on_step(final_step, iteration)

        for i in range(self.max_iterations):
            step = await self.step(session)
            steps.append(step)
            total_tokens += step.response.usage.get("total_tokens", 0)

            signature = step_signature(step)
            if signature == last_signature:
                repeated_step_count += 1
            else:
                repeated_step_count = 1
                last_signature = signature

            # Only window-track steps that actually used tools; a plain text
            # step is a natural end state, not thrashing.
            if step.response.tool_calls:
                signature_window.append(signature)
                if len(signature_window) > _NO_PROGRESS_WINDOW:
                    signature_window.pop(0)

            if on_step is not None:
                await on_step(step, i)

            if not step.response.tool_calls:
                # LLM produced a plain answer — we're done.
                log.debug("agent finished after %d iterations", i + 1)
                break

            if repeated_step_count >= _MAX_IDENTICAL_STEP_REPEATS:
                reason = (
                    "The last tool call pattern repeated without progress. "
                    "Stop using tools and summarize the verified result."
                )
                log.warning(
                    "agent guardrail: %d identical repeated steps, forcing final",
                    repeated_step_count,
                )
                await _emit_forced_final(reason, len(steps) - 1)
                break

            if (
                len(signature_window) >= _NO_PROGRESS_WINDOW
                and len(set(signature_window)) < _MIN_UNIQUE_SIGNATURES_IN_WINDOW
            ):
                reason = (
                    f"Over the last {_NO_PROGRESS_WINDOW} tool steps only "
                    f"{len(set(signature_window))} distinct actions were used "
                    "and no forward progress was made. Stop calling tools and "
                    "summarize what has been verified so far."
                )
                log.warning(
                    "agent guardrail: no-progress window, %d unique of %d steps",
                    len(set(signature_window)),
                    _NO_PROGRESS_WINDOW,
                )
                await _emit_forced_final(reason, len(steps) - 1)
                break

            remaining = self.max_iterations - (i + 1)
            if remaining <= _SOFT_LIMIT_TAIL:
                reason = (
                    "The iteration budget is almost exhausted. Stop calling "
                    "tools and give the user the best final answer from the "
                    "verified tool results already collected."
                )
                log.warning(
                    "agent guardrail: soft limit at step %d/%d, forcing final",
                    i + 1,
                    self.max_iterations,
                )
                await _emit_forced_final(reason, len(steps) - 1)
                break
        else:
            log.warning("agent hit max_iterations=%d", self.max_iterations)
            hit_limit = True

        final_text = steps[-1].response.content if steps else ""
        return AgentRun(
            final_text=final_text,
            steps=steps,
            hit_limit=hit_limit,
            total_tokens=total_tokens,
            forced_final=forced_final,
        )

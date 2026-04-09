from __future__ import annotations

import asyncio
import json

from codesmith.agent import (
    Agent,
    AgentStep,
    normalize_response_for_model_quirks,
    should_offer_tools_for_text,
    step_signature,
)
from codesmith.config import Config, LLMConfig
from codesmith.llm import LLMResponse
from codesmith.session import Session
from codesmith.tools.base import ToolResult
from codesmith.tools.registry import ToolRegistry


def test_small_talk_messages_skip_tools() -> None:
    assert should_offer_tools_for_text("привет") is False
    assert should_offer_tools_for_text("А как ты работаешь?") is False
    assert should_offer_tools_for_text("Спасибо!") is False
    assert should_offer_tools_for_text("Сколько в тебе токенов?") is False


def test_real_tasks_keep_tools_enabled() -> None:
    assert should_offer_tools_for_text("Исправь баг в API") is True
    assert should_offer_tools_for_text("Create a Python script that sorts CSV rows") is True
    assert should_offer_tools_for_text(r"Прочитай файл src/app.py и почини тесты") is True


def test_normalize_synthetic_answer_tool_call_into_plain_text() -> None:
    response = LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "answer",
                    "arguments": '{"text":"Я отвечаю обычным текстом."}',
                },
            }
        ],
        finish_reason="tool_calls",
        usage={"total_tokens": 12},
        model="ollama/qwen2.5-coder:7b",
        raw=None,
    )

    normalized = normalize_response_for_model_quirks(response)

    assert normalized.content == "Я отвечаю обычным текстом."
    assert normalized.tool_calls == []
    assert normalized.finish_reason == "stop"


def test_normalize_embedded_answer_json_in_content() -> None:
    response = LLMResponse(
        content=(
            '{"id":"call-1","type":"function","function":{"name":"answer",'
            '"arguments":"{\\"text\\":\\"Нормальный финальный ответ.\\"}"}}'
        ),
        tool_calls=[],
        finish_reason="stop",
        usage={"total_tokens": 12},
        model="ollama/qwen2.5-coder:7b",
        raw=None,
    )

    normalized = normalize_response_for_model_quirks(response)

    assert normalized.content == "Нормальный финальный ответ."
    assert normalized.tool_calls == []


def test_step_signature_ignores_tool_call_ids() -> None:
    first = AgentStep(
        response=LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                }
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 12},
            model="ollama/qwen2.5-coder:7b",
            raw=None,
        ),
        tool_results=[("call-1", ToolResult(ok=True, content="1", metadata={"exit_code": 0}))],
    )
    second = AgentStep(
        response=LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                }
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 12},
            model="ollama/qwen2.5-coder:7b",
            raw=None,
        ),
        tool_results=[("call-2", ToolResult(ok=True, content="1", metadata={"exit_code": 0}))],
    )

    assert step_signature(first) == step_signature(second)


def test_agent_forces_final_answer_after_identical_repeated_steps(monkeypatch) -> None:
    cfg = Config(llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"))
    agent = Agent(config=cfg, llm=None, registry=ToolRegistry())  # type: ignore[arg-type]
    session = Session.create(cfg.tools.filesystem.workspace_root)
    session.add_user("Create hello.py and run it.")

    repeated_step = AgentStep(
        response=LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
                }
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 20},
            model="ollama/qwen2.5-coder:7b",
            raw=None,
        ),
        tool_results=[("call-1", ToolResult(ok=True, content="1", metadata={"exit_code": 0}))],
    )
    forced = LLMResponse(
        content="I am stopping here because the same verified step kept repeating.",
        tool_calls=[],
        finish_reason="stop",
        usage={"total_tokens": 11},
        model="ollama/qwen2.5-coder:7b",
        raw=None,
    )

    async def fake_step(_session):
        return repeated_step

    async def fake_force(_session, reason):
        assert "repeated" in reason
        return forced

    monkeypatch.setattr(agent, "step", fake_step)
    monkeypatch.setattr(agent, "_force_final_answer", fake_force)

    run = asyncio.run(agent.run(session))

    assert run.final_text == forced.content
    assert run.hit_limit is False
    assert run.forced_final is True
    assert len(run.steps) == 4


def _make_tool_step(*, call_id: str, code: str) -> AgentStep:
    return AgentStep(
        response=LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "execute_python",
                        "arguments": json.dumps({"code": code}),
                    },
                }
            ],
            finish_reason="tool_calls",
            usage={"total_tokens": 10},
            model="ollama/qwen2.5-coder:7b",
            raw=None,
        ),
        tool_results=[
            (call_id, ToolResult(ok=True, content=code, metadata={"exit_code": 0}))
        ],
    )


def test_agent_forces_final_on_no_progress_window(monkeypatch) -> None:
    """Two alternating tool calls for many rounds must trigger the window guardrail."""
    cfg = Config(llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"))
    agent = Agent(config=cfg, llm=None, registry=ToolRegistry())  # type: ignore[arg-type]
    session = Session.create(cfg.tools.filesystem.workspace_root)
    session.add_user("Run code please.")

    # Alternating signatures avoid the identical-repeat guardrail but still
    # form a no-progress cycle (only 2 unique signatures over 6 steps < 3).
    alt_a = _make_tool_step(call_id="a", code="print(1)")
    alt_b = _make_tool_step(call_id="b", code="print(2)")
    sequence = [alt_a, alt_b, alt_a, alt_b, alt_a, alt_b, alt_a, alt_b]
    forced = LLMResponse(
        content="Stopping because the same pair of actions kept rotating.",
        tool_calls=[],
        finish_reason="stop",
        usage={"total_tokens": 9},
        model="ollama/qwen2.5-coder:7b",
        raw=None,
    )

    call_idx = {"i": 0}

    async def fake_step(_session):
        i = call_idx["i"]
        call_idx["i"] += 1
        return sequence[i]

    forced_reason: dict[str, str] = {}

    async def fake_force(_session, reason):
        forced_reason["text"] = reason
        return forced

    monkeypatch.setattr(agent, "step", fake_step)
    monkeypatch.setattr(agent, "_force_final_answer", fake_force)

    run = asyncio.run(agent.run(session))

    assert run.forced_final is True
    assert run.final_text == forced.content
    assert "distinct actions" in forced_reason["text"]
    # The loop should stop well before max_iterations=20 — identical-repeat
    # and no-progress-window are both possible trigger paths depending on
    # exact ordering, but either way we want a bounded step count.
    assert len(run.steps) <= 7


def test_agent_soft_limit_forces_final_near_max_iterations(monkeypatch) -> None:
    """If the model still wants tools at the tail of the budget, force a final."""
    cfg = Config(llm=LLMConfig(provider="ollama", model="qwen2.5-coder:7b"))
    agent = Agent(config=cfg, llm=None, registry=ToolRegistry())  # type: ignore[arg-type]
    agent.max_iterations = 4  # tight budget so the soft tail trips fast
    session = Session.create(cfg.tools.filesystem.workspace_root)
    session.add_user("Edit some files.")

    # Every step has a DIFFERENT signature so the identical-repeat guardrail
    # can't fire; we want the soft-limit trail to catch us.
    steps_seq = [
        _make_tool_step(call_id=f"c{i}", code=f"print({i})") for i in range(10)
    ]
    forced = LLMResponse(
        content="Soft limit reached — stopping here.",
        tool_calls=[],
        finish_reason="stop",
        usage={"total_tokens": 8},
        model="ollama/qwen2.5-coder:7b",
        raw=None,
    )

    call_idx = {"i": 0}

    async def fake_step(_session):
        i = call_idx["i"]
        call_idx["i"] += 1
        return steps_seq[i]

    seen_reasons: list[str] = []

    async def fake_force(_session, reason):
        seen_reasons.append(reason)
        return forced

    monkeypatch.setattr(agent, "step", fake_step)
    monkeypatch.setattr(agent, "_force_final_answer", fake_force)

    run = asyncio.run(agent.run(session))

    assert run.forced_final is True
    assert run.hit_limit is False
    assert run.final_text == forced.content
    assert any("iteration budget" in r for r in seen_reasons)
    # With max_iterations=4 and tail=2, the forced final must trigger on the
    # SECOND step, which means exactly 3 steps total (2 real + 1 forced).
    assert len(run.steps) == 3

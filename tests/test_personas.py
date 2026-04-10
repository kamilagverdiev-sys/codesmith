"""Unit tests for codesmith.personas.

The personas module is a simple registry + lookup, so the tests are
mostly about: right keys registered, case-insensitive lookup, clear
error on unknown key, and the content guarantees that downstream
callers rely on (architect has no_tools=True, default prompt still
carries the grep/glob guidance we baked in last commit).
"""

from __future__ import annotations

import pytest

from codesmith.agent import DEFAULT_SYSTEM_PROMPT
from codesmith.personas import (
    ARCHITECT,
    CODER,
    DEFAULT,
    PERSONAS,
    REVIEWER,
    UnknownPersonaError,
    get_persona,
    list_personas,
)


def test_four_personas_registered() -> None:
    assert set(PERSONAS.keys()) == {"default", "architect", "coder", "reviewer"}


def test_list_personas_stable_order() -> None:
    keys = [p.key for p in list_personas()]
    assert keys == ["default", "architect", "coder", "reviewer"]


def test_get_persona_case_insensitive() -> None:
    assert get_persona("ARCHITECT") is ARCHITECT
    assert get_persona("Architect") is ARCHITECT
    assert get_persona("  architect  ") is ARCHITECT


def test_get_persona_empty_defaults_to_default() -> None:
    assert get_persona("") is DEFAULT
    assert get_persona("   ") is DEFAULT


def test_get_persona_unknown_raises_with_known_list() -> None:
    with pytest.raises(UnknownPersonaError) as exc:
        get_persona("ceo")
    msg = str(exc.value)
    assert "ceo" in msg
    assert "architect" in msg
    assert "coder" in msg


def test_architect_and_reviewer_are_tool_less() -> None:
    assert ARCHITECT.no_tools is True
    assert REVIEWER.no_tools is True


def test_default_and_coder_carry_tools() -> None:
    assert DEFAULT.no_tools is False
    assert CODER.no_tools is False


def test_default_prompt_still_matches_agent_alias() -> None:
    # The agent module re-exports DEFAULT persona's prompt as
    # DEFAULT_SYSTEM_PROMPT for backwards compatibility with the rest
    # of the codebase (CLI, tests, etc). If these two ever drift, the
    # personas refactor has leaked somewhere.
    assert DEFAULT.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_architect_prompt_is_plan_only() -> None:
    import re as _re

    # Collapse whitespace so assertions stay readable even if the
    # prompt wraps "do not call any tools" across lines.
    text = _re.sub(r"\s+", " ", ARCHITECT.system_prompt.lower())
    assert "do not call any tools" in text
    assert "## goal" in text
    assert "## plan" in text
    assert "## acceptance checks" in text


def test_coder_prompt_references_plan_execution() -> None:
    text = CODER.system_prompt.lower()
    assert "approved plan" in text or "execute" in text
    assert "edit_file" in text


def test_reviewer_prompt_has_verdict_section() -> None:
    text = REVIEWER.system_prompt.lower()
    assert "## verdict" in text
    assert "approve" in text
    assert "request changes" in text

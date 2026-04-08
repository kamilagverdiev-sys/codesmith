from __future__ import annotations

from codesmith.config import LLMConfig, _fill_api_keys_from_env, _normalize_secret


def test_normalize_secret_rejects_placeholders() -> None:
    assert _normalize_secret("") is None
    assert _normalize_secret("   ") is None
    assert _normalize_secret("sk-ant-...") is None
    assert _normalize_secret("replace-me") is None
    assert _normalize_secret("your-api-key") is None


def test_normalize_secret_keeps_real_values() -> None:
    assert _normalize_secret("sk-real-key") == "sk-real-key"


def test_fill_api_keys_keeps_explicit_real_key() -> None:
    llm = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-live")
    result = _fill_api_keys_from_env(llm)
    assert result.api_key == "sk-live"


def test_fill_api_keys_drops_placeholder_from_config() -> None:
    llm = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-ant-...")
    result = _fill_api_keys_from_env(llm)
    assert result.api_key is None

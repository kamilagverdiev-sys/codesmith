from __future__ import annotations

from codesmith.config import Config, LLMConfig
from codesmith.model_selection import (
    default_model_profile_key,
    list_model_profiles,
    pull_target_for_selection,
    resolve_model_profile,
)


def _config() -> Config:
    return Config(
        llm=LLMConfig(
            provider="ollama",
            model="qwen2.5-coder:7b",
            api_base="http://127.0.0.1:11434",
        ),
        llm_fallbacks=[
            LLMConfig(provider="anthropic", model="claude-sonnet-4-6"),
        ],
        default_model_profile="local-auto",
    )


def test_default_model_profile_key_uses_config_value() -> None:
    cfg = _config()
    assert default_model_profile_key(cfg) == "local-auto"


def test_local_auto_prefers_qwen3_quality_when_installed() -> None:
    cfg = _config()
    resolved = resolve_model_profile(
        cfg,
        "local-auto",
        ["qwen3-coder:30b", "qwen2.5-coder:7b"],
    )

    assert resolved.primary.provider == "ollama"
    assert resolved.primary.model == "qwen3-coder:30b"
    assert resolved.key == "local-auto"


def test_local_auto_falls_back_to_fast_model_when_needed() -> None:
    cfg = _config()
    resolved = resolve_model_profile(cfg, "local-auto", ["qwen2.5-coder:7b"])

    assert resolved.primary.provider == "ollama"
    assert resolved.primary.model == "qwen2.5-coder:7b"


def test_direct_ollama_profile_uses_exact_tag() -> None:
    cfg = _config()
    resolved = resolve_model_profile(
        cfg,
        "ollama:my-custom-model:latest",
        ["my-custom-model:latest"],
    )

    assert resolved.primary.provider == "ollama"
    assert resolved.primary.model == "my-custom-model:latest"


def test_pull_target_uses_profile_primary_model() -> None:
    cfg = _config()
    assert pull_target_for_selection(cfg, "local-quality", ["qwen2.5-coder:7b"]) == "qwen3-coder:30b"


def test_list_model_profiles_includes_dynamic_installed_models() -> None:
    cfg = _config()
    profiles = list_model_profiles(cfg, ["qwen2.5-coder:7b", "my-custom-model:latest"])
    keys = {item.key for item in profiles}

    assert "local-auto" in keys
    assert "local-quality" in keys
    assert "local-fast" in keys
    assert "config-default" in keys
    assert "ollama:my-custom-model:latest" in keys

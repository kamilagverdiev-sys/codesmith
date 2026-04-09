from __future__ import annotations

from pathlib import Path

from codesmith.config import (
    LLMConfig,
    _fill_api_keys_from_env,
    _normalize_secret,
    load_config,
)


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


def test_load_config_expands_tilde_paths(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
llm:
  provider: ollama
  model: qwen2.5-coder:7b
memory:
  palace_path: "~/.codesmith/test-palace"
tools:
  filesystem:
    workspace_root: "~/.codesmith/test-workspaces"
sessions:
  path: "~/.codesmith/test-sessions.db"
logging:
  log_file: "~/.codesmith/test.log"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(cfg_path)

    assert config.memory.palace_path.is_absolute()
    assert config.tools.filesystem.workspace_root.is_absolute()
    assert config.sessions.path.is_absolute()
    assert config.logging.log_file is not None
    assert config.logging.log_file.is_absolute()

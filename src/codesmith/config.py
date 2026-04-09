"""Configuration loading and validation.

Loads config.yaml, overlays environment variables for secrets,
validates via Pydantic. Single source of truth for all settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Single LLM provider config (used for primary and fallbacks)."""

    provider: str  # e.g. "anthropic", "openai", "ollama"
    model: str  # LiteLLM model identifier
    api_key: str | None = None  # filled from env
    api_base: str | None = None
    temperature: float = 0.3
    max_tokens: int = 8000


class SandboxConfig(BaseModel):
    image: str = "codesmith-sandbox:latest"
    timeout_seconds: int = 30
    memory_mb: int = 512
    cpus: float = 1.0
    network: Literal["none", "bridge"] = "none"
    pids_limit: int = 64
    workspace_mount: str = "/work"


class SelfRepairConfig(BaseModel):
    enabled: bool = True
    max_attempts: int = 5
    include_previous_errors: bool = True


class CompactionConfig(BaseModel):
    enabled: bool = True
    trigger_at_messages: int = 50
    trigger_at_tokens: int = 20000


class LoopsConfig(BaseModel):
    self_repair: SelfRepairConfig = SelfRepairConfig()
    compaction: CompactionConfig = CompactionConfig()


class MemoryConfig(BaseModel):
    enabled: bool = False
    palace_path: Path = Path("~/.codesmith/palace").expanduser()
    mode: Literal["raw", "aaak", "rooms"] = "raw"
    search_top_k: int = 5


class FilesystemToolConfig(BaseModel):
    enabled: bool = True
    workspace_root: Path = Path("~/.codesmith/workspaces").expanduser()


class WebSearchToolConfig(BaseModel):
    enabled: bool = False
    provider: Literal["tavily", "brave"] = "tavily"
    max_calls_per_session: int = 5


class ToolsConfig(BaseModel):
    filesystem: FilesystemToolConfig = FilesystemToolConfig()
    web_search: WebSearchToolConfig = WebSearchToolConfig()
    memory: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})


class SessionsConfig(BaseModel):
    backend: Literal["sqlite", "memory"] = "sqlite"
    path: Path = Path("~/.codesmith/sessions.db").expanduser()


class APIAuthConfig(BaseModel):
    enabled: bool = False


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    auth: APIAuthConfig = APIAuthConfig()


class LimitsConfig(BaseModel):
    max_concurrent_sessions: int = 5
    max_concurrent_sandboxes: int = 3
    session_timeout_seconds: int = 600


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["rich", "json"] = "rich"
    log_llm_calls: bool = True
    log_file: Path | None = None


class Config(BaseModel):
    llm: LLMConfig
    llm_fallbacks: list[LLMConfig] = Field(default_factory=list)
    default_model_profile: str = "config-default"
    sandbox: SandboxConfig = SandboxConfig()
    loops: LoopsConfig = LoopsConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()
    sessions: SessionsConfig = SessionsConfig()
    api: APIConfig = APIConfig()
    limits: LimitsConfig = LimitsConfig()
    logging: LoggingConfig = LoggingConfig()


# ============================================================
# Loading
# ============================================================

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    # ollama: usually no key; api_base via OLLAMA_API_BASE
}


def _normalize_secret(value: str | None) -> str | None:
    """Treat empty or example placeholder values as missing secrets."""
    if value is None:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    placeholder_fragments = (
        "...",
        "your-",
        "replace-me",
        "changeme",
        "example",
        "paste-",
    )
    if any(fragment in lowered for fragment in placeholder_fragments):
        return None

    return cleaned


def _fill_api_keys_from_env(llm: LLMConfig) -> LLMConfig:
    """Pull API key from environment based on provider name.

    We DO NOT read keys from config.yaml to avoid committing them.
    """
    llm.api_key = _normalize_secret(llm.api_key)
    if llm.api_key:
        return llm  # user overrode explicitly, trust them

    env_key = _PROVIDER_ENV_KEYS.get(llm.provider.lower())
    if env_key:
        llm.api_key = _normalize_secret(os.getenv(env_key))

    # Ollama base URL from env
    if llm.provider.lower() == "ollama" and not llm.api_base:
        llm.api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    return llm


def _expand_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser()


def _normalize_paths(config: Config) -> Config:
    """Expand `~` in path fields loaded from YAML into host-specific paths."""
    config.memory.palace_path = config.memory.palace_path.expanduser()
    config.tools.filesystem.workspace_root = (
        config.tools.filesystem.workspace_root.expanduser()
    )
    config.sessions.path = config.sessions.path.expanduser()
    config.logging.log_file = _expand_optional_path(config.logging.log_file)
    return config


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config from YAML + .env.

    Args:
        path: path to config.yaml (defaults to cwd/config.yaml)

    Raises:
        FileNotFoundError: if config file missing
        pydantic.ValidationError: if config invalid
    """
    # 1. Load .env into process environment (does nothing if no .env file)
    load_dotenv()

    # 2. Parse YAML
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            f"Did you copy config.example.yaml to config.yaml?"
        )

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    # 3. Validate via Pydantic
    config = Config.model_validate(raw)
    config = _normalize_paths(config)

    # 4. Overlay API keys from environment
    config.llm = _fill_api_keys_from_env(config.llm)
    config.llm_fallbacks = [_fill_api_keys_from_env(f) for f in config.llm_fallbacks]

    return config

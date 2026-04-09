"""Model profile catalog and runtime resolution helpers.

Lets Codesmith expose a few opinionated model profiles without forcing
users to hand-edit config.yaml for every switch. Profiles can target
either the current config, a curated local Ollama setup, or any
installed Ollama model discovered at runtime.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from codesmith.config import Config, LLMConfig, _fill_api_keys_from_env

_DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"

_PROFILE_LOCAL_AUTO = "local-auto"
_PROFILE_LOCAL_FAST = "local-fast"
_PROFILE_LOCAL_QUALITY = "local-quality"
_PROFILE_CONFIG_DEFAULT = "config-default"

_OLLAMA_DIRECT_PREFIX = "ollama:"

_LOCAL_FAST_MODEL = "qwen2.5-coder:7b"
_LOCAL_QUALITY_MODEL = "qwen3-coder:30b"
_OLLAMA_TAG_CACHE_TTL_SECONDS = 5.0
_OLLAMA_TAG_CACHE: dict[str, tuple[float, list[str]]] = {}


class UnknownModelProfileError(ValueError):
    """Raised when a requested model profile key is unknown."""


@dataclass
class ResolvedModelProfile:
    key: str
    label: str
    description: str
    primary: LLMConfig
    fallbacks: list[LLMConfig]
    source: str
    recommended: bool = False
    availability: str = ""
    available: bool = True


@dataclass
class ModelProfileSummary:
    key: str
    label: str
    description: str
    primary_provider: str
    primary_model: str
    fallbacks: list[str]
    available: bool
    availability: str
    source: str
    recommended: bool = False


def _clone_llm(
    config: Config,
    *,
    provider: str,
    model: str,
    api_base: str | None = None,
) -> LLMConfig:
    llm = LLMConfig(
        provider=provider,
        model=model,
        api_base=api_base,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    return _fill_api_keys_from_env(llm)


def _config_fallbacks(config: Config) -> list[LLMConfig]:
    return [
        _fill_api_keys_from_env(
            LLMConfig.model_validate(fallback.model_dump())
        )
        for fallback in config.llm_fallbacks
    ]


def _dedupe_chain(items: list[LLMConfig]) -> list[LLMConfig]:
    seen: set[tuple[str, str]] = set()
    deduped: list[LLMConfig] = []
    for item in items:
        key = (item.provider.casefold(), item.model.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _ollama_base(config: Config) -> str:
    if config.llm.provider.lower() == "ollama" and config.llm.api_base:
        return config.llm.api_base
    for fallback in config.llm_fallbacks:
        if fallback.provider.lower() == "ollama" and fallback.api_base:
            return fallback.api_base
    return _DEFAULT_OLLAMA_BASE


def list_installed_ollama_models(config: Config) -> list[str]:
    """Return installed Ollama model tags, or [] if Ollama is unreachable."""
    base = _ollama_base(config).rstrip("/")
    cached = _OLLAMA_TAG_CACHE.get(base)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _OLLAMA_TAG_CACHE_TTL_SECONDS:
        return list(cached[1])
    req = urllib.request.Request(
        f"{base}/api/tags",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        _OLLAMA_TAG_CACHE[base] = (now, [])
        return []

    models = payload.get("models", [])
    tags = []
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name.strip():
            tags.append(name.strip())
    deduped = sorted(set(tags))
    _OLLAMA_TAG_CACHE[base] = (now, deduped)
    return deduped


def _provider_status(llm: LLMConfig, installed_ollama_models: set[str]) -> tuple[bool, str]:
    provider = llm.provider.lower()
    if provider == "ollama":
        if llm.model in installed_ollama_models:
            return True, f"{llm.model} installed"
        return False, f"{llm.model} not pulled"
    if provider in {"anthropic", "openai", "azure", "cohere", "mistral", "groq"}:
        if llm.api_key:
            return True, f"{provider} key set"
        return False, f"{provider} API key missing"
    return True, "available"


def _availability_for_chain(
    primary: LLMConfig,
    fallbacks: list[LLMConfig],
    installed_ollama_models: set[str],
) -> tuple[bool, str]:
    primary_ok, primary_msg = _provider_status(primary, installed_ollama_models)
    if primary_ok:
        return True, f"primary ready: {primary_msg}"

    for fallback in fallbacks:
        fallback_ok, fallback_msg = _provider_status(fallback, installed_ollama_models)
        if fallback_ok:
            return True, f"primary unavailable ({primary_msg}); fallback ready: {fallback_msg}"

    return False, f"no reachable provider in chain; primary {primary_msg}"


def _resolved_config_default(config: Config, installed_ollama_models: set[str]) -> ResolvedModelProfile:
    primary = _fill_api_keys_from_env(LLMConfig.model_validate(config.llm.model_dump()))
    fallbacks = _config_fallbacks(config)
    available, availability = _availability_for_chain(primary, fallbacks, installed_ollama_models)
    return ResolvedModelProfile(
        key=_PROFILE_CONFIG_DEFAULT,
        label="Config default",
        description="Use the primary provider and fallback chain from config.yaml.",
        primary=primary,
        fallbacks=fallbacks,
        source="config",
        available=available,
        availability=availability,
    )


def _resolved_local_fast(config: Config, installed_ollama_models: set[str]) -> ResolvedModelProfile:
    primary = _clone_llm(
        config,
        provider="ollama",
        model=_LOCAL_FAST_MODEL,
        api_base=_ollama_base(config),
    )
    fallbacks = _dedupe_chain(_config_fallbacks(config))
    available, availability = _availability_for_chain(primary, fallbacks, installed_ollama_models)
    return ResolvedModelProfile(
        key=_PROFILE_LOCAL_FAST,
        label="Local fast",
        description="Fast local coding profile tuned for responsiveness.",
        primary=primary,
        fallbacks=fallbacks,
        source="builtin",
        available=available,
        availability=availability,
    )


def _resolved_local_quality(config: Config, installed_ollama_models: set[str]) -> ResolvedModelProfile:
    primary = _clone_llm(
        config,
        provider="ollama",
        model=_LOCAL_QUALITY_MODEL,
        api_base=_ollama_base(config),
    )
    fallbacks = _dedupe_chain(
        [
            _clone_llm(
                config,
                provider="ollama",
                model=_LOCAL_FAST_MODEL,
                api_base=_ollama_base(config),
            ),
            *_config_fallbacks(config),
        ]
    )
    available, availability = _availability_for_chain(primary, fallbacks, installed_ollama_models)
    return ResolvedModelProfile(
        key=_PROFILE_LOCAL_QUALITY,
        label="Local quality",
        description="Prefer Qwen3-Coder 30B for stronger coding quality; fall back gracefully.",
        primary=primary,
        fallbacks=fallbacks,
        source="builtin",
        recommended=True,
        available=available,
        availability=availability,
    )


def _resolved_dynamic_ollama(
    config: Config,
    key: str,
    installed_ollama_models: set[str],
) -> ResolvedModelProfile:
    model = key[len(_OLLAMA_DIRECT_PREFIX):]
    if not model:
        raise UnknownModelProfileError("empty ollama model profile")
    primary = _clone_llm(
        config,
        provider="ollama",
        model=model,
        api_base=_ollama_base(config),
    )
    fallbacks = _dedupe_chain(_config_fallbacks(config))
    available, availability = _availability_for_chain(primary, fallbacks, installed_ollama_models)
    return ResolvedModelProfile(
        key=key,
        label=f"Installed - {model}",
        description="Use this exact installed Ollama model tag.",
        primary=primary,
        fallbacks=fallbacks,
        source="installed",
        available=available,
        availability=availability,
    )


def default_model_profile_key(config: Config) -> str:
    return config.default_model_profile or _PROFILE_CONFIG_DEFAULT


def resolve_model_profile(
    config: Config,
    selection: str | None = None,
    installed_models: list[str] | None = None,
) -> ResolvedModelProfile:
    installed = set(installed_models or list_installed_ollama_models(config))
    key = (selection or default_model_profile_key(config)).strip()
    if not key:
        key = _PROFILE_CONFIG_DEFAULT

    if key == _PROFILE_LOCAL_AUTO:
        auto_key = _PROFILE_LOCAL_QUALITY if _LOCAL_QUALITY_MODEL in installed else _PROFILE_LOCAL_FAST
        profile = resolve_model_profile(config, auto_key, sorted(installed))
        profile.key = _PROFILE_LOCAL_AUTO
        profile.label = "Local auto"
        profile.description = (
            "Auto-pick the best local coding profile available on this machine."
        )
        profile.source = "builtin"
        profile.recommended = True
        return profile

    if key == _PROFILE_CONFIG_DEFAULT:
        return _resolved_config_default(config, installed)
    if key == _PROFILE_LOCAL_FAST:
        return _resolved_local_fast(config, installed)
    if key == _PROFILE_LOCAL_QUALITY:
        return _resolved_local_quality(config, installed)
    if key.startswith(_OLLAMA_DIRECT_PREFIX):
        return _resolved_dynamic_ollama(config, key, installed)
    raise UnknownModelProfileError(f"unknown model profile: {key}")


def list_model_profiles(
    config: Config,
    installed_models: list[str] | None = None,
) -> list[ModelProfileSummary]:
    installed = sorted(set(installed_models or list_installed_ollama_models(config)))
    keys = [
        _PROFILE_LOCAL_AUTO,
        _PROFILE_LOCAL_QUALITY,
        _PROFILE_LOCAL_FAST,
        _PROFILE_CONFIG_DEFAULT,
        *[f"{_OLLAMA_DIRECT_PREFIX}{model}" for model in installed],
    ]
    summaries: list[ModelProfileSummary] = []
    for key in keys:
        profile = resolve_model_profile(config, key, installed)
        summaries.append(
            ModelProfileSummary(
                key=profile.key,
                label=profile.label,
                description=profile.description,
                primary_provider=profile.primary.provider,
                primary_model=profile.primary.model,
                fallbacks=[
                    f"{fallback.provider}/{fallback.model}"
                    for fallback in profile.fallbacks
                ],
                available=profile.available,
                availability=profile.availability,
                source=profile.source,
                recommended=profile.recommended,
            )
        )
    return summaries


def pull_target_for_selection(
    config: Config,
    selection: str | None,
    installed_models: list[str] | None = None,
) -> str:
    profile = resolve_model_profile(config, selection, installed_models)
    if profile.primary.provider.lower() != "ollama":
        raise UnknownModelProfileError(
            f"profile {profile.key} does not target a local Ollama model"
        )
    return profile.primary.model

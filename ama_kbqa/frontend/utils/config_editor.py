"""Read/write config.toml with backup support for the Settings page."""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

import toml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.toml"

# Providers whose /models endpoint we offer as a dropdown in the Settings page.
# (OpenRouter exposes hundreds of models — a free-text field is friendlier there.)
FETCHABLE_PROVIDERS = ("kit",)

# Env var holding each provider's API key (mirrors config._get_api_key).
_PROVIDER_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "kit": "KIT_API_KEY",
}


def load_config_raw() -> dict:
    """Load config.toml directly (bypasses the cached loader)."""
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def save_config(config: dict, backup: bool = True) -> None:
    """Write config dict to config.toml.

    Args:
        config: Full configuration dictionary to write.
        backup: If True, create a .bak copy before overwriting.
    """
    if backup and CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, CONFIG_PATH.with_suffix(".toml.bak"))

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        toml.dump(config, f)

    # Invalidate the cached config so the app picks up the new values
    _invalidate_config_cache()


def apply_to_session(config: dict) -> None:
    """Apply config dict to the in-memory cache without writing to disk.

    This modifies the running process's config but does not persist.
    """
    import ama_kbqa.config as cfg_module
    cfg_module._config_cache = config


def _invalidate_config_cache() -> None:
    """Reset the config module's cache so the next load reads from disk."""
    try:
        import ama_kbqa.config as cfg_module
        cfg_module._config_cache = None
    except Exception:
        pass


def _provider_api_key(provider: str, provider_config: dict) -> str:
    """Resolve a provider's API key (config value → env var). Mirrors
    ``config._get_api_key`` but never raises for the local provider."""
    if "api_key" in provider_config:
        return str(provider_config["api_key"])
    if provider == "llamacpp":
        return "sk-no-key-required"
    env_var = _PROVIDER_API_KEY_ENV.get(provider)
    key = os.getenv(env_var) if env_var else None
    if not key:
        raise RuntimeError(
            f"API key not set for provider '{provider}' "
            f"(expected env var {env_var or '—'})."
        )
    return key


def _fetch_provider_models_raw(provider: str, *, timeout: float) -> list[dict]:
    """Issue ``GET {base_url}/models`` and return the raw ``data`` entries.

    Shared by ``fetch_provider_models`` and ``fetch_provider_models_meta``.
    Raises ``RuntimeError`` (with a human-readable message) on any
    misconfiguration or transport/HTTP error so the caller can fall back to a
    free-text field / static list.
    """
    cfg = load_config_raw()
    provider_config = cfg.get(provider, {})
    base_url = provider_config.get("base_url")
    if not base_url:
        raise RuntimeError(f"No base_url configured for provider '{provider}'.")
    api_key = _provider_api_key(provider, provider_config)

    import httpx

    url = f"{base_url.rstrip('/')}/models"
    try:
        resp = httpx.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"{provider} /models returned HTTP {e.response.status_code}."
        ) from e
    except (httpx.HTTPError, ValueError) as e:
        raise RuntimeError(f"Could not reach {provider} /models: {e}") from e

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /models response from '{provider}'.")
    return [m for m in data if isinstance(m, dict) and m.get("id")]


def fetch_provider_models(provider: str, *, timeout: float = 10.0) -> list[str]:
    """Return the sorted model IDs advertised by an OpenAI-compatible provider.

    Issues ``GET {base_url}/models`` with the provider's API key and parses the
    standard ``{"data": [{"id": ...}]}`` shape. Raises ``RuntimeError`` (with a
    human-readable message) on any misconfiguration or transport/HTTP error so
    the caller can fall back to a free-text field.
    """
    entries = _fetch_provider_models_raw(provider, timeout=timeout)
    ids = sorted({m["id"] for m in entries})
    if not ids:
        raise RuntimeError(f"Provider '{provider}' returned no models.")
    return ids


def fetch_provider_models_meta(provider: str, *, timeout: float = 10.0) -> list[dict]:
    """Return the full ``/models`` catalog for ``provider``, normalized.

    Like ``fetch_provider_models`` but keeps the metadata a caller needs to
    tell a real chat model apart from an embedding/reranker/TTS/STT model, a
    routing alias, or a preset — not just the bare id. Each entry:

    - ``id``, ``name`` (display name; falls back to ``id`` if the endpoint
      omits it)
    - ``connection_type`` (e.g. ``"local"`` vs ``"external"``)
    - ``preset`` (bool)
    - ``tags`` (list of tag name strings, e.g. ``["Alias"]``)
    - ``hidden`` (bool, from Open WebUI-style ``info.meta.hidden``)
    - ``capabilities`` (the raw ``info.meta.capabilities`` dict, or ``None``)

    Raises ``RuntimeError`` under the same conditions as
    ``fetch_provider_models``.
    """
    raw = _fetch_provider_models_raw(provider, timeout=timeout)
    entries = []
    for m in raw:
        info = m.get("info") if isinstance(m.get("info"), dict) else {}
        meta = info.get("meta") if isinstance(info.get("meta"), dict) else {}
        tags = [
            t.get("name")
            for t in (m.get("tags") or [])
            if isinstance(t, dict) and t.get("name")
        ]
        entries.append({
            "id": m["id"],
            "name": m.get("name") or m["id"],
            "connection_type": m.get("connection_type"),
            "preset": bool(m.get("preset")),
            "tags": tags,
            "hidden": bool(meta.get("hidden")),
            "capabilities": meta.get("capabilities"),
        })
    if not entries:
        raise RuntimeError(f"Provider '{provider}' returned no models.")
    return entries

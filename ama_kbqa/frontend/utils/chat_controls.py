"""Sidebar model/temperature controls for the demo chat page.

Kept separate from ``1_Chat.py`` so the pure helpers are unit-testable without
executing the Streamlit page. The demo build pins the chat provider to KIT, so
the only knobs exposed are *which* KIT chat model to use and the temperature;
the endpoint is never user-selectable.

The model picker is populated by auto-discovery, not a hardcoded whitelist:
``available_models()`` queries the live KIT ``/models`` endpoint and keeps
whatever locally-hosted chat LLMs it currently advertises (see
``filter_selectable_models`` / ``_is_local_chat_model``). When KIT adds or
retires a model, the picker follows without a code change. A small static
fallback (``_OFFLINE_FALLBACK_MODELS``) covers the no-key / endpoint-down
case.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ama_kbqa.frontend.utils.config_editor import fetch_provider_models_meta
from ama_kbqa.pricing import format_cost_usd, get_model_pricing


# KIT's ``/models`` endpoint advertises far more than locally-hosted chat
# LLMs: image generation (``flux``), embeddings (``qwen3-embedding``),
# rerankers (``qwen3-reranker``), text-to-speech (``voxtral``/``tts``) and
# speech-to-text (``whisper``) models; Azure/Google-routed ("external")
# models; generic routing aliases (``alias.*``, tagged ``Alias``); and preset
# bundles (``preset: true``, e.g. ``standard-local``). None of those belong
# in the demo's chat-model picker.
#
# The primary signal for "is this a chat LLM" is metadata KIT's endpoint
# already exposes per entry (see ``config_editor.fetch_provider_models_meta``):
# ``connection_type == "local"``, not a preset, not tagged ``Alias``, not
# ``hidden``, and with non-null ``capabilities``. The id-keyword list below is
# a safety net for the (unlikely) case that metadata doesn't discriminate for
# some future model.
_NON_CHAT_ID_KEYWORDS: tuple[str, ...] = (
    "embedding",
    "rerank",
    "flux",
    "voxtral",
    "tts",
    "whisper",
    "image",
)

# Ordered preference for the model the demo pre-selects on load. The first of
# these that's actually available today wins; if none are available the
# picker falls back to the first model returned (never empty / mis-indexed).
# Ordered by responsiveness, not capability: a public demo must not default
# to a model that stalls. On 2026-09-14 Mistral Small 4 (also the base of
# KIT's own "Standard-Local" preset) answered in seconds, DeepSeek V4 Flash
# took anywhere from 1s to 47s, and GLM-5.3 did not answer within 120s.
DEFAULT_MODEL_PREFERENCE: tuple[str, ...] = (
    "kit.mistral-small-4-119b-a8b",
    "kit.deepseek-v4-flash",
    "kit.glm-5.3",
)

# Static fallback used only when the live KIT endpoint can't be reached (no
# API key, demo running offline, endpoint down). Deliberately NOT
# ``pricing.known_models()`` — that table is keyed by whatever KIT models
# OpenRouter pricing was last fetched for and still lists models KIT has
# since retired (see ``data/model_pricing.json``). Keep in sync with
# DEFAULT_MODEL_PREFERENCE when KIT's lineup changes.
_OFFLINE_FALLBACK_MODELS: tuple[str, ...] = (
    "kit.deepseek-v4-flash",
    "kit.glm-5.3",
    "kit.mistral-small-4-119b-a8b",
)

# Populated by ``available_models()`` from the endpoint's own ``name`` field
# (e.g. "DeepSeek V4 Flash"). ``display_model_name`` reads it so the picker
# can show a human-friendly label while the selectbox value stays the model
# id. Never cleared: a stale entry from a previous successful fetch is a
# better fallback than none while a later fetch fails.
_DISPLAY_NAMES: dict[str, str] = {}


def _is_local_chat_model(entry: dict) -> bool:
    """True if a KIT ``/models`` entry (see ``fetch_provider_models_meta``)
    is a selectable, locally-hosted chat LLM."""
    model_id = entry.get("id") or ""
    if entry.get("connection_type") != "local":
        return False
    if not model_id.startswith("kit."):
        return False
    if entry.get("preset"):
        return False
    if "Alias" in (entry.get("tags") or []):
        return False
    if entry.get("hidden"):
        return False
    if entry.get("capabilities") is None:
        return False
    lowered = model_id.lower()
    if any(kw in lowered for kw in _NON_CHAT_ID_KEYWORDS):
        return False
    return True


def default_model(models: list[str]) -> str:
    """The model the picker should select by default.

    The first entry of ``DEFAULT_MODEL_PREFERENCE`` that's in ``models``,
    else the first available model (so the picker is never empty /
    mis-indexed), else — only when ``models`` is itself empty — the first
    preference as a last-resort placeholder.
    """
    available = set(models)
    for candidate in DEFAULT_MODEL_PREFERENCE:
        if candidate in available:
            return candidate
    if models:
        return models[0]
    return DEFAULT_MODEL_PREFERENCE[0]


def filter_selectable_models(entries: list[dict]) -> list[str]:
    """Local KIT chat model ids from a raw ``/models`` catalog.

    ``entries`` is the shape returned by
    ``config_editor.fetch_provider_models_meta`` — one dict per model with
    ``id``, ``connection_type``, ``preset``, ``tags``, ``hidden`` and
    ``capabilities``. Keeps only locally-hosted (``connection_type ==
    "local"``) chat LLMs; drops presets, routing aliases, and non-chat models
    (image/embedding/reranker/TTS/STT). See ``_is_local_chat_model``.
    """
    return sorted({e["id"] for e in entries if _is_local_chat_model(e)})


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_kit_models_meta() -> list[dict]:
    """Cached wrapper around the live KIT ``/models`` fetch (5-minute TTL)."""
    return fetch_provider_models_meta("kit")


def available_models() -> list[str]:
    """Selectable KIT chat models.

    Uses the live KIT endpoint when reachable; otherwise falls back to a
    small static list of the models known to be locally hosted, so the demo
    dropdown still works without a KIT key (e.g. local development).
    """
    try:
        entries = _fetch_kit_models_meta()
    except Exception:  # noqa: BLE001 — any fetch failure → offline fallback
        return list(_OFFLINE_FALLBACK_MODELS)
    ids = filter_selectable_models(entries)
    _DISPLAY_NAMES.update({e["id"]: e["name"] for e in entries if e.get("name")})
    return ids or list(_OFFLINE_FALLBACK_MODELS)


def display_model_name(model: str) -> str:
    """Name shown in the model dropdown.

    Prefers the endpoint's own display name (e.g. "DeepSeek V4 Flash", as
    last seen via ``available_models()``); falls back to the id with the
    ``kit.`` provider prefix stripped when no endpoint name is known (offline
    fallback, or an id from before the first successful fetch).
    """
    name = _DISPLAY_NAMES.get(model)
    if name:
        return name
    return model[len("kit."):] if model.startswith("kit.") else model


def price_caption(model: Optional[str]) -> Optional[str]:
    """One-line per-million-token price for ``model`` (illustrative), or None."""
    entry = get_model_pricing(model)
    if not entry:
        return None
    per_in = format_cost_usd(entry["prompt_usd_per_token"] * 1_000_000)
    per_out = format_cost_usd(entry["completion_usd_per_token"] * 1_000_000)
    # Escape the ``$`` so Streamlit markdown does not treat the two dollar
    # signs as LaTeX math delimiters (which italicises the text between them).
    caption = f"{per_in} per 1M in and {per_out} per 1M out"
    return caption.replace("$", r"\$")


def apply_chat_settings(model: str, temperature: float) -> None:
    """Point the in-memory config — and this process's environment — at the
    chosen KIT model and temperature.

    Mutates the cached config dict so agents created or queried next *in this
    process* pick the values up via ``get_chat_model_name()`` /
    ``get_chat_temperature()``. It never writes ``config.toml`` to disk.

    It also sets ``AMA_KBQA_CHAT_MODEL`` / ``AMA_KBQA_CHAT_TEMPERATURE`` in
    ``os.environ``. The orchestrator and specialist MCP tool servers run as
    subprocesses spawned with ``env=os.environ.copy()`` (see
    ``orchestrator_agent/agent.py``, ``framework/mcp_client.py``) and load
    their *own* config.toml independently — mutating this process's
    in-memory config alone never reaches them, so without the env vars they'd
    silently keep using whatever chat model config.toml has (which today is
    a removed KIT model). ``config.get_chat_model_name()`` /
    ``get_chat_temperature()`` check these env vars first and fall back to
    config.toml when unset, so default (non-demo) behaviour is unchanged.

    Like the ``_config_cache`` mutation above, this is process-global: a
    server process handling multiple concurrent demo sessions would apply
    the override to all of them, not just the session that picked it. Same
    caveat as today, just extended to the env var.
    """
    import os

    import ama_kbqa.config as cfg_module

    cfg = cfg_module.load_config()
    cfg.setdefault("llm", {})["chat_provider"] = "kit"
    cfg.setdefault("kit", {})["chat_model"] = model
    cfg["llm"]["chat_temperature"] = float(temperature)
    cfg_module._config_cache = cfg

    os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] = model
    os.environ[cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR] = str(float(temperature))

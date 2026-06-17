"""Sidebar model/temperature controls for the demo chat page.

Kept separate from ``1_Chat.py`` so the pure helpers are unit-testable without
executing the Streamlit page. The demo build pins the chat provider to KIT, so
the only knobs exposed are *which* KIT chat model to use and the temperature;
the endpoint is never user-selectable.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from ama_kbqa.frontend.utils.config_editor import fetch_provider_models
from ama_kbqa.pricing import format_cost_usd, get_model_pricing, known_models


# Explicit allow-list of the currently-hosted local KIT *LLM chat* models.
#
# KIT's ``/models`` endpoint advertises many non-chat models that must never
# appear in a chat-model picker: image generation (``flux``), embeddings
# (``qwen3-embedding``), rerankers (``qwen3-reranker``), text-to-speech
# (``voxtral``/``tts``) and speech-to-text (``whisper``). It also routes a set
# of Azure-hosted OpenAI models (``azure.*``) and exposes generic routing
# aliases (``standard-extern``/``standard-local``). None of those belong in the
# demo, so rather than chase an ever-growing blacklist we whitelist exactly the
# KIT LLMs we want to offer.
#
# Update this set when KIT changes what it hosts — query the live endpoint with
# ``fetch_provider_models("kit")`` (see ``available_models``) and add the new
# chat model id(s) here. Ids must match the endpoint exactly (case-sensitive).
KIT_LLM_WHITELIST: frozenset[str] = frozenset({
    "kit.gemma4-31b-it",
    "kit.gpt-oss-120b",
    "kit.minimax-m2.7-229b",
    "kit.mistral-small-4-119b-a8b",
    "kit.qwen3.5-397b-A17b",
})


def filter_selectable_models(models: list[str]) -> list[str]:
    """Keep only whitelisted KIT LLM chat models for the demo dropdown.

    The demo offers an explicit allow-list (``KIT_LLM_WHITELIST``) of the
    currently-hosted local KIT chat models. Everything else advertised by the
    KIT endpoint — image/embedding/reranker/TTS/STT models, Azure-routed
    models, and generic routing aliases — is dropped so only real chat models
    reach the picker.
    """
    return sorted(m for m in set(models) if m in KIT_LLM_WHITELIST)


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_kit_models() -> list[str]:
    """Cached wrapper around the live KIT ``/models`` fetch (5-minute TTL)."""
    return fetch_provider_models("kit")


def available_models() -> list[str]:
    """Selectable KIT chat models.

    Uses the live KIT endpoint when reachable; otherwise falls back to the
    models we have local pricing for, so the demo dropdown still works without
    a KIT key (e.g. local development).
    """
    try:
        raw = _fetch_kit_models()
    except Exception:  # noqa: BLE001 — any fetch failure → offline fallback
        raw = known_models()
    selectable = filter_selectable_models(raw)
    return selectable or filter_selectable_models(known_models())


def display_model_name(model: str) -> str:
    """Name shown in the model dropdown: drop the ``kit.`` provider prefix."""
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
    """Point the in-memory config at the chosen KIT model and temperature.

    Session-only: it mutates the cached config dict so agents created or queried
    next pick the values up via ``get_chat_model_name()`` /
    ``get_chat_temperature()``. It never writes ``config.toml`` to disk.
    """
    import ama_kbqa.config as cfg_module

    cfg = cfg_module.load_config()
    cfg.setdefault("llm", {})["chat_provider"] = "kit"
    cfg.setdefault("kit", {})["chat_model"] = model
    cfg["llm"]["chat_temperature"] = float(temperature)
    cfg_module._config_cache = cfg


def reranker_available() -> bool:
    """Whether the optional ``rerank`` extra (sentence-transformers) is installed.

    Used to disable the reranker toggle in builds without the extra, instead
    of letting the server fall back with a per-call warning.
    """
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def apply_retrieval_settings(
    hybrid_enabled: bool, fusion: str, reranker_enabled: bool
) -> None:
    """Apply the sidebar retrieval (RAG) settings for this session.

    Two channels, never written to disk:

    1. The in-memory config cache, so any retrieval code running in *this*
       process sees the new values (same pattern as ``apply_chat_settings``).
    2. ``AMA_RETRIEVAL_*`` environment variables, because the actual retrieval
       runs inside the MCP server subprocesses, which inherit the parent
       environment when spawned. Changes therefore only reach a server
       subprocess started *after* this call; the chat page drops the persisted
       agent on change so the next question launches a fresh server.
    """
    import os

    import ama_kbqa.config as cfg_module

    cfg = cfg_module.load_config()
    retrieval = cfg.setdefault("retrieval", {})
    retrieval["hybrid_enabled"] = bool(hybrid_enabled)
    retrieval["fusion"] = fusion
    retrieval["reranker_enabled"] = bool(reranker_enabled)
    cfg_module._config_cache = cfg

    prefix = cfg_module.RETRIEVAL_ENV_PREFIX
    os.environ[prefix + "HYBRID_ENABLED"] = "true" if hybrid_enabled else "false"
    os.environ[prefix + "FUSION"] = fusion
    os.environ[prefix + "RERANKER_ENABLED"] = "true" if reranker_enabled else "false"

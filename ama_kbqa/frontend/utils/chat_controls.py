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

import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st

from ama_kbqa.config import get_frontend_chat_models
from ama_kbqa.frontend.utils.config_editor import (
    fetch_provider_models_meta,
    fetch_provider_models_pricing,
)
from ama_kbqa.pricing import format_cost_usd, get_model_pricing, register_runtime_pricing


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


# Display label per provider, used by ``display_choice`` so the visitor can
# see at a glance whether the next question is free (KIT) or billed.
_PROVIDER_LABELS: dict[str, str] = {
    "kit": "KIT",
    "openrouter": "OpenRouter",
}

# Availability rule for the non-KIT endpoints: the provider's key env var is
# set. Mirrors ``config._get_api_key``'s map; KIT is absent on purpose because
# its entries come from the live catalog (with an offline fallback) rather
# than from config, exactly as before.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
}

# The "type your own model id" entry. It is not a model: it is a placeholder
# whose ``key`` the client sends back together with the id the user typed (see
# ``ama_kbqa/api/runs.py``). The sentinel lives in the model slot so the key
# has the same ``provider:model`` shape as every other entry and can never
# collide with a real OpenRouter id (those always contain a "/").
CUSTOM_MODEL_SENTINEL = "__custom__"
CUSTOM_PROVIDER = "openrouter"
CUSTOM_CHOICE_KEY = f"{CUSTOM_PROVIDER}:{CUSTOM_MODEL_SENTINEL}"
CUSTOM_CHOICE_NAME = "OpenRouter (custom)"


@dataclass(frozen=True)
class ChatModelChoice:
    """One selectable entry in the demo's model picker.

    Carries the provider next to the model id: the two together are what
    ``apply_chat_settings`` needs to point both this process and the MCP
    tool-server subprocesses at the right endpoint. Prices are per token (the
    unit both OpenRouter's catalog and ``pricing`` use), None when unknown.

    ``custom`` marks the free-text placeholder rather than a real model.
    """

    provider: str
    model: str
    name: str
    prompt_usd_per_token: Optional[float] = None
    completion_usd_per_token: Optional[float] = None
    default: bool = False
    custom: bool = False

    @property
    def key(self) -> str:
        """Stable selectbox value. The model id alone is not unique across
        providers (the same id can exist on KIT and on OpenRouter)."""
        return f"{self.provider}:{self.model}"


def custom_choice() -> ChatModelChoice:
    """The "OpenRouter (custom)" placeholder entry."""
    return ChatModelChoice(
        provider=CUSTOM_PROVIDER,
        model=CUSTOM_MODEL_SENTINEL,
        name=CUSTOM_CHOICE_NAME,
        custom=True,
    )


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_openrouter_catalog() -> dict[str, dict]:
    """Cached wrapper around the live OpenRouter ``/models`` fetch (5-min TTL).

    Same TTL as the KIT fetch: long enough that the demo does not hit the
    endpoint on every rerun, short enough that a price change shows up within
    the day.
    """
    return fetch_provider_models_pricing("openrouter")


def _per_token(per_million: Optional[float]) -> Optional[float]:
    """Config prices are quoted per 1M tokens; everything else works per token."""
    return None if per_million is None else per_million / 1_000_000


def available_choices() -> tuple[list[ChatModelChoice], list[str]]:
    """All selectable chat models, plus the notices explaining what is missing.

    KIT entries come first and are exactly what ``available_models()`` returns
    today (live catalog, offline fallback). Then the ``[[frontend.chat_models]]``
    entries in config order, minus:

    - every entry whose provider's API key is unset (one notice per provider,
      rather than one per configured model),
    - every OpenRouter entry the live catalog does not know (a retired or
      mistyped id would otherwise fail only once the visitor asks a question).

    When the OpenRouter catalog is unreachable the configured entries are kept
    unvalidated and unpriced: a flaky network must not empty the picker.

    The free-text "OpenRouter (custom)" entry is appended last, under the same
    key rule as the presets: no OpenRouter key, no entry.

    Prices resolved here are registered with ``pricing.register_runtime_pricing``
    so the per-answer estimate in the chat footer works for billed models.
    """
    choices: list[ChatModelChoice] = [
        ChatModelChoice(provider="kit", model=model, name=display_model_name(model))
        for model in available_models()
    ]
    notices: list[str] = []

    catalog: Optional[dict[str, dict]] = None
    catalog_unreachable = False

    def _openrouter_available() -> bool:
        env_var = _PROVIDER_KEY_ENV["openrouter"]
        if os.getenv(env_var):
            return True
        notice = f"OpenRouter models hidden: {env_var} not set"
        if notice not in notices:
            notices.append(notice)
        return False

    for entry in get_frontend_chat_models():
        provider = entry["provider"]
        if provider == "openrouter" and not _openrouter_available():
            continue

        name = entry["name"]
        prompt = _per_token(entry["prompt_usd_per_m"])
        completion = _per_token(entry["completion_usd_per_m"])

        if provider == "openrouter":
            if catalog is None and not catalog_unreachable:
                try:
                    catalog = _fetch_openrouter_catalog()
                # Any failure at all (no key, HTTP error, junk payload) means
                # "catalog unreachable": keep the config entries unvalidated.
                except Exception:  # noqa: BLE001
                    catalog_unreachable = True
            if catalog is not None:
                meta = catalog.get(entry["id"])
                if meta is None:
                    notices.append(
                        f"OpenRouter model {entry['id']} is not in the live "
                        "catalog and was hidden"
                    )
                    continue
                if not meta["supports_tools"]:
                    # Every agent in this demo answers by calling tools. A
                    # model that cannot tool-call would return an unusable
                    # answer on the first question, so hide it here rather
                    # than let the visitor find out mid-demo.
                    notices.append(
                        f"OpenRouter model {entry['id']} cannot call tools "
                        "and was hidden"
                    )
                    continue
                name = name or meta["name"]
                if prompt is None:
                    prompt = meta["prompt"]
                if completion is None:
                    completion = meta["completion"]

        choice = ChatModelChoice(
            provider=provider,
            model=entry["id"],
            name=name or entry["id"],
            prompt_usd_per_token=prompt,
            completion_usd_per_token=completion,
            default=entry["default"],
        )
        register_runtime_pricing(
            choice.model, choice.prompt_usd_per_token, choice.completion_usd_per_token
        )
        choices.append(choice)

    if _openrouter_available():
        choices.append(custom_choice())

    return choices, notices


def default_choice(choices: list[ChatModelChoice]) -> ChatModelChoice:
    """The entry the picker should pre-select.

    An explicit ``default = true`` in config wins (that is what the flag is
    for). Otherwise the KIT preference order applies, so the demo keeps opening
    on a fast, free model and only spends money when the visitor chooses to.
    Falls back to the first entry, and finally to a synthetic KIT placeholder
    so the caller never has to handle an empty list.

    The custom placeholder is never chosen as a fallback default: it carries no
    model id, so a build that opened on it could not answer anything.
    """
    for choice in choices:
        if choice.default and not choice.custom:
            return choice
    kit_by_model = {c.model: c for c in choices if c.provider == "kit"}
    for candidate in DEFAULT_MODEL_PREFERENCE:
        if candidate in kit_by_model:
            return kit_by_model[candidate]
    for choice in choices:
        if not choice.custom:
            return choice
    fallback = DEFAULT_MODEL_PREFERENCE[0]
    return ChatModelChoice(
        provider="kit", model=fallback, name=display_model_name(fallback)
    )


def display_choice(choice: ChatModelChoice) -> str:
    """Label shown in the dropdown, e.g. "DeepSeek V4 Pro · OpenRouter"."""
    if choice.custom:
        return choice.name
    label = _PROVIDER_LABELS.get(choice.provider, choice.provider)
    return f"{choice.name} · {label}"


def price_caption_for(choice: ChatModelChoice) -> Optional[str]:
    """One-line per-million-token price for ``choice``, or None.

    KIT keeps today's illustrative table (nobody is billed for it). Non-KIT
    entries use the price the picker actually resolved and say "(billed)", so
    the visitor is never surprised by a number that turns out to be real. The
    custom entry has no price until an id is typed, and guessing one would be
    worse than showing none.
    """
    if choice.custom:
        return None
    if choice.provider == "kit":
        return price_caption(choice.model)
    if choice.prompt_usd_per_token is None or choice.completion_usd_per_token is None:
        return None
    per_in = format_cost_usd(choice.prompt_usd_per_token * 1_000_000)
    per_out = format_cost_usd(choice.completion_usd_per_token * 1_000_000)
    # Escape the ``$`` so Streamlit markdown does not read the pair of dollar
    # signs as LaTeX math delimiters.
    caption = f"{per_in} per 1M in and {per_out} per 1M out (billed)"
    return caption.replace("$", r"\$")


def apply_chat_settings(model: str, temperature: float, provider: str = "kit") -> None:
    """Point the in-memory config — and this process's environment — at the
    chosen provider, model and temperature.

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

    The provider matters as much as the model: posting an OpenRouter model id
    to the KIT endpoint fails with "Model not found", and the failure would
    only surface inside a specialist mid-answer. ``AMA_KBQA_CHAT_PROVIDER`` is
    what carries the pick across the subprocess boundary — without it the
    parent process would talk to OpenRouter while every MCP tool server
    silently kept using KIT, which still produces an answer and so hides the
    bug completely.

    ``provider`` defaults to "kit" so the two-argument call sites (the
    Streamlit page, the smoke script's bare model id, older tests) keep
    pinning KIT exactly as before.

    Like the ``_config_cache`` mutation above, this is process-global: a
    server process handling multiple concurrent demo sessions would apply
    the override to all of them, not just the session that picked it. Same
    caveat as today, just extended to the env vars.
    """
    import ama_kbqa.config as cfg_module

    cfg = cfg_module.load_config()
    cfg.setdefault("llm", {})["chat_provider"] = provider
    cfg.setdefault(provider, {})["chat_model"] = model
    cfg["llm"]["chat_temperature"] = float(temperature)
    cfg_module._config_cache = cfg

    os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] = provider
    os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] = model
    os.environ[cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR] = str(float(temperature))

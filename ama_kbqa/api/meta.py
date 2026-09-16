"""``GET /api/meta``: everything the React app needs to draw its first screen.

Every value comes from the helpers the Streamlit chat page already uses
(``agent_factory``, ``chat_controls``, ``config``), so the two frontends
offer the same agents, suggestions, models and defaults.

The model list works with either shape of ``chat_controls``:

* **Provider-aware** (branches demo-booth, demo-llamacpp): the module has
  ``available_choices()``, which returns ``ChatModelChoice`` objects carrying
  a provider next to the model id, plus notices about hidden entries. Each
  choice becomes one model whose id is ``choice.key`` (``"provider:model"``).
* **KIT only** (demo-v2-int): the live KIT ``/models`` catalog, filtered by
  ``filter_selectable_models``, with the offline fallback. Ids are the bare
  model ids and the provider is always ``"kit"``.

The shape is detected by looking for ``available_choices`` on the module, so
the same file runs unchanged on all three branches. ``POST /api/runs``
validates its ``model`` against the same cached catalog (:func:`get_catalog`)
and gets the provider and bare model id back from it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ama_kbqa.config import get_chat_temperature, get_live_graph_enabled
from ama_kbqa.frontend.utils import chat_controls
from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    ORCHESTRATOR_MODES,
    is_orchestrator,
)
from ama_kbqa.frontend.utils.config_editor import fetch_provider_models_meta

_LOG = logging.getLogger(__name__)

TITLE = "AMA-KBQA Assistant"

# Same default picker entry as chat.py's DEFAULT_AGENT (a Streamlit page, so
# it cannot be imported here without executing it).
DEFAULT_AGENT = "Orchestrator (Router)"

# KIT-only path: the live KIT /models result is reused for 5 minutes, like
# chat_controls' st.cache_data wrapper. A failed fetch (no key, endpoint
# down) caches the offline fallback for a shorter time so /meta does not wait
# out the HTTP timeout on every call, but recovers soon after KIT comes back.
MODELS_TTL_SECONDS = 300.0
FALLBACK_TTL_SECONDS = 60.0
# Provider-aware path: available_choices() caches its own fetches (KIT and
# OpenRouter for 300 s, the llama.cpp server for 30 s because the operator
# may swap the model behind it), so rebuilding the list this often is cheap
# and keeps the llama.cpp refresh as short as the Streamlit page's.
CHOICES_TTL_SECONDS = 30.0

# Indirection so tests can move the clock without patching time.monotonic
# for the whole process (asyncio relies on it).
_clock = time.monotonic

_models_lock = threading.Lock()
_catalog_cache: Optional[tuple[float, "ModelCatalog"]] = None  # (expires_at, catalog)
_default_temperature: Optional[float] = None


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

# Streamlit markdown of the AGENT_SUGGESTIONS keys, e.g.
# ":blue[:material/movie:] Director of Inception".
_SUGGESTION_RE = re.compile(
    r"^:(?P<color>[a-z]+)\[:material/(?P<icon>[a-z0-9_]+):\]\s*(?P<label>.+?)\s*$"
)
_COLOR_WRAP_RE = re.compile(r":[a-z]+\[(.*?)\]")
_ICON_TOKEN_RE = re.compile(r":material/[a-z0-9_]+:")


def parse_suggestion_label(key: str) -> dict[str, Optional[str]]:
    """Split a Streamlit suggestion key into label, Material Symbols icon name
    and colour. Keys that do not follow the ``:color[:material/icon:] label``
    shape keep their text (markup stripped) with ``icon``/``color`` None."""
    match = _SUGGESTION_RE.match(key)
    if match:
        return {
            "label": match["label"],
            "icon": match["icon"],
            "color": match["color"],
        }
    stripped = _COLOR_WRAP_RE.sub(r"\1", key)
    stripped = _ICON_TOKEN_RE.sub("", stripped)
    return {"label": " ".join(stripped.split()), "icon": None, "color": None}


def suggestions_meta() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for agent, items in AGENT_SUGGESTIONS.items():
        out[agent] = [
            {**parse_suggestion_label(key), "question": question}
            for key, question in items.items()
        ]
    return out


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def agents_meta() -> list[dict[str, Any]]:
    """Picker entries in AGENT_INFO order. Follow-ups (multiturn) exist only
    for directly-selected specialists, exactly as on the Streamlit page."""
    agents = []
    for name, info in AGENT_INFO.items():
        orchestrator = is_orchestrator(name)
        agents.append({
            "name": name,
            "tagline": info.get("tagline", ""),
            "description": info.get("description", ""),
            "databases": info.get("databases", ""),
            "tools": info.get("tools", ""),
            "orchestrator": orchestrator,
            "federated": bool(ORCHESTRATOR_MODES.get(name, False)),
            "followups": not orchestrator,
        })
    return agents


def default_agent() -> str:
    if DEFAULT_AGENT in AGENT_INFO:
        return DEFAULT_AGENT
    return next(iter(AGENT_INFO))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelOption:
    """One entry of the model picker, resolved for the API."""

    key: str       # /meta "id"; what POST /api/runs sends back
    provider: str  # endpoint the run is pointed at ("kit", "openrouter", ...)
    model: str     # bare model id: what the endpoint and the pricing table know
    name: str
    price: Optional[str]

    def to_meta(self) -> dict[str, Any]:
        return {
            "id": self.key,
            "name": self.name,
            "price": self.price,
            "provider": self.provider,
        }


class ModelCatalog:
    """The resolved model list: options in picker order, the default key and
    the notices explaining what is hidden. Immutable once built."""

    def __init__(self, options: Iterable[ModelOption], default_key: str,
                 notices: Iterable[str] = ()) -> None:
        self.options: tuple[ModelOption, ...] = tuple(options)
        self.default_key = default_key
        self.notices: tuple[str, ...] = tuple(notices)
        self._by_key = {option.key: option for option in self.options}

    def resolve(self, key: str) -> Optional[ModelOption]:
        return self._by_key.get(key)

    def keys(self) -> list[str]:
        return [option.key for option in self.options]


def provider_aware() -> bool:
    """True when this branch's chat_controls has the provider-aware picker."""
    return callable(getattr(chat_controls, "available_choices", None))


def _plain_price(caption: Optional[str]) -> Optional[str]:
    # chat_controls escapes "$" for Streamlit markdown; JSON wants it plain.
    return caption.replace("\\$", "$") if caption else None


def _kit_option(model_id: str, endpoint_name: Optional[str] = None) -> ModelOption:
    if endpoint_name and endpoint_name != model_id:
        name = endpoint_name
    else:
        # Strips the "kit." prefix when the endpoint gave no display name.
        name = chat_controls.display_model_name(model_id)
    return ModelOption(
        key=model_id,
        provider="kit",
        model=model_id,
        name=name,
        price=_plain_price(chat_controls.price_caption(model_id)),
    )


def _kit_catalog() -> tuple[ModelCatalog, float]:
    try:
        entries = fetch_provider_models_meta("kit")
    except Exception:  # noqa: BLE001 - any failure means offline fallback
        entries = []
    ids = chat_controls.filter_selectable_models(entries) if entries else []
    if ids:
        names = {e["id"]: e.get("name") for e in entries}
        options = [_kit_option(m, names.get(m)) for m in ids]
        ttl = MODELS_TTL_SECONDS
    else:
        options = [_kit_option(m) for m in chat_controls._OFFLINE_FALLBACK_MODELS]
        ttl = FALLBACK_TTL_SECONDS
    default_key = chat_controls.default_model([o.key for o in options])
    return ModelCatalog(options, default_key), ttl


def _choices_catalog() -> tuple[ModelCatalog, float]:
    try:
        choices, notices = chat_controls.available_choices()
        choices, notices = list(choices), list(notices)
    except Exception as exc:  # noqa: BLE001 - never leave the picker empty
        _LOG.exception("available_choices() failed")
        choices = []
        notices = [
            f"Could not list the chat models ({type(exc).__name__}); "
            "only the default model is offered."
        ]
    default = chat_controls.default_choice(choices)
    # default_choice falls back to a synthetic entry when the list is empty;
    # offer it so the default key always resolves.
    if all(choice.key != default.key for choice in choices):
        choices.append(default)

    options: list[ModelOption] = []
    seen: set[str] = set()
    for choice in choices:
        if choice.key in seen:
            continue
        seen.add(choice.key)
        options.append(ModelOption(
            key=choice.key,
            provider=choice.provider,
            model=choice.model,
            name=chat_controls.display_choice(choice),
            price=_plain_price(chat_controls.price_caption_for(choice)),
        ))
    return ModelCatalog(options, default.key, notices), CHOICES_TTL_SECONDS


def get_catalog() -> ModelCatalog:
    """The current model catalog, cached in-process (blocking; call it from a
    worker thread, never on the event loop)."""
    global _catalog_cache
    with _models_lock:
        now = _clock()
        if _catalog_cache is not None and _catalog_cache[0] > now:
            return _catalog_cache[1]
        catalog, ttl = _choices_catalog() if provider_aware() else _kit_catalog()
        _catalog_cache = (now + ttl, catalog)
        return catalog


def get_models() -> list[dict[str, Any]]:
    return [option.to_meta() for option in get_catalog().options]


def get_model_ids() -> list[str]:
    return get_catalog().keys()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def default_temperature() -> float:
    """The configured chat temperature, read once per process.

    Captured at startup because ``apply_chat_settings`` (called for every run)
    writes the chosen temperature into ``AMA_KBQA_CHAT_TEMPERATURE``, which
    ``get_chat_temperature()`` honours: read later, the "default" would be
    whatever the last visitor picked.
    """
    global _default_temperature
    if _default_temperature is None:
        _default_temperature = float(get_chat_temperature())
    return _default_temperature


def demo_settings() -> dict[str, Any]:
    """Demo-mode limits, from the same env vars and defaults as chat.py."""
    return {
        "enabled": os.environ.get("DEMO_MODE", "0") == "1",
        "max_queries_per_session": int(os.environ.get("DEMO_MAX_QUERIES_PER_SESSION", "20")),
        "min_seconds_between_queries": float(
            os.environ.get("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "3")
        ),
    }


def build_meta() -> dict[str, Any]:
    catalog = get_catalog()
    return {
        "title": TITLE,
        "agents": agents_meta(),
        "default_agent": default_agent(),
        "suggestions": suggestions_meta(),
        "models": [option.to_meta() for option in catalog.options],
        "default_model": catalog.default_key,
        "model_notices": list(catalog.notices),
        "default_temperature": default_temperature(),
        "live_graph": bool(get_live_graph_enabled()),
        "demo": demo_settings(),
    }


def reset_caches() -> None:
    """Forget the cached model catalog and default temperature (tests)."""
    global _catalog_cache, _default_temperature
    with _models_lock:
        _catalog_cache = None
    _default_temperature = None

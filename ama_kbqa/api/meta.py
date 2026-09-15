"""``GET /api/meta``: everything the React app needs to draw its first screen.

Every value comes from the helpers the Streamlit chat page already uses
(``agent_factory``, ``chat_controls``, ``config``), so the two frontends
offer the same agents, suggestions, models and defaults.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Optional

from ama_kbqa.config import get_chat_temperature, get_live_graph_enabled
from ama_kbqa.frontend.utils.agent_factory import (
    AGENT_INFO,
    AGENT_SUGGESTIONS,
    ORCHESTRATOR_MODES,
    is_orchestrator,
)
from ama_kbqa.frontend.utils.chat_controls import (
    _OFFLINE_FALLBACK_MODELS,
    default_model,
    display_model_name,
    filter_selectable_models,
    price_caption,
)
from ama_kbqa.frontend.utils.config_editor import fetch_provider_models_meta

TITLE = "AMA-KBQA Assistant"

# Same default picker entry as chat.py's DEFAULT_AGENT (a Streamlit page, so
# it cannot be imported here without executing it).
DEFAULT_AGENT = "Orchestrator (Router)"

# Live KIT /models result is reused for 5 minutes, like chat_controls'
# st.cache_data wrapper. A failed fetch (no key, endpoint down) caches the
# offline fallback for a shorter time so /meta does not wait out the HTTP
# timeout on every call, but recovers soon after KIT comes back.
MODELS_TTL_SECONDS = 300.0
FALLBACK_TTL_SECONDS = 60.0

# Indirection so tests can move the clock without patching time.monotonic
# for the whole process (asyncio relies on it).
_clock = time.monotonic

_models_lock = threading.Lock()
_models_cache: Optional[tuple[float, list[dict]]] = None  # (expires_at, models)
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

def _model_entry(model_id: str, endpoint_name: Optional[str] = None) -> dict[str, Any]:
    if endpoint_name and endpoint_name != model_id:
        name = endpoint_name
    else:
        # Strips the "kit." prefix when the endpoint gave no display name.
        name = display_model_name(model_id)
    caption = price_caption(model_id)
    return {
        "id": model_id,
        "name": name,
        # price_caption escapes "$" for Streamlit markdown; JSON wants it plain.
        "price": caption.replace("\\$", "$") if caption else None,
    }


def _fetch_models() -> tuple[list[dict[str, Any]], float]:
    try:
        entries = fetch_provider_models_meta("kit")
    except Exception:  # noqa: BLE001 - any failure means offline fallback
        entries = []
    ids = filter_selectable_models(entries) if entries else []
    if not ids:
        return [_model_entry(m) for m in _OFFLINE_FALLBACK_MODELS], FALLBACK_TTL_SECONDS
    names = {e["id"]: e.get("name") for e in entries}
    return [_model_entry(m, names.get(m)) for m in ids], MODELS_TTL_SECONDS


def get_models() -> list[dict[str, Any]]:
    """Selectable KIT chat models, cached in-process (blocking; call it from
    a worker thread, never on the event loop)."""
    global _models_cache
    with _models_lock:
        now = _clock()
        if _models_cache is not None and _models_cache[0] > now:
            return [dict(m) for m in _models_cache[1]]
        models, ttl = _fetch_models()
        _models_cache = (now + ttl, models)
        return [dict(m) for m in models]


def get_model_ids() -> list[str]:
    return [m["id"] for m in get_models()]


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
    models = get_models()
    return {
        "title": TITLE,
        "agents": agents_meta(),
        "default_agent": default_agent(),
        "suggestions": suggestions_meta(),
        "models": models,
        "default_model": default_model([m["id"] for m in models]),
        "default_temperature": default_temperature(),
        "live_graph": bool(get_live_graph_enabled()),
        "demo": demo_settings(),
    }


def reset_caches() -> None:
    """Forget the cached model list and default temperature (tests)."""
    global _models_cache, _default_temperature
    with _models_lock:
        _models_cache = None
    _default_temperature = None

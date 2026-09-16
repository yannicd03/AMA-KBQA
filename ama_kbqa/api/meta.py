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

from ama_kbqa.config import (
    CONFIG_PATH,
    get_chat_temperature,
    get_federation_enabled,
    get_federation_max_specialists,
    get_frontend_chat_models,
    get_frontend_settings_level,
    get_hybrid_enabled,
    get_live_graph_enabled,
    get_reranker_enabled,
    load_config,
)
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
    # True for the "type your own id" placeholder, which carries no model of
    # its own: POST /api/runs must send a ``custom_model`` alongside its key.
    custom: bool = False

    def to_meta(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.key,
            "name": self.name,
            "price": self.price,
            "provider": self.provider,
        }
        # Emitted only when true: the key is a marker for the one placeholder
        # entry, not a field every model carries.
        if self.custom:
            payload["custom"] = True
        return payload


class ModelCatalog:
    """The resolved model list: options in picker order, the default key and
    the notices explaining what is hidden. Immutable once built."""

    def __init__(self, options: Iterable[ModelOption], default_key: str,
                 notices: Iterable[str] = (), live: Optional[bool] = None) -> None:
        self.options: tuple[ModelOption, ...] = tuple(options)
        self.default_key = default_key
        self.notices: tuple[str, ...] = tuple(notices)
        # True when the list came from a live catalog fetch, False when it is
        # the offline fallback, None when this path cannot tell (the
        # provider-aware picker fetches per provider). Read by
        # ``endpoint_rows`` to report the chat endpoint's reachability without
        # issuing a second request.
        self.live = live
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
        live = True
    else:
        options = [_kit_option(m) for m in chat_controls._OFFLINE_FALLBACK_MODELS]
        ttl = FALLBACK_TTL_SECONDS
        live = False
    default_key = chat_controls.default_model([o.key for o in options])
    return ModelCatalog(options, default_key, live=live), ttl


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
            # getattr: the older provider-aware branches have no such field.
            custom=bool(getattr(choice, "custom", False)),
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
# The custom (free-text) model entry
# ---------------------------------------------------------------------------
# The picker's "OpenRouter (custom)" entry has no model id of its own, so it
# cannot be resolved from the catalog like every other pick. The client sends
# its key *plus* the id the visitor typed, and the id is validated by shape
# here rather than against an allowlist — the whole point of the entry is to
# reach a model this build has never heard of.
#
# Deliberately NOT minted into the catalog: ``get_catalog`` is memoised with a
# TTL and shared by every session, so a typed id added to it would vanish on
# the next refresh and leak between visitors in the meantime.

CUSTOM_MODEL_MAX_LENGTH = 200

# OpenRouter ids look like "vendor/model" with optional ":variant" and dotted
# versions. Anything with whitespace, quotes or control characters is a typo
# or an injection attempt, not a model id.
_CUSTOM_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@/:+-]*$")

_CUSTOM_MODEL_HELP = (
    "Type an OpenRouter model id, for example deepseek/deepseek-v4-pro."
)


class InvalidCustomModel(ValueError):
    """The typed model id is empty or not shaped like a model id."""


def custom_model_option(option: ModelOption, typed: Optional[str]) -> ModelOption:
    """Resolve the custom placeholder plus a typed id into a real option.

    ``option`` is the placeholder the catalog resolved (``custom`` set);
    ``typed`` is the raw string from the client. Whitespace is trimmed, an
    empty value is rejected, and the result is a normal :class:`ModelOption`
    on the placeholder's provider, so everything downstream (the run record,
    the multiturn agent key, ``apply_chat_settings``) treats it like any other
    pick.

    Price stays None: nobody knows what an arbitrary id costs, and the chat
    footer showing no estimate is the honest outcome — a guessed number would
    be worse.

    Raises:
        InvalidCustomModel: empty, over-long, or wrongly shaped.
    """
    candidate = (typed or "").strip()
    if not candidate:
        raise InvalidCustomModel(_CUSTOM_MODEL_HELP)
    if len(candidate) > CUSTOM_MODEL_MAX_LENGTH:
        raise InvalidCustomModel(
            f"That model id is too long (max {CUSTOM_MODEL_MAX_LENGTH} characters)."
        )
    if not _CUSTOM_MODEL_RE.match(candidate):
        raise InvalidCustomModel(
            f"{candidate!r} is not a valid model id. {_CUSTOM_MODEL_HELP}"
        )
    return ModelOption(
        key=f"{option.provider}:{candidate}",
        provider=option.provider,
        model=candidate,
        name=candidate,
        price=None,
    )


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


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------
# /meta *describes* the settings panel instead of the React app hardcoding it,
# so the demo branches differ by CONFIG plus a few rows of data rather than by
# divergent frontend code. Everything branch-specific goes through two seams:
#
#   * ``endpoint_rows()``   - the chat/embedding endpoints this build talks to
#     (demo-booth: one row per provider; demo-llamacpp: the local chat and
#     embedding servers with a probed ``status``).
#   * ``diagnostic_rows()`` - a handful of read-only facts about the build.
#
# A branch extends exactly those two functions; nothing else here and nothing
# in the React panel needs to change. Rows are plain dicts with a fixed key
# set, and the panel renders whatever it is given: an unknown provider, role
# or status renders neutrally rather than breaking the layout.
#
# Two hard rules for anything added here:
#   1. never an API key, token or password. ``api_key_state()`` reports only
#      whether a key is configured, plus the env var's NAME as a hint;
#   2. never a host path. ``CONFIG_PATH.name``, not ``CONFIG_PATH``.

LEVEL_MINIMAL = "minimal"
LEVEL_FULL = "full"

# Per-row reachability. Unknown values are allowed (a branch may invent one);
# the client falls back to neutral styling for anything it does not know.
STATUS_OK = "ok"
STATUS_UNREACHABLE = "unreachable"
STATUS_UNKNOWN = "unknown"
STATUS_NO_KEY = "no_key"

ENDPOINTS_LABEL = "Endpoints"
DIAGNOSTICS_LABEL = "This build"

# Sentinel for provider_endpoint_row(key_env=...): derive the env var from the
# provider. Pass ``key_env=None`` for an endpoint that needs no key at all.
KEY_ENV_AUTO = "auto"

# Env var holding each provider's API key; mirrors ``config._get_api_key``.
# Providers absent here need no key (a local llama-server).
PROVIDER_KEY_ENV: dict[str, str] = {
    "kit": "KIT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Display names for the providers this code line knows about. Anything else
# shows under its config key, which is what a branch operator recognises.
PROVIDER_LABELS: dict[str, str] = {
    "kit": "KIT KI-Toolbox",
    "openrouter": "OpenRouter",
    "llamacpp": "llama.cpp (local)",
}


def provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider) or provider


def api_key_state(env_var: Optional[str]) -> dict[str, Any]:
    """Whether a provider's API key is configured, and where it comes from.

    Carries no part of the key itself: ``configured`` is a boolean and
    ``hint`` is the environment variable's *name*.
    """
    configured = bool((os.environ.get(env_var) or "").strip()) if env_var else False
    return {"configured": configured, "hint": env_var}


def endpoint_row(
    row_id: str,
    label: str,
    *,
    role: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    model_source: Optional[str] = None,
    api_key: Optional[dict[str, Any]] = None,
    status: str = STATUS_UNKNOWN,
    detail: Optional[str] = None,
) -> dict[str, Any]:
    """One row of the Endpoints section.

    Every key is always present (``None`` when unknown) so the client can
    render rows uniformly. ``api_key=None`` means "needs no key" and the
    client omits the line.
    """
    return {
        "id": row_id,
        "label": label,
        "role": role,
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "model_source": model_source,
        "api_key": api_key,
        "status": status,
        "detail": detail,
    }


def provider_endpoint_row(
    provider: str,
    *,
    role: str = "chat",
    row_id: Optional[str] = None,
    label: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    model_source: Optional[str] = None,
    status: Optional[str] = None,
    detail: Optional[str] = None,
    key_env: Optional[str] = KEY_ENV_AUTO,
) -> dict[str, Any]:
    """A row for a provider configured in config.toml.

    Reads ``[<provider>] base_url`` and the provider's API-key env var, so a
    branch adds a provider row in one call. ``status`` defaults to ``ok`` /
    ``no_key`` from the key alone; pass it explicitly (e.g.
    ``STATUS_UNREACHABLE`` after probing a local server).
    """
    section = load_config().get(provider, {})
    if key_env == KEY_ENV_AUTO:
        key_env = PROVIDER_KEY_ENV.get(provider)
    key = api_key_state(key_env) if key_env else None
    if status is None:
        if key is None:
            status = STATUS_UNKNOWN
        else:
            status = STATUS_OK if key["configured"] else STATUS_NO_KEY
    if detail is None and status == STATUS_NO_KEY and key and key["hint"]:
        detail = f"Set {key['hint']} in the environment to reach this endpoint."
    return endpoint_row(
        row_id or f"{provider}-{role}",
        label or provider_label(provider),
        role=role,
        provider=provider,
        base_url=section.get("base_url") if base_url is None else base_url,
        model=model,
        model_source=model_source,
        api_key=key,
        status=status,
        detail=detail,
    )


# Said of every endpoint that costs real money, in both key states, so nobody
# picks one of its models without knowing.
BILLED_DETAIL = "Billed to this deployment's account when picked in the picker above."


def _chat_model_entries() -> list[dict[str, Any]]:
    """The ``[[frontend.chat_models]]`` entries, or ``[]`` when unreadable.

    A hand-edited config must degrade the panel to "KIT only" rather than take
    /api/meta down, which is what the picker does with it too.
    """
    try:
        return list(get_frontend_chat_models())
    except Exception:  # noqa: BLE001 - a broken config degrades to KIT only
        _LOG.warning("Could not read [[frontend.chat_models]]", exc_info=True)
        return []


def optional_chat_providers() -> list[str]:
    """The non-KIT chat providers this build offers, in config order.

    Derived from ``[[frontend.chat_models]]`` rather than a hard-coded list, so
    the Endpoints section follows whatever this build actually ships: drop the
    OpenRouter entries from config and the OpenRouter row goes with them.
    """
    providers: list[str] = []
    for entry in _chat_model_entries():
        provider = entry.get("provider")
        if provider and provider != "kit" and provider not in providers:
            providers.append(provider)
    return providers


def endpoint_rows() -> list[dict[str, Any]]:
    """The endpoints this build talks to, newest-relevant first.

    **Branch seam.** demo-booth appends one ``provider_endpoint_row`` per
    offered provider; demo-llamacpp replaces/adds local-server rows with a
    probed ``status`` (``STATUS_UNREACHABLE`` when llama-server is down).
    Return ``[]`` to hide the section entirely.

    The KIT-only build shows what it actually uses: the configured chat
    provider (models come from its live catalog, picked in the sidebar) and
    the configured embedding provider (model fixed in config.toml).
    """
    config = load_config()
    llm = config.get("llm", {})
    rows: list[dict[str, Any]] = []

    chat_provider = llm.get("chat_provider")
    if chat_provider:
        key = api_key_state(PROVIDER_KEY_ENV.get(chat_provider))
        status: Optional[str] = None
        detail: Optional[str] = "Chat completions for every agent and sub-agent."
        if not key["configured"] and PROVIDER_KEY_ENV.get(chat_provider):
            status, detail = None, None  # derived below: no_key + its hint
        elif get_catalog().live is False:
            status = STATUS_UNREACHABLE
            detail = ("The model catalog could not be fetched; the picker "
                      "shows the offline fallback list.")
        rows.append(provider_endpoint_row(
            chat_provider,
            role="chat",
            model_source="Live /models catalog, chosen in the picker above",
            status=status,
            detail=detail,
        ))

    embedding_provider = llm.get("embedding_provider")
    if embedding_provider:
        rows.append(provider_endpoint_row(
            embedding_provider,
            role="embedding",
            model=config.get(embedding_provider, {}).get("embedding_model"),
            model_source="config.toml",
            detail="Embeds your question for vector search over the graphs.",
        ))

    # One row per *provider* the picker offers next to KIT, not one per model.
    # Reachability is read off the API key alone: probing OpenRouter would put
    # a network round-trip in front of the demo's first screen on every
    # /api/meta call. A provider without its key stays visible with
    # status "no_key", because hiding it would answer "can I use OpenRouter?"
    # with silence.
    entries = _chat_model_entries()
    for provider in optional_chat_providers():
        if provider == chat_provider:
            continue  # already listed above as the configured chat endpoint
        env_var = PROVIDER_KEY_ENV.get(provider)
        count = sum(1 for entry in entries if entry.get("provider") == provider)
        detail = BILLED_DETAIL
        if env_var and not api_key_state(env_var)["configured"]:
            detail = (
                f"{BILLED_DETAIL} Set {env_var} in the environment to offer "
                "its models in the picker."
            )
        rows.append(provider_endpoint_row(
            provider,
            role="chat",
            model_source=(
                f"{count} preset{'' if count == 1 else 's'} from config.toml plus "
                "any id you type, chosen in the picker above"
            ),
            detail=detail,
        ))
    return rows


def endpoint_notices(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Build-level warnings shown under the endpoint list.

    **Branch seam.** The default turns every row that is not reachable into
    one line; a branch can append its own (llama-server not started, an
    OpenRouter key missing).

    Because this build lists one row per provider, a deployment with no
    OpenRouter key gets one line for OpenRouter rather than one per configured
    model. Lines the model picker already serves through ``model_notices`` are
    dropped, so nobody reads the identical sentence twice on one screen.
    """
    optional = set(optional_chat_providers())
    collected: list[str] = []
    for row in rows:
        status = row.get("status")
        if status == STATUS_NO_KEY:
            collected.append(_no_key_notice(row, optional))
        elif status == STATUS_UNREACHABLE:
            collected.append(f"{row.get('label')}: not reachable right now.")

    already_shown = _model_notices()
    notices: list[str] = []
    for notice in collected:
        if notice in notices or notice in already_shown:
            continue
        notices.append(notice)
    return notices


def _no_key_notice(row: dict[str, Any], optional: set[str]) -> str:
    """One line for a row whose API key is missing.

    An optional endpoint says what the missing key costs: its models are not
    in the picker at all. KIT keeps the plain sentence, because the KIT half
    of the picker still works off the offline fallback list when its key is
    unset.
    """
    label = row.get("label")
    hint = (row.get("api_key") or {}).get("hint")
    if row.get("provider") in optional and hint:
        return (f"{label}: no API key configured ({hint}), so its billed "
                "models are hidden from the picker.")
    return f"{label}: no API key configured."


def _model_notices() -> tuple[str, ...]:
    """What /api/meta already serves as ``model_notices`` (never raises)."""
    try:
        return get_catalog().notices
    except Exception:  # noqa: BLE001 - a notice list is never worth a 500
        return ()


def diagnostic_row(label: str, value: str, detail: Optional[str] = None) -> dict[str, Any]:
    """One read-only label/value fact. Keep values short and human."""
    return {"label": label, "value": value, "detail": detail}


def diagnostic_rows() -> list[dict[str, Any]]:
    """A few read-only facts about this build.

    **Branch seam.** Keep it small and useful (this is not a config dump) and
    never leak a secret or a host path. Return ``[]`` to hide the section.
    """
    retrieval = "hybrid (dense + BM25)" if get_hybrid_enabled() else "dense vectors"
    if get_reranker_enabled():
        retrieval += " + reranker"
    federation = (
        f"on, up to {get_federation_max_specialists()} specialists"
        if get_federation_enabled() else "off"
    )
    return [
        diagnostic_row("Retrieval", retrieval),
        diagnostic_row("Federation", federation),
        diagnostic_row("Live graph", "on" if get_live_graph_enabled() else "off"),
        # Name only: the container mounts its own file over this path.
        diagnostic_row("Config", CONFIG_PATH.name),
    ]


def settings_controls() -> dict[str, bool]:
    """Which of the always-known controls the panel offers. All true here;
    a branch can drop one without touching the React code."""
    return {
        "model": True,
        "temperature": True,
        "simplified_view": True,
        "live_graph": bool(get_live_graph_enabled()),
    }


def settings_meta() -> dict[str, Any]:
    """The declarative description of this build's settings panel.

    ``level`` comes from ``[frontend].settings_level``
    (``AMA_FRONTEND_SETTINGS_LEVEL``): ``minimal`` offers the controls only,
    ``full`` adds the endpoint list and the build facts.
    """
    level = get_frontend_settings_level()
    block: dict[str, Any] = {
        "level": level,
        "controls": settings_controls(),
        "endpoints": None,
        "diagnostics": None,
    }
    if level != LEVEL_FULL:
        return block
    rows = [row for row in endpoint_rows() if row]
    if rows:
        block["endpoints"] = {
            "label": ENDPOINTS_LABEL,
            "rows": rows,
            "notices": endpoint_notices(rows),
        }
    facts = [row for row in diagnostic_rows() if row]
    if facts:
        block["diagnostics"] = {"label": DIAGNOSTICS_LABEL, "rows": facts}
    return block


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
        "settings": settings_meta(),
    }


def reset_caches() -> None:
    """Forget the cached model catalog and default temperature (tests)."""
    global _catalog_cache, _default_temperature
    with _models_lock:
        _catalog_cache = None
    _default_temperature = None

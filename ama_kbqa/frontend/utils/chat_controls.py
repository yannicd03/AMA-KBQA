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


def filter_selectable_models(models: list[str]) -> list[str]:
    """Keep only KIT chat models worth offering in the demo dropdown.

    Drops embedding models (not valid chat models), Azure OpenAI models (with a
    carve-out for the open-source ``gpt-oss`` model which the demo keeps), and
    the generic ``standard extern`` / ``standard local`` KIT routing aliases,
    which are not concrete models worth offering in the picker.
    """
    out: list[str] = []
    for m in models:
        ml = m.lower()
        if "embedding" in ml:
            continue
        if ml.startswith("azure.") and "gpt-oss" not in ml:
            continue
        # Normalise away the provider prefix and separators so we catch
        # ``kit.standard-extern``, ``standard_local``, etc.
        norm = ml.split(".", 1)[-1].replace("-", " ").replace("_", " ")
        if norm in ("standard extern", "standard local"):
            continue
        out.append(m)
    return sorted(set(out))


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
    return f"{per_in} per 1M in and {per_out} per 1M out"


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

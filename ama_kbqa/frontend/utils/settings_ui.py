"""Streamlit widgets for the Settings page.

Kept separate from ``4_Settings.py`` so the widgets are importable (and
testable) without executing the page script.
"""

from __future__ import annotations

import streamlit as st

from ama_kbqa.frontend.utils.config_editor import (
    FETCHABLE_PROVIDERS,
    fetch_provider_models,
)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_models(provider: str) -> list[str]:
    """Cached wrapper around the endpoint model fetch (5-minute TTL)."""
    return fetch_provider_models(provider)


def model_field(label: str, provider: str, current: str, *, key: str) -> str:
    """Render a model selector for ``provider`` and return the chosen model.

    For providers that advertise a ``/models`` endpoint (e.g. KIT) this is a
    dropdown populated live from the endpoint, with a refresh button. For every
    other provider — or if the fetch fails — it falls back to a free-text input
    so the page never blocks on a network call.
    """
    if provider not in FETCHABLE_PROVIDERS:
        return st.text_input(label, value=current, key=key)

    try:
        models = _cached_models(provider)
    except Exception as e:  # noqa: BLE001 — any fetch failure → manual fallback
        st.warning(
            f"Couldn't fetch {provider} models — {e} Enter the model name manually."
        )
        value = st.text_input(label, value=current, key=f"{key}_text")
        if st.button("🔄 Retry fetch", key=f"{key}_retry"):
            _cached_models.clear()
            st.rerun()
        return value

    # Keep the currently-configured model selectable even if the endpoint no
    # longer lists it (e.g. a model that was retired or renamed).
    options = list(models) if (current in models or not current) else [current, *models]
    index = options.index(current) if current in options else 0
    value = st.selectbox(
        label,
        options=options,
        index=index,
        key=f"{key}_select",
        help=f"Fetched live from the {provider} /models endpoint.",
    )
    if st.button("🔄 Refresh models", key=f"{key}_refresh"):
        _cached_models.clear()
        st.rerun()
    return value

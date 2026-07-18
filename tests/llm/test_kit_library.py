"""Import-surface test for the ``chatkit`` dependency (formerly the vendored
``ama_kbqa.llm`` module, now published at github.com/yannicd03/ChatKIT).

The retry-core and KIT-client implementation tests now live in the ChatKIT
package itself (its own ``tests/test_retry.py`` etc.) — no need to duplicate
them here. This module only guards the contract this repo actually relies on:

* every name AMA's code imports from ``chatkit`` is importable from the core
  install (no langchain-openai / langchain-core required);
* the LangChain-only surface (``ChatKIT``, ``get_kit_model``, ...) is NOT
  silently usable in this environment — i.e. the ``chatkit[langchain]`` extra
  hasn't snuck in as a transitive dependency of something else.
"""

from __future__ import annotations

import pytest


def test_core_surface_is_importable():
    """Every chatkit name AMA's code uses must be importable eagerly (core install,
    no langchain dependency)."""
    from chatkit import (  # noqa: F401
        DEFAULT_BACKOFF_STEPS,
        DEFAULT_KIT_MODEL,
        DEFAULT_MAX_ATTEMPTS,
        KIT_API_KEY_ENV,
        KIT_BASE_URL,
        KIT_VPN_HOST,
        NON_TRANSIENT_MARKERS,
        TRANSIENT_MARKERS,
        TransientRetry,
        create_with_retry,
        is_transient_error,
        kit_api_key,
        kit_chat_create,
        kit_client,
    )


def test_langchain_is_not_installed():
    """Guard against the `chatkit[langchain]` extra (or bare langchain) sneaking in
    as a transitive dependency: this repo is deliberately langchain-free."""
    with pytest.raises(ImportError):
        import langchain  # noqa: F401


def test_langchain_only_names_raise_helpful_error():
    """Without the `chatkit[langchain]` extra, LangChain-only names raise a helpful
    ImportError rather than importing successfully or failing obscurely."""
    import chatkit

    for name in (
        "ChatKIT",
        "get_kit_model",
        "make_chat_kit_class",
        "normalize_kit_messages",
        "salvage_tool_calls",
    ):
        with pytest.raises(ImportError, match="langchain"):
            getattr(chatkit, name)

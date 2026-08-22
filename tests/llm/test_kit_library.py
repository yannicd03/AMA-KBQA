"""Import-surface test for the ``chatkit`` dependency (formerly the vendored
``ama_kbqa.llm`` module, now published at github.com/yannicd03/ChatKIT).

The retry-core and KIT-client implementation tests now live in the ChatKIT
package itself (its own ``tests/test_retry.py`` etc.) — no need to duplicate
them here. This module only guards the contract this repo actually relies on:
every name AMA's code imports from ``chatkit`` is importable from the core
install (no langchain-openai / langchain-core required beyond what the repo
already depends on).

Formerly this module also asserted the repo was langchain-free and that
``chatkit``'s LangChain-only surface (``ChatKIT``, ``get_kit_model``, ...)
raised a helpful ``ImportError`` without the ``chatkit[langchain]`` extra.
That invariant is superseded as of the LangGraph rewrite
(``.agent/Tasks/active/langgraph-rewrite.md``): the repo now depends on
``langchain-core``/``langchain-openai``/``langgraph`` directly, and
``ama_kbqa/graph/model.py`` uses ``chatkit.client.get_kit_model`` for the
"kit" provider. See that PRD's Phase 1 dependency/test-removal notes and the
ADR superseding ``Decisions/transient-retry-and-chatkit-extraction.md``'s
"langchain-free repo" clause.
"""

from __future__ import annotations


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

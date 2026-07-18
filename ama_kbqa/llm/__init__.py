"""Reusable LLM-endpoint helpers (retry core + KIT client), consolidated here so the
same transient-resilience logic backs the raw OpenAI SDK path, a LangGraph-ready
``ChatKIT``, and this repo's ``BaseKBQAAgent``.

Designed to be lifted into a standalone package later: ``retry`` is provider-agnostic
and ``kit`` holds only KIT-endpoint specifics. ``ChatKIT`` is imported lazily (it needs
``langchain-openai``); the retry core and raw-SDK helpers have no langchain dependency.
"""

from ama_kbqa.llm.retry import (
    DEFAULT_BACKOFF_STEPS,
    DEFAULT_MAX_ATTEMPTS,
    TransientRetry,
    create_with_retry,
    is_transient_error,
)

__all__ = [
    "TransientRetry",
    "create_with_retry",
    "is_transient_error",
    "DEFAULT_BACKOFF_STEPS",
    "DEFAULT_MAX_ATTEMPTS",
    "kit_client",
    "kit_chat_create",
    "ChatKIT",
]


def __getattr__(name):
    # Lazily expose the KIT surface so importing ama_kbqa.llm doesn't require the
    # openai/langchain deps until something actually uses them.
    if name in ("kit_client", "kit_chat_create", "ChatKIT"):
        from ama_kbqa.llm import kit

        return getattr(kit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

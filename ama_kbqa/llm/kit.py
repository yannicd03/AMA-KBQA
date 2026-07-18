"""KIT AI Toolbox endpoint: OpenAI- and LangGraph-compatible access, one retry core.

The KIT endpoint is OpenAI-compatible, so this is deliberately thin — it bakes in the
base URL, the ``KIT_API_KEY`` lookup, sane socket timeouts, and (crucially) KIT's
transient-error handling from :mod:`ama_kbqa.llm.retry`. KIT's "Open WebUI: Server
Connection Error" arrives with a non-5xx status, so the OpenAI SDK's own retries miss
it; our :class:`TransientRetry` catches it by message.

Two surfaces over the same core:
  * :func:`kit_client` / :func:`kit_chat_create` — raw ``openai.OpenAI`` path, no
    langchain dependency. What OpenAI-SDK code (this repo's BaseKBQAAgent) uses.
  * :class:`ChatKIT` — a ``langchain_openai.ChatOpenAI`` subclass, so it drops into any
    LangGraph graph / ``bind_tools`` flow. Imported lazily; needs ``langchain-openai``.
"""

from __future__ import annotations

import os
from typing import Optional

from ama_kbqa.llm.retry import TransientRetry, create_with_retry

# Overridable via env so the same code serves a mirror/self-hosted KIT gateway.
KIT_BASE_URL = os.environ.get("KIT_BASE_URL", "https://ki-toolbox.scc.kit.edu/api/v1")
KIT_API_KEY_ENV = "KIT_API_KEY"
DEFAULT_KIT_MODEL = os.environ.get("KIT_DEFAULT_MODEL", "kit.qwen3.5-397b-A17b")


def kit_api_key(explicit: Optional[str] = None) -> str:
    """Resolve the KIT API key: explicit arg > ``KIT_API_KEY`` env. Raises if unset."""
    key = explicit or os.environ.get(KIT_API_KEY_ENV)
    if not key:
        raise KeyError(
            f"{KIT_API_KEY_ENV} is not set and no api_key was passed to ChatKIT/kit_client."
        )
    return key


def kit_client(api_key: Optional[str] = None, base_url: Optional[str] = None, **openai_kwargs):
    """A configured ``openai.OpenAI`` pointed at KIT.

    ``max_retries=0`` because :class:`TransientRetry` owns retries — layering the SDK's
    own retries on top would double the backoff and hide the KIT-specific transient.
    """
    from openai import OpenAI  # local import: keep the retry core import-light
    import httpx

    kwargs = {
        "base_url": base_url or KIT_BASE_URL,
        "api_key": kit_api_key(api_key),
        # Structured timeout so a socket hang trips connect/read/write/pool budgets
        # individually rather than blocking on a single bulk value.
        "timeout": httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0),
        "max_retries": 0,
    }
    kwargs.update(openai_kwargs)
    return OpenAI(**kwargs)


def kit_chat_create(
    client=None,
    *,
    retry: Optional[TransientRetry] = None,
    on_retry=None,
    **create_kwargs,
):
    """One ``chat.completions.create`` against KIT with stepped-backoff retry.

    Builds a default client/model if not supplied. Pass a shared ``retry`` to keep the
    backoff level persistent across a sequence of calls.
    """
    client = client or kit_client()
    create_kwargs.setdefault("model", DEFAULT_KIT_MODEL)
    return create_with_retry(client, retry=retry, on_retry=on_retry, **create_kwargs)


def _build_chatkit_class():
    """Define ChatKIT against ChatOpenAI. Deferred so importing this module (and the
    retry core) never requires langchain."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "ChatKIT requires the 'langchain-openai' package. Install it with "
            "`uv add langchain-openai` (the raw-SDK surface kit_client()/kit_chat_create() "
            "has no langchain dependency)."
        ) from exc

    class ChatKIT(ChatOpenAI):
        """LangGraph-compatible chat model for the KIT AI Toolbox endpoint.

        A ``ChatOpenAI`` preconfigured with KIT's base URL + ``KIT_API_KEY``. Usable
        anywhere ``ChatOpenAI`` is (LangGraph nodes, ``.bind_tools()``, streaming,
        structured output).

        Note: connection/5xx/429 retries are handled by ``max_retries`` (OpenAI SDK).
        KIT's non-5xx "Open WebUI" transient is fully handled on the raw-SDK surface
        (:func:`kit_chat_create`); wrap this model with ``TransientRetry`` if you need
        that exact handling on the LangChain path too.
        """

        def __init__(
            self,
            model: str = DEFAULT_KIT_MODEL,
            *,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            max_retries: int = 3,
            **kwargs,
        ):
            super().__init__(
                model=model,
                api_key=kit_api_key(api_key),
                base_url=base_url or KIT_BASE_URL,
                max_retries=max_retries,
                **kwargs,
            )

    return ChatKIT


def __getattr__(name):
    if name == "ChatKIT":
        return _build_chatkit_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

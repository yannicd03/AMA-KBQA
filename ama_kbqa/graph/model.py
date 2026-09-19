"""Chat-model factory for the Phase 1 LangGraph engine.

Mirrors ``ama_kbqa.config._create_client`` (base_url, API-key env-var
mapping, the ``httpx.Timeout(connect=20, read=60, write=10, pool=5)`` budget,
OpenRouter headers/provider preferences, ``AMA_LLM_SEED``) but returns a
LangChain ``BaseChatModel`` instead of a raw ``openai.OpenAI`` client, because
``ama_kbqa.graph.builder``'s ``call_model`` node drives the model through
LangChain's ``bind_tools``/``ainvoke`` surface.

Retry semantics (see ``base_agent.py:_create_with_retry`` for the legacy
behaviour this mirrors):

- ``provider == "kit"`` returns ``chatkit.client.get_kit_model(...)``. Its
  ``ChatKIT`` model already wraps every ``_agenerate`` call in whatever
  ``TransientRetry`` instance is passed as ``retry=`` (see
  ``chatkit/client.py:ChatKIT._agenerate``/``_effective_retry``). Passing the
  agent's own ``self._retry`` here keeps the same per-question,
  self-resetting backoff ladder ``_create_with_retry`` gives the legacy path
  — retry is NOT wrapped again in the calling node.
- Every other provider returns a plain ``ChatOpenAI(max_retries=0)``, which
  has no such built-in hook; the calling node
  (``ama_kbqa.graph.builder.call_model``) wraps its ``ainvoke`` call in the
  same ``TransientRetry`` instance instead.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import httpx
from langchain_core.language_models import BaseChatModel

from ama_kbqa.config import (
    _get_api_key,
    get_chat_max_tokens,
    get_chat_model_name,
    get_chat_seed,
    get_chat_temperature,
    get_provider_preferences,
    get_synthesis_max_tokens,
    get_synthesis_model_name,
    get_synthesis_provider_preferences,
    get_synthesis_temperature,
    load_config,
)

# Mirrors config.py::_create_client's OpenRouter-specific headers exactly.
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/MaxKlat29/AMAKBQA",
    "X-Title": "ama-kbqa",
}

_TIMEOUT = httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0)


def build_chat_model(
    provider: str,
    *,
    purpose: Literal["chat", "synthesis"] = "chat",
    max_retries: int = 0,
    retry: Optional[Any] = None,
) -> BaseChatModel:
    """Build a LangChain chat model for ``provider``.

    Args:
        provider: provider name as it appears in config.toml (e.g. "kit",
            "openrouter"). Callers pass whichever provider the relevant
            config section names — ``config["llm"]["chat_provider"]`` for
            ``purpose="chat"``, ``config["synthesis"]["synthesis_provider"]``
            for ``purpose="synthesis"`` — exactly as ``get_chat_client()``/
            ``get_synthesis_client()`` do today. The model name is looked up
            independently via ``get_chat_model_name()``/
            ``get_synthesis_model_name()``, matching those functions' own
            (provider-independent) config keys.
        purpose: selects which config getters (temperature, max_tokens,
            provider preferences, model name) to use.
        max_retries: the underlying SDK's own retry count. 0 by default —
            see the module docstring for why.
        retry: a ``chatkit.retry.TransientRetry`` instance. Only consulted
            for ``provider == "kit"``; ignored (the caller wraps ``ainvoke``
            itself) for every other provider.

    Returns:
        A ready-to-use LangChain ``BaseChatModel``.

    Raises:
        ValueError: if ``provider`` is not a section in config.toml.
        KeyError: if the required API-key environment variable is unset.
    """
    config = load_config()
    if provider not in config:
        raise ValueError(
            f"Provider '{provider}' not found in config.toml. "
            f"Available providers: {list(config.keys())}"
        )
    provider_config = config[provider]
    base_url = provider_config["base_url"]
    api_key = _get_api_key(provider, provider_config)

    if purpose == "synthesis":
        model_name = get_synthesis_model_name()
        temperature = get_synthesis_temperature()
        max_tokens = get_synthesis_max_tokens()
        provider_prefs = get_synthesis_provider_preferences()
    else:
        model_name = get_chat_model_name()
        temperature = get_chat_temperature()
        max_tokens = get_chat_max_tokens()
        provider_prefs = get_provider_preferences()

    seed = get_chat_seed()
    extra_body = {"provider": provider_prefs} if provider_prefs else None

    if provider == "kit":
        from chatkit.client import get_kit_model

        kit_kwargs: Dict[str, Any] = {}
        if seed is not None:
            kit_kwargs["seed"] = seed
        if extra_body is not None:
            kit_kwargs["extra_body"] = extra_body
        return get_kit_model(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=_TIMEOUT,
            streaming=False,
            preflight=False,
            retry=retry,
            **kit_kwargs,
        )

    from langchain_openai import ChatOpenAI

    openai_kwargs: Dict[str, Any] = {}
    if seed is not None:
        openai_kwargs["seed"] = seed
    if extra_body is not None:
        openai_kwargs["extra_body"] = extra_body
    if provider == "openrouter":
        openai_kwargs["default_headers"] = _OPENROUTER_HEADERS

    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=_TIMEOUT,
        max_retries=max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        **openai_kwargs,
    )

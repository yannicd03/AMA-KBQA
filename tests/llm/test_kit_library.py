"""Tests for the consolidated LLM library (retry core + KIT surface).

Hermetic: the retry backoff is driven by an injected fake clock, and the KIT client
is exercised via a stub so no network or real key is needed.
"""

from __future__ import annotations

import types

import pytest

from ama_kbqa.llm import (
    TransientRetry,
    create_with_retry,
    is_transient_error,
)
from ama_kbqa.llm import kit


def _client(create):
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


# --- retry core ------------------------------------------------------------------
def test_is_transient_error_classifies():
    assert is_transient_error(Exception("Open WebUI: Server Connection Error"))
    assert is_transient_error(Exception("429 rate limit"))
    assert not is_transient_error(Exception("401 unauthorized"))
    # Auth wins even if a transient-looking token co-occurs.
    assert not is_transient_error(Exception("401 unauthorized (connection error)"))


def test_transient_retry_escalates_and_resets():
    waits: list[float] = []
    r = TransientRetry(sleep=waits.append)
    n = {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] <= 3:
            raise Exception("503 service unavailable")
        return "OK"

    assert r.run(fn) == "OK"
    assert waits == [2.0, 5.0, 10.0]
    assert r.level == 0  # reset after success


def test_transient_retry_reraises_deterministic():
    waits: list[float] = []
    r = TransientRetry(sleep=waits.append)
    with pytest.raises(Exception, match="invalid api key"):
        r.run(lambda: (_ for _ in ()).throw(Exception("invalid api key")))
    assert waits == []


def test_create_with_retry_wraps_client():
    waits: list[float] = []
    r = TransientRetry(sleep=waits.append)
    n = {"i": 0}

    def create(**kw):
        n["i"] += 1
        if n["i"] == 1:
            raise Exception("connection reset")
        return "resp"

    out = create_with_retry(_client(create), retry=r, model="m", messages=[])
    assert out == "resp"
    assert waits == [2.0]


# --- KIT surface -----------------------------------------------------------------
def test_kit_api_key_prefers_explicit(monkeypatch):
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    assert kit.kit_api_key("sk-explicit") == "sk-explicit"
    monkeypatch.setenv("KIT_API_KEY", "sk-env")
    assert kit.kit_api_key() == "sk-env"


def test_kit_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    with pytest.raises(KeyError, match="KIT_API_KEY"):
        kit.kit_api_key()


def test_kit_client_uses_kit_base_url(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("KIT_API_KEY", "sk-env")
    kit.kit_client()
    assert captured["base_url"] == kit.KIT_BASE_URL
    assert captured["api_key"] == "sk-env"
    assert captured["max_retries"] == 0  # TransientRetry owns retries


def test_chatkit_requires_langchain():
    """Without langchain-openai installed, ChatKIT access raises a helpful ImportError
    (the raw-SDK surface stays usable)."""
    pytest.importorskip  # keep import-time light
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="langchain-openai"):
            _ = kit.ChatKIT  # triggers module __getattr__ -> _build_chatkit_class
    else:  # pragma: no cover - only when the optional dep is present
        assert kit.ChatKIT is not None

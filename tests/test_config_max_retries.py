"""Tests for the max_retries override threaded through config.py's client builders.

BaseKBQAAgent and Orchestrator now wrap their chat/synthesis clients in a
TransientRetry (chatkit.retry). Layering the OpenAI SDK's own retries
under that stepped backoff would double the backoff and, per
chatkit.raw's rationale, still miss KIT's non-5xx "Open WebUI: Server
Connection Error" transient — so those call sites now build their client with
max_retries=0. Callers that stay unwrapped (postprocessing.py's judge/choice
client, get_embedding_client) must keep the SDK's own retries (default 3).

Uses the "llamacpp" provider from config.toml because it carries an inline
api_key, so client construction needs no environment variables / network.
"""

from __future__ import annotations

from ama_kbqa.config import _create_client, get_chat_client, get_synthesis_client


def test_create_client_defaults_to_three_retries_when_unset():
    client = _create_client("llamacpp", model_type="chat")
    assert client.max_retries == 3


def test_create_client_honors_explicit_max_retries_zero():
    client = _create_client("llamacpp", model_type="chat", max_retries=0)
    assert client.max_retries == 0


def test_create_client_embedding_path_unaffected_by_default():
    """Unwrapped callers (e.g. get_embedding_client) never pass max_retries, so
    they must keep the SDK's own retry behavior."""
    client = _create_client("llamacpp", model_type="embedding")
    assert client.max_retries == 3


def test_get_chat_client_threads_max_retries_through(monkeypatch):
    import ama_kbqa.config as config_module

    monkeypatch.setattr(
        config_module, "load_config", lambda: {"llm": {"chat_provider": "llamacpp"}, "llamacpp": {
            "base_url": "http://localhost:8080/v1", "api_key": "sk-no-key-required",
        }},
    )

    wrapped = get_chat_client(max_retries=0)
    unwrapped = get_chat_client()

    assert wrapped.max_retries == 0
    assert unwrapped.max_retries == 3


def test_get_synthesis_client_threads_max_retries_through(monkeypatch):
    import ama_kbqa.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "llm": {"chat_provider": "llamacpp"},
            "synthesis": {"synthesis_provider": "llamacpp"},
            "llamacpp": {
                "base_url": "http://localhost:8080/v1",
                "api_key": "sk-no-key-required",
            },
        },
    )

    wrapped = get_synthesis_client(max_retries=0)
    unwrapped = get_synthesis_client()

    assert wrapped.max_retries == 0
    assert unwrapped.max_retries == 3

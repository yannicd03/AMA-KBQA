"""Tests for the Settings-page config helpers, focused on the provider
``/models`` fetch that backs the KIT model dropdown."""

from __future__ import annotations

import httpx
import pytest

from ama_kbqa.frontend.utils import config_editor as ce


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err", request=httpx.Request("GET", "http://x/models"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


@pytest.fixture
def kit_config(monkeypatch):
    monkeypatch.setattr(
        ce, "load_config_raw",
        lambda: {"kit": {"base_url": "https://kit.example/api/v1"}},
    )
    monkeypatch.setenv("KIT_API_KEY", "secret-key")


def test_fetch_returns_sorted_unique_ids(kit_config, monkeypatch):
    payload = {"data": [{"id": "b-model"}, {"id": "a-model"}, {"id": "a-model"}]}
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp(payload))
    assert ce.fetch_provider_models("kit") == ["a-model", "b-model"]


def test_fetch_sends_bearer_token_to_models_url(kit_config, monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["auth"] = kwargs.get("headers", {}).get("Authorization")
        return _FakeResp({"data": [{"id": "m"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    ce.fetch_provider_models("kit")
    assert seen["url"] == "https://kit.example/api/v1/models"
    assert seen["auth"] == "Bearer secret-key"


def test_fetch_raises_on_http_error(kit_config, monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp({}, status=401))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        ce.fetch_provider_models("kit")


def test_fetch_raises_on_empty_model_list(kit_config, monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _FakeResp({"data": []}))
    with pytest.raises(RuntimeError, match="no models"):
        ce.fetch_provider_models("kit")


def test_fetch_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(
        ce, "load_config_raw",
        lambda: {"kit": {"base_url": "https://kit.example/api/v1"}},
    )
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not set"):
        ce.fetch_provider_models("kit")


def test_fetch_raises_when_base_url_missing(monkeypatch):
    monkeypatch.setattr(ce, "load_config_raw", lambda: {"kit": {}})
    monkeypatch.setenv("KIT_API_KEY", "secret-key")
    with pytest.raises(RuntimeError, match="No base_url"):
        ce.fetch_provider_models("kit")


def test_settings_retrieval_section_reaches_config_getters(monkeypatch):
    """The [retrieval] keys written by the Settings page must match the keys
    the config getters read, so 'Apply to Session' actually takes effect."""
    import ama_kbqa.config as cfg

    # Ensure the env overlay doesn't mask the session values under test.
    for suffix in ("HYBRID_ENABLED", "FUSION", "RERANKER_ENABLED",
                   "PREFETCH_LIMIT", "RERANK_CANDIDATES"):
        monkeypatch.delenv(cfg.RETRIEVAL_ENV_PREFIX + suffix, raising=False)

    edited = {
        "retrieval": {
            "hybrid_enabled": True,
            "fusion": "dbsf",
            "prefetch_limit": 33,
            "reranker_enabled": True,
            "reranker_model": "some/model",
            "rerank_candidates": 12,
        }
    }
    monkeypatch.setattr(cfg, "_config_cache", None)
    ce.apply_to_session(edited)

    assert cfg.get_hybrid_enabled() is True
    assert cfg.get_fusion() == "dbsf"
    assert cfg.get_prefetch_limit() == 33
    assert cfg.get_reranker_enabled() is True
    assert cfg.get_reranker_model() == "some/model"
    assert cfg.get_rerank_candidates() == 12

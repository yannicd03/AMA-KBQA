"""GET /api/meta and GET /api/health: shape, suggestion parsing, model list."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ama_kbqa.api import app as app_module
from ama_kbqa.api import meta
from ama_kbqa.frontend.utils.agent_factory import AGENT_INFO, AGENT_SUGGESTIONS
from ama_kbqa.frontend.utils.chat_controls import _OFFLINE_FALLBACK_MODELS

from .conftest import MODEL, OTHER_MODEL


class TestParseSuggestionLabel:
    def test_streamlit_markup_is_split_into_parts(self):
        assert meta.parse_suggestion_label(":blue[:material/movie:] Director of Inception") == {
            "label": "Director of Inception",
            "icon": "movie",
            "color": "blue",
        }

    def test_underscored_icon_and_apostrophe(self):
        parsed = meta.parse_suggestion_label(":orange[:material/location_on:] Einstein's birthplace")
        assert parsed == {"label": "Einstein's birthplace", "icon": "location_on", "color": "orange"}

    def test_plain_text_is_unparseable(self):
        assert meta.parse_suggestion_label("Just a label") == {
            "label": "Just a label", "icon": None, "color": None,
        }

    def test_unparseable_markup_is_stripped(self):
        parsed = meta.parse_suggestion_label(":red[Hot] topic :material/star:")
        assert parsed == {"label": "Hot topic", "icon": None, "color": None}

    def test_every_shipped_suggestion_parses(self):
        # Guards drift: a new AGENT_SUGGESTIONS key in another markup shape
        # would silently lose its chip icon in the React app.
        for items in AGENT_SUGGESTIONS.values():
            for key in items:
                parsed = meta.parse_suggestion_label(key)
                assert parsed["icon"] and parsed["color"], key
                assert ":" not in parsed["label"] or "material" not in parsed["label"]


@pytest.fixture
def client(monkeypatch, patch_models):
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    for var in ("DEMO_MODE", "DEMO_MAX_QUERIES_PER_SESSION", "DEMO_MIN_SECONDS_BETWEEN_QUERIES"):
        monkeypatch.delenv(var, raising=False)
    with TestClient(app_module.create_app()) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_meta_shape(client):
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {
        "title", "agents", "default_agent", "suggestions", "models",
        "default_model", "default_temperature", "live_graph", "demo",
    }
    assert body["title"] == "AMA-KBQA Assistant"
    assert body["default_agent"] == "Orchestrator (Router)"
    assert isinstance(body["default_temperature"], float)
    assert body["live_graph"] is True
    assert body["demo"] == {
        "enabled": False, "max_queries_per_session": 20, "min_seconds_between_queries": 3.0,
    }

    # Agents: AGENT_INFO order, flags derived from the orchestrator modes.
    assert [a["name"] for a in body["agents"]] == list(AGENT_INFO)
    by_name = {a["name"]: a for a in body["agents"]}
    assert by_name["Orchestrator (Router)"] | {} == {
        "name": "Orchestrator (Router)",
        **{k: AGENT_INFO["Orchestrator (Router)"][k]
           for k in ("tagline", "description", "databases", "tools")},
        "orchestrator": True, "federated": False, "followups": False,
    }
    assert by_name["Orchestrator (Federated)"]["federated"] is True
    assert by_name["KQAPro"]["orchestrator"] is False
    assert by_name["KQAPro"]["followups"] is True
    assert by_name["SciQA"]["followups"] is True

    # Suggestions: one list per agent, parsed chips with their question.
    assert set(body["suggestions"]) == set(AGENT_SUGGESTIONS)
    first = body["suggestions"]["Orchestrator (Router)"][0]
    assert first == {
        "label": "Director of Inception", "icon": "movie", "color": "blue",
        "question": "Who is the director of Inception?",
    }


def test_meta_models_come_from_the_filtered_kit_catalog(client):
    body = client.get("/api/meta").json()
    # Embedding and external models are filtered out; ids sorted.
    assert [m["id"] for m in body["models"]] == [MODEL, OTHER_MODEL]
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id[MODEL]["name"] == "GPT-OSS 120B"
    assert by_id[OTHER_MODEL]["name"] == "Mistral Small 4"
    # Plain "$" (no Streamlit escaping); null when the pricing table has no entry.
    price = by_id[MODEL]["price"]
    assert price and price.startswith("$") and "\\" not in price
    assert "per 1M in and" in price
    assert by_id[OTHER_MODEL]["price"] is None
    # The demo's preferred default wins when available.
    assert body["default_model"] == OTHER_MODEL


def test_meta_falls_back_to_offline_models(monkeypatch):
    def boom(provider, **_kwargs):
        raise RuntimeError("no key")

    monkeypatch.setattr(meta, "fetch_provider_models_meta", boom)
    meta.reset_caches()
    try:
        models = meta.get_models()
    finally:
        meta.reset_caches()
    assert [m["id"] for m in models] == list(_OFFLINE_FALLBACK_MODELS)
    # No endpoint name: the id without the "kit." prefix.
    assert all(m["name"] == m["id"][len("kit."):] for m in models)


def test_model_list_is_cached_for_the_ttl(monkeypatch, patch_models):
    now = {"t": 1000.0}
    monkeypatch.setattr(meta, "_clock", lambda: now["t"])
    meta.get_models()
    meta.get_models()
    assert patch_models == ["kit"]  # one fetch for two calls

    now["t"] += meta.MODELS_TTL_SECONDS + 1
    meta.get_models()
    assert patch_models == ["kit", "kit"]


def test_demo_settings_follow_env(client, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_QUERIES_PER_SESSION", "5")
    monkeypatch.setenv("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "1.5")
    assert client.get("/api/meta").json()["demo"] == {
        "enabled": True, "max_queries_per_session": 5, "min_seconds_between_queries": 1.5,
    }


def test_default_temperature_is_pinned_at_startup(monkeypatch, patch_models):
    # apply_chat_settings writes the visitor's temperature into this env var;
    # /meta must keep reporting the value the process started with.
    monkeypatch.setenv("AMA_KBQA_CHAT_TEMPERATURE", "0.7")
    with TestClient(app_module.create_app()) as c:
        monkeypatch.setenv("AMA_KBQA_CHAT_TEMPERATURE", "1.9")
        assert c.get("/api/meta").json()["default_temperature"] == 0.7

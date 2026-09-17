"""demo-v2-int's own Endpoints row: the OpenRouter chat endpoint.

Deliberately NOT in ``test_meta_settings.py``. That file pins the settings
contract all three demo branches implement, and an OpenRouter row is this
branch's addition — the same split demo-booth made into
``test_meta_settings_booth.py`` (see .agent/System/demo_react_frontend.md,
"Forward-merge friction to expect").

What matters here: the row is *derived from config*, not hard-coded; a build
without the key keeps the row (and says what the missing key costs) rather
than hiding the endpoint; and none of it ever serialises key material.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import ama_kbqa.config as cfg
from ama_kbqa.api import app as app_module
from ama_kbqa.api import meta

OR_SECRET = "sk-or-v1-0123456789abcdef-super-secret"


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setattr(meta, "get_frontend_settings_level", lambda: meta.LEVEL_FULL)


@pytest.fixture
def kit_key(monkeypatch):
    monkeypatch.setenv("KIT_API_KEY", "sk-kit-0123456789abcdef")


@pytest.fixture
def client(monkeypatch, patch_models, kit_path, kit_key, full):
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    with TestClient(app_module.create_app()) as c:
        yield c


def endpoints_of(client) -> dict:
    return client.get("/api/meta").json()["settings"]["endpoints"]


def rows_by_provider(client) -> dict:
    return {row["provider"]: row for row in endpoints_of(client)["rows"]}


# ---------------------------------------------------------------------------
# The row itself
# ---------------------------------------------------------------------------

def test_openrouter_chat_row_is_listed_next_to_kit(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    rows = endpoints_of(client)["rows"]
    by_provider = {row["provider"]: row for row in rows}
    assert set(by_provider) == {"kit", "openrouter"}

    row = by_provider["openrouter"]
    assert row["role"] == "chat"
    assert row["id"] == "openrouter-chat"
    assert row["label"] == meta.provider_label("openrouter")
    assert row["base_url"] == cfg.load_config()["openrouter"]["base_url"]
    assert row["status"] == "ok"
    # Presence and the env var NAME only — never the key.
    assert row["api_key"] == {"configured": True, "hint": "OPENROUTER_API_KEY"}
    # A billed endpoint says so, in both key states.
    assert "billed" in row["detail"].lower()
    # One row per provider, not one per configured model.
    assert row["model"] is None
    assert "picker" in row["model_source"]


def test_the_row_is_derived_from_config_not_hard_coded(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    assert "openrouter" in rows_by_provider(client)

    # Drop the config entries and the row goes with them.
    monkeypatch.setattr(meta, "get_frontend_chat_models", list)
    assert set(rows_by_provider(client)) == {"kit"}


def test_optional_chat_providers_comes_from_the_config_entries(monkeypatch):
    monkeypatch.setattr(meta, "get_frontend_chat_models", lambda: [
        {"provider": "openrouter", "id": "a/b"},
        {"provider": "openrouter", "id": "c/d"},
        {"provider": "kit", "id": "kit.x"},
    ])
    # De-duplicated, KIT excluded (it is the configured chat provider already).
    assert meta.optional_chat_providers() == ["openrouter"]


def test_a_broken_config_degrades_to_kit_only(client, monkeypatch):
    def boom():
        raise ValueError("hand-edited config.toml")

    monkeypatch.setattr(meta, "get_frontend_chat_models", boom)
    # /api/meta must still answer rather than 500 mid-demo.
    assert set(rows_by_provider(client)) == {"kit"}


def test_the_shipped_config_really_offers_the_two_presets():
    """Guards the TOML array-table hazard from the consumer side: a stray
    plain key written under [[frontend.chat_models]] would silently land
    inside the last entry and change what the picker offers."""
    entries = cfg.get_frontend_chat_models()
    assert [(e["provider"], e["id"]) for e in entries] == [
        ("openrouter", "deepseek/deepseek-v4-pro"),
        ("openrouter", "deepseek/deepseek-v4.1-flash"),
    ]
    assert [e["name"] for e in entries] == ["DeepSeek V4 Pro", "DeepSeek V4.1 Flash"]
    # settings_level must still be a [frontend] key, not a key of the last
    # array entry (which is exactly what bad ordering produces). Its *value*
    # is branch-specific ("full" on this branch, "minimal" on demo-public),
    # so assert only that it still resolves to a valid level; pinning "full"
    # here breaks demo-public on every forward merge. The key's placement
    # against the shipped config is guarded by test_config_frontend_settings.py.
    assert cfg.get_frontend_settings_level() in cfg.FRONTEND_SETTINGS_LEVELS
    assert all(e["default"] is False for e in entries)


# ---------------------------------------------------------------------------
# Degraded state
# ---------------------------------------------------------------------------

def test_no_key_keeps_the_row_and_explains_what_it_costs(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    section = endpoints_of(client)
    row = {r["provider"]: r for r in section["rows"]}["openrouter"]

    # Still listed: hiding it would answer "can I use OpenRouter?" with silence.
    assert row["status"] == "no_key"
    assert row["api_key"] == {"configured": False, "hint": "OPENROUTER_API_KEY"}
    assert "OPENROUTER_API_KEY" in row["detail"]

    assert section["notices"] == [
        "OpenRouter: no API key configured (OPENROUTER_API_KEY), so its billed "
        "models are hidden from the picker."
    ]


def test_a_notice_the_picker_already_shows_is_not_repeated(client, monkeypatch):
    """The model picker serves its own ``model_notices``; the endpoint list
    must not print the identical sentence a second time on the same screen."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rows = endpoints_of(client)["rows"]
    duplicated = meta._no_key_notice(
        {r["provider"]: r for r in rows}["openrouter"], {"openrouter"}
    )
    monkeypatch.setattr(meta, "_model_notices", lambda: (duplicated,))
    assert meta.endpoint_notices(rows) == []


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def test_the_openrouter_key_never_reaches_the_client(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    raw = client.get("/api/meta").text
    assert OR_SECRET not in raw
    # Not even a masked head or tail of it.
    assert OR_SECRET[:12] not in raw
    assert OR_SECRET[-8:] not in raw
    assert "secret" not in raw.lower()


def test_no_host_path_is_served(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", OR_SECRET)
    raw = client.get("/api/meta").text
    assert str(cfg.CONFIG_PATH) not in raw
    assert str(cfg.CONFIG_PATH.parent) not in raw
    # The config is named, never located.
    assert "config.toml" in raw

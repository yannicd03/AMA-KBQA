"""The ``settings`` block of GET /api/meta: the declarative description of
the settings panel the React app renders.

These tests pin the *contract* the three demo branches implement against
(demo-booth, demo-llamacpp, demo-public), so they check the shape and the
guarantees rather than the exact KIT wording:

* the level flag switches the endpoint list and the build facts on and off;
* endpoint rows carry a fixed key set, so the panel renders them uniformly;
* no API key, or any part of one, is ever serialised;
* rows a branch adds through the seams pass through untouched, including
  providers and statuses this code line has never heard of.

Both picker shapes are covered (``client`` forces the KIT-only path,
``provider_harness`` installs the provider-aware one), so the file runs
unchanged on every branch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import ama_kbqa.config as cfg
from ama_kbqa.api import app as app_module
from ama_kbqa.api import meta

SECRET = "sk-kit-0123456789abcdef-super-secret"
ROW_KEYS = {
    "id", "label", "role", "provider", "base_url", "model",
    "model_source", "api_key", "status", "detail",
}


@pytest.fixture
def kit_key(monkeypatch):
    monkeypatch.setenv("KIT_API_KEY", SECRET)
    return SECRET


@pytest.fixture
def full(monkeypatch):
    monkeypatch.setattr(meta, "get_frontend_settings_level", lambda: meta.LEVEL_FULL)


@pytest.fixture
def client(monkeypatch, patch_models, kit_path, kit_key, full):
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    with TestClient(app_module.create_app()) as c:
        yield c


def settings_of(client) -> dict:
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    return resp.json()["settings"]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_settings_block_shape(client):
    block = settings_of(client)
    assert set(block) == {"level", "controls", "endpoints", "diagnostics"}
    assert block["level"] == "full"
    assert block["controls"] == {
        "model": True, "temperature": True, "simplified_view": True, "live_graph": True,
    }
    assert block["endpoints"]["label"]
    assert block["diagnostics"]["label"]


def test_controls_follow_the_live_graph_flag(monkeypatch, patch_models, kit_path, kit_key, full):
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: False)
    with TestClient(app_module.create_app()) as c:
        body = c.get("/api/meta").json()
    assert body["live_graph"] is False
    assert body["settings"]["controls"]["live_graph"] is False


def test_endpoint_rows_describe_what_this_build_uses(client, openrouter_key):
    # NOTE: this assertion used to pin "every row is provider kit" and exactly
    # the roles {chat, embedding}. That became false when this branch started
    # offering OpenRouter models in the picker — the build really does talk to
    # a second endpoint, so pinning one provider here would assert the demo is
    # something it is not. The contract every branch shares (uniform key set,
    # the KIT rows, no secrets) is what stays here; the OpenRouter row is
    # demo-v2-int's own and is pinned in test_meta_settings_openrouter.py.
    rows = settings_of(client)["endpoints"]["rows"]
    for row in rows:
        assert set(row) == ROW_KEYS
        assert row["label"]
        assert row["detail"]
        assert row["status"] in {"ok", "no_key", "unreachable", "unknown"}

    kit_rows = {row["role"]: row for row in rows if row["provider"] == "kit"}
    assert set(kit_rows) == {"chat", "embedding"}
    for row in kit_rows.values():
        assert row["base_url"] == cfg.load_config()["kit"]["base_url"]
        assert row["api_key"] == {"configured": True, "hint": "KIT_API_KEY"}
        assert row["status"] == "ok"
    # The chat model is whatever the picker holds; the embedding model is fixed.
    assert kit_rows["chat"]["model"] is None
    assert kit_rows["chat"]["model_source"]
    assert kit_rows["embedding"]["model"] == cfg.load_config()["kit"]["embedding_model"]
    # Nothing to warn about while every key is set and the catalog is live.
    assert settings_of(client)["endpoints"]["notices"] == []


def test_diagnostics_are_short_label_value_rows(client):
    rows = settings_of(client)["diagnostics"]["rows"]
    assert rows
    labels = [row["label"] for row in rows]
    assert labels == sorted(set(labels), key=labels.index)  # no duplicates
    for row in rows:
        assert set(row) == {"label", "value", "detail"}
        assert isinstance(row["label"], str) and row["label"]
        assert isinstance(row["value"], str) and row["value"]
        # A value is a caption, not a config dump, and never a host path.
        assert len(row["value"]) <= 60
        assert "/" not in row["value"] or row["value"].startswith("http")
    assert "config.toml" in [row["value"] for row in rows]


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def test_no_part_of_the_api_key_is_served(client, kit_key):
    raw = client.get("/api/meta").text
    assert kit_key not in raw
    # Not even a masked tail or head of it.
    assert kit_key[-8:] not in raw
    assert kit_key[:12] not in raw
    assert "secret" not in raw.lower()


def test_api_key_state_reports_presence_only(monkeypatch):
    monkeypatch.setenv("SOME_PROVIDER_KEY", SECRET)
    state = meta.api_key_state("SOME_PROVIDER_KEY")
    assert state == {"configured": True, "hint": "SOME_PROVIDER_KEY"}

    monkeypatch.setenv("SOME_PROVIDER_KEY", "   ")
    assert meta.api_key_state("SOME_PROVIDER_KEY")["configured"] is False
    monkeypatch.delenv("SOME_PROVIDER_KEY")
    assert meta.api_key_state("SOME_PROVIDER_KEY") == {
        "configured": False, "hint": "SOME_PROVIDER_KEY",
    }
    # No env var at all (a local server): nothing to report.
    assert meta.api_key_state(None) == {"configured": False, "hint": None}


# ---------------------------------------------------------------------------
# Degraded states
# ---------------------------------------------------------------------------

def test_missing_key_marks_the_row_and_raises_a_notice(
    monkeypatch, patch_models, kit_path, full, openrouter_key,
):
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    with TestClient(app_module.create_app()) as c:
        section = c.get("/api/meta").json()["settings"]["endpoints"]
    chat = section["rows"][0]
    assert chat["status"] == "no_key"
    assert chat["api_key"] == {"configured": False, "hint": "KIT_API_KEY"}
    assert "KIT_API_KEY" in chat["detail"]
    assert section["notices"]
    assert all("no API key" in notice for notice in section["notices"])


def test_offline_catalog_marks_the_chat_row_unreachable(
    monkeypatch, kit_path, kit_key, full, openrouter_key,
):
    def boom(provider, **_kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(meta, "fetch_provider_models_meta", boom)
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    meta.reset_caches()
    try:
        with TestClient(app_module.create_app()) as c:
            section = c.get("/api/meta").json()["settings"]["endpoints"]
    finally:
        meta.reset_caches()
    chat = section["rows"][0]
    assert chat["role"] == "chat"
    assert chat["status"] == "unreachable"
    assert "fallback" in chat["detail"]
    assert section["notices"] == [f"{chat['label']}: not reachable right now."]


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------

def test_minimal_keeps_the_controls_and_drops_the_rest(monkeypatch, patch_models, kit_path, kit_key):
    monkeypatch.setattr(meta, "get_frontend_settings_level", lambda: meta.LEVEL_MINIMAL)
    monkeypatch.setattr(meta, "get_live_graph_enabled", lambda: True)
    with TestClient(app_module.create_app()) as c:
        body = c.get("/api/meta").json()
    block = body["settings"]
    assert block["level"] == "minimal"
    assert block["endpoints"] is None
    assert block["diagnostics"] is None
    assert block["controls"]["model"] is True
    # The rest of /meta is untouched: a minimal build is still a full demo.
    assert body["models"] and body["default_model"] and body["live_graph"] is True


def test_empty_seams_hide_their_sections(monkeypatch, patch_models, kit_path, kit_key, full):
    monkeypatch.setattr(meta, "endpoint_rows", list)
    monkeypatch.setattr(meta, "diagnostic_rows", list)
    block = meta.settings_meta()
    assert block["level"] == "full"
    assert block["endpoints"] is None and block["diagnostics"] is None


# ---------------------------------------------------------------------------
# Branch seams
# ---------------------------------------------------------------------------

def test_rows_a_branch_adds_are_served_untouched(monkeypatch, patch_models, kit_path, kit_key, full):
    """What demo-booth / demo-llamacpp do: append rows in ``endpoint_rows``.

    Unknown providers, unknown statuses and extra keys must survive the trip
    (the React panel renders anything it does not know neutrally).
    """
    local = meta.endpoint_row(
        "llamacpp-chat", "llama-server (chat)", role="chat", provider="llamacpp",
        base_url="http://host.docker.internal:8080/v1", model="local-model",
        model_source="whatever the server has loaded", api_key=None,
        status=meta.STATUS_UNREACHABLE, detail="Start llama-server on :8080.",
    )
    exotic = {**meta.endpoint_row("acme-1", "ACME Cloud", role="reranking",
                                  provider="acme", status="degraded"),
              "region": "eu-central-1"}
    monkeypatch.setattr(meta, "endpoint_rows", lambda: [local, exotic])
    monkeypatch.setattr(
        meta, "diagnostic_rows",
        lambda: [meta.diagnostic_row("GPU", "RTX 5070 Ti", detail="16 GB VRAM")],
    )
    with TestClient(app_module.create_app()) as c:
        block = c.get("/api/meta").json()["settings"]

    assert block["endpoints"]["rows"] == [local, exotic]
    assert block["endpoints"]["rows"][1]["region"] == "eu-central-1"
    # Only statuses the default notice rule knows about produce a line.
    assert block["endpoints"]["notices"] == ["llama-server (chat): not reachable right now."]
    assert block["diagnostics"]["rows"] == [
        {"label": "GPU", "value": "RTX 5070 Ti", "detail": "16 GB VRAM"},
    ]


def test_provider_endpoint_row_builds_a_row_from_config(monkeypatch, kit_key):
    row = meta.provider_endpoint_row("kit", role="chat")
    assert row["id"] == "kit-chat"
    assert row["label"] == meta.provider_label("kit")
    assert row["base_url"] == cfg.load_config()["kit"]["base_url"]
    assert row["status"] == "ok"

    # A provider with no key env var at all: nothing to report, no guess.
    local = meta.provider_endpoint_row(
        "llamacpp", role="chat", key_env=None, status=meta.STATUS_OK,
    )
    assert local["api_key"] is None
    assert local["base_url"] == cfg.load_config()["llamacpp"]["base_url"]
    assert local["status"] == "ok"

    # An unknown provider degrades to its config key as the label.
    unknown = meta.provider_endpoint_row("acme", role="chat", key_env=None)
    assert unknown["label"] == "acme"
    assert unknown["base_url"] is None
    assert unknown["status"] == "unknown"


# ---------------------------------------------------------------------------
# The other picker shape
# ---------------------------------------------------------------------------

def test_settings_block_also_serves_the_provider_aware_picker(provider_harness):
    """Runs on demo-booth / demo-llamacpp, where chat_controls is
    provider-aware. The level comes from that branch's own config, so this
    only pins the shape."""
    block = provider_harness.client.get("/api/meta").json()["settings"]
    assert block["level"] in cfg.FRONTEND_SETTINGS_LEVELS
    assert set(block["controls"]) == {"model", "temperature", "simplified_view", "live_graph"}
    for section in ("endpoints", "diagnostics"):
        if block[section] is not None:
            assert block[section]["rows"]

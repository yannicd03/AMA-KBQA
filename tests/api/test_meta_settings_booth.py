"""demo-booth's own settings-panel content: the billed chat endpoints.

Kept next to ``test_meta_settings.py`` rather than inside it. That file pins
the *contract* all three demo branches implement and is merged forward from
demo-v2-int; this one holds what only the booth build does, so the two do not
collide on every merge.

The booth picker offers KIT (free) next to OpenRouter and DeepSeek, which bill
the booth account for real. So the panel has to answer two questions between
demos, without a network round-trip and without ever serving key material:
which chat endpoints can this laptop reach, and which of them cost money.
"""

from __future__ import annotations

import pytest

import ama_kbqa.config as cfg
from ama_kbqa.api import meta
from ama_kbqa.frontend.utils import chat_controls

BILLED = "billed to the booth account"
KIT_SECRET = "sk-kit-0123456789abcdef-super-secret"


@pytest.fixture
def kit_key(monkeypatch):
    """KIT reachable, so the free half of the picker is never the story here."""
    monkeypatch.setenv("KIT_API_KEY", KIT_SECRET)
    return KIT_SECRET


@pytest.fixture
def offline_catalogs(monkeypatch):
    """Keep the picker off the network.

    ``tests/frontend`` has an autouse fixture for this; ``tests/api`` does not,
    and with real keys in .env a call to ``available_choices()`` would be a
    live HTTP fetch: slow, and order-dependent through ``st.cache_data``.
    """
    def unreachable(*_args, **_kwargs):
        raise RuntimeError("model catalog fetch disabled in tests")

    monkeypatch.setattr(chat_controls, "_fetch_kit_models_meta", unreachable)
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", unreachable)


def chat_model_entry(provider: str, model_id: str) -> dict:
    """One normalised ``[[frontend.chat_models]]`` entry."""
    return {
        "provider": provider, "id": model_id, "name": "",
        "prompt_usd_per_m": None, "completion_usd_per_m": None, "default": False,
    }


@pytest.fixture
def rows(patch_models, kit_key):
    """``endpoint_rows()`` against the branch's real config, no network."""
    return meta.endpoint_rows


# ---------------------------------------------------------------------------
# The rows follow config, not a hard-coded list
# ---------------------------------------------------------------------------

def test_billed_rows_come_from_the_configured_chat_models(monkeypatch, rows, booth_keys):
    """Drop a provider from ``[[frontend.chat_models]]`` and its row goes too."""
    monkeypatch.setattr(meta, "get_frontend_chat_models",
                        lambda: [chat_model_entry("deepseek", "deepseek-flash")])
    assert [row["id"] for row in rows()] == ["kit-chat", "kit-embedding", "deepseek-chat"]

    # A KIT-only build: nothing billed to announce, so no extra rows at all.
    monkeypatch.setattr(meta, "get_frontend_chat_models", list)
    assert [row["id"] for row in rows()] == ["kit-chat", "kit-embedding"]
    assert meta.optional_chat_providers() == []


def test_this_branch_offers_openrouter_and_deepseek(rows, booth_keys):
    """The shipped config.toml, read through the same seam."""
    assert meta.optional_chat_providers() == ["openrouter", "deepseek"]
    by_id = {row["id"]: row for row in rows()}
    for row_id, provider in (("openrouter-chat", "openrouter"),
                             ("deepseek-chat", "deepseek")):
        row = by_id[row_id]
        assert row["provider"] == provider
        assert row["role"] == "chat"
        assert row["base_url"] == cfg.load_config()[provider]["base_url"]
        # One row per provider, so the model is the picker's choice, and the
        # row says how many of that provider's models the picker offers.
        assert row["model"] is None
        assert "config.toml" in row["model_source"]


def test_the_configured_chat_provider_is_not_listed_twice(monkeypatch, rows, booth_keys):
    """A build whose own chat_provider is OpenRouter keeps one OpenRouter row."""
    config = cfg.load_config()
    monkeypatch.setattr(meta, "load_config",
                        lambda: {**config, "llm": {**config["llm"],
                                                   "chat_provider": "openrouter"}})
    ids = [row["id"] for row in rows()]
    assert ids.count("openrouter-chat") == 1


# ---------------------------------------------------------------------------
# Reachability follows the API keys
# ---------------------------------------------------------------------------

def test_a_billed_row_is_ok_with_its_key_and_no_key_without_it(
    monkeypatch, rows, booth_keys,
):
    """The row stays visible in both states; the status and detail change.

    Hiding it would answer the presenter's question ("can I use DeepSeek?")
    with silence, which reads the same as "this build has no DeepSeek".
    """
    by_id = {row["id"]: row for row in rows()}
    for row_id, env_var in (("openrouter-chat", "OPENROUTER_API_KEY"),
                            ("deepseek-chat", "DEEPSEEK_API_KEY")):
        row = by_id[row_id]
        assert row["status"] == "ok"
        assert row["api_key"] == {"configured": True, "hint": env_var}
        assert BILLED in row["detail"]

    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    by_id = {row["id"]: row for row in rows()}
    for row_id, env_var in (("openrouter-chat", "OPENROUTER_API_KEY"),
                            ("deepseek-chat", "DEEPSEEK_API_KEY")):
        row = by_id[row_id]
        assert row["status"] == "no_key"
        assert row["api_key"] == {"configured": False, "hint": env_var}
        # Still unmistakably billed, and it says how to switch it on.
        assert BILLED in row["detail"]
        assert env_var in row["detail"]
    # The free KIT pair is untouched by any of this.
    assert by_id["kit-chat"]["status"] == "ok"
    assert "free" in by_id["kit-chat"]["detail"].lower()


def test_no_endpoint_is_probed_over_http(monkeypatch, rows, booth_keys):
    """Reachability is read off the environment, never off the network.

    ``patch_models`` already stands in for the KIT catalog fetch; nothing may
    reach for the billed providers on top of it, or /api/meta would put two
    round-trips in front of the demo's first screen. So the rows must come out
    of config and the environment without going through the picker at all.
    """
    def boom(*_args, **_kwargs):
        raise AssertionError("endpoint_rows() must not hit the network")

    for name in ("fetch_provider_models_meta", "fetch_provider_models_pricing",
                 "_fetch_kit_models_meta", "_fetch_openrouter_catalog"):
        monkeypatch.setattr(chat_controls, name, boom, raising=False)
    assert len(rows()) == 4


# ---------------------------------------------------------------------------
# Notices: one per provider, never the same sentence twice
# ---------------------------------------------------------------------------

def test_one_notice_per_provider_not_one_per_model(rows, no_booth_keys):
    """config.toml lists five OpenRouter models and two DeepSeek ones."""
    notices = meta.endpoint_notices(rows())
    assert len(notices) == len(meta.optional_chat_providers()) == 2
    assert sum("OPENROUTER_API_KEY" in notice for notice in notices) == 1
    assert sum("DEEPSEEK_API_KEY" in notice for notice in notices) == 1
    for notice in notices:
        assert "no API key" in notice
        assert "hidden from the picker" in notice


def test_notices_say_the_same_thing_the_picker_says(rows, no_booth_keys, offline_catalogs):
    """The two sections must not contradict each other on one screen."""
    model_notices = chat_controls.available_choices()[1]
    assert model_notices == [
        "OpenRouter models hidden: OPENROUTER_API_KEY not set",
        "DeepSeek models hidden: DEEPSEEK_API_KEY not set",
    ]
    endpoint = meta.endpoint_notices(rows())
    for env_var in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        assert any(env_var in notice for notice in model_notices)
        assert any(env_var in notice for notice in endpoint)
    # Consistent, but not the identical sentence in two panels.
    assert not set(endpoint) & set(model_notices)


def test_a_line_the_model_picker_already_serves_is_dropped(monkeypatch, rows, no_booth_keys):
    """model_notices is rendered above the endpoint list, so a byte-identical
    line there must not be repeated here."""
    endpoint_list = rows()
    notices = meta.endpoint_notices(endpoint_list)
    assert len(notices) == 2
    already_shown = tuple(notices[:1])

    class FakeCatalog:
        notices = already_shown
        live = True

    monkeypatch.setattr(meta, "get_catalog", FakeCatalog)
    assert meta.endpoint_notices(endpoint_list) == notices[1:]


def test_two_rows_with_the_same_complaint_produce_one_line(patch_models, kit_key):
    row = meta.endpoint_row(
        "openrouter-chat", "OpenRouter", role="chat", provider="openrouter",
        api_key={"configured": False, "hint": "OPENROUTER_API_KEY"},
        status=meta.STATUS_NO_KEY,
    )
    twin = {**row, "id": "openrouter-chat-2"}
    assert meta.endpoint_notices([row, twin]) == meta.endpoint_notices([row])
    assert len(meta.endpoint_notices([row, twin])) == 1


# ---------------------------------------------------------------------------
# Build facts
# ---------------------------------------------------------------------------

def test_diagnostics_count_the_selectable_chat_providers(patch_models, kit_key, booth_keys):
    facts = {row["label"]: row for row in meta.diagnostic_rows()}
    assert facts["Chat providers"]["value"] == "3 of 3 selectable"
    # Embeddings do not follow the chat pick: apply_chat_settings only ever
    # rewrites the chat provider and model.
    assert facts["Embeddings"]["value"].startswith(meta.provider_label("kit"))


def test_diagnostics_follow_the_keys_actually_present(patch_models, kit_key, no_booth_keys):
    facts = {row["label"]: row for row in meta.diagnostic_rows()}
    assert facts["Chat providers"]["value"] == "1 of 3 selectable"
    assert "free" in facts["Chat providers"]["detail"]


def test_build_facts_stay_short_and_leak_nothing(patch_models, kit_key, booth_keys):
    for row in meta.diagnostic_rows():
        assert set(row) == {"label", "value", "detail"}
        assert row["value"] and len(row["value"]) <= 60
        # Never a config dump, a secret, or a path out of the user's home.
        assert "/home/" not in str(row)
        for secret in booth_keys:
            assert secret not in str(row)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------

def test_meta_and_the_picker_agree_on_the_key_env_vars():
    """Two modules, one rule for "is this endpoint offered".

    If they drift, the panel reports a provider as reachable that the picker
    hides (or the reverse), which is exactly the confusion the panel exists to
    prevent.
    """
    for provider in meta.optional_chat_providers():
        assert meta.PROVIDER_KEY_ENV[provider] == chat_controls._PROVIDER_KEY_ENV[provider]


def test_every_offered_provider_has_a_base_url_and_a_label():
    config = cfg.load_config()
    for provider in meta.optional_chat_providers():
        assert config.get(provider, {}).get("base_url")
        assert meta.provider_label(provider) != provider

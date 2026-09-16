"""The provider-aware model picker path (branches demo-booth, demo-llamacpp).

``chat_controls`` there exposes ``available_choices`` / ``default_choice`` /
``display_choice`` / ``price_caption_for`` and a provider-aware
``apply_chat_settings``. These tests install fakes of that API (conftest
``fake_choices``), so they run the same on every branch.
"""

from __future__ import annotations

from ama_kbqa.api import meta, runs
from ama_kbqa.frontend.utils import chat_controls
from ama_kbqa.pricing import estimate_cost_usd, format_cost_usd

from .conftest import MODEL, NOTICES, read_sse

KIT_KEY = f"kit:{MODEL}"
OR_KEY = "openrouter:deepseek-v4-pro"
DS_KEY = "deepseek:deepseek-v4-pro"
LOCAL_KEY = "llamacpp:local-model"


def test_detection_follows_the_module(fake_choices):
    assert meta.provider_aware() is True


def test_kit_only_detection(kit_path):
    assert meta.provider_aware() is False


def test_meta_lists_one_entry_per_choice(provider_harness):
    body = provider_harness.client.get("/api/meta").json()
    assert body["models"] == [
        {"id": KIT_KEY, "name": "GPT-OSS 120B · kit",
         "price": "$0.10 per 1M in and $0.30 per 1M out", "provider": "kit"},
        {"id": OR_KEY, "name": "DeepSeek V4 Pro · openrouter",
         "price": "$1.00 per 1M in and $2.00 per 1M out (billed)", "provider": "openrouter"},
        {"id": DS_KEY, "name": "DeepSeek V4 Pro · deepseek",
         "price": None, "provider": "deepseek"},
        {"id": LOCAL_KEY, "name": "local-model · llamacpp",
         "price": "Runs locally: no API cost", "provider": "llamacpp"},
    ]
    # default_choice() decides (here: the entry flagged default in config).
    assert body["default_model"] == DS_KEY
    assert body["model_notices"] == NOTICES


def test_choices_are_cached_for_their_ttl(fake_choices, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(meta, "_clock", lambda: now["t"])
    first = meta.get_catalog()
    assert meta.get_catalog() is first
    assert fake_choices.calls == 1
    assert first.resolve(OR_KEY).provider == "openrouter"
    assert first.resolve(OR_KEY).model == "deepseek-v4-pro"

    now["t"] += meta.CHOICES_TTL_SECONDS + 1
    meta.get_catalog()
    assert fake_choices.calls == 2


def test_listing_failure_keeps_the_default_and_says_so(fake_choices):
    fake_choices.error = RuntimeError("bad config")
    catalog = meta.get_catalog()
    assert catalog.keys() == ["kit:kit.placeholder"]
    assert catalog.default_key == "kit:kit.placeholder"
    assert catalog.notices[0].startswith("Could not list the chat models (RuntimeError)")


def test_run_gets_provider_and_bare_model_id(provider_harness):
    run_id, frames = provider_harness.run(model=OR_KEY)
    assert provider_harness.applied == [("deepseek-v4-pro", 1.0, "openrouter")]

    done = frames[-1]["data"]
    assert done["status"] == "done"
    assert done["model"] == "deepseek-v4-pro"
    assert done["model_key"] == OR_KEY
    assert done["provider"] == "openrouter"

    record = provider_harness.client.get(f"/api/runs/{run_id}").json()
    assert (record["model"], record["model_key"], record["provider"]) == (
        "deepseek-v4-pro", OR_KEY, "openrouter",
    )


def test_only_current_keys_are_accepted(provider_harness):
    # Bare ids (even ones that exist behind a key) and unknown keys are 400.
    for bad in ("deepseek-v4-pro", MODEL, "openrouter:unknown", "anthropic:deepseek-v4-pro"):
        resp = provider_harness.post(model=bad)
        assert resp.status_code == 400, bad
        assert resp.json() == {"detail": f"Unknown model: {bad}"}
    assert provider_harness.created == []
    assert provider_harness.applied == []


def test_followup_is_keyed_by_the_full_model_key(provider_harness):
    h = provider_harness

    def ask(model):
        resp = h.post(model=model)
        assert resp.status_code == 201, resp.text
        assert read_sse(h.client, resp.json()["run_id"])[-1]["event"] == "done"
        return resp.json()["continuation"]

    assert ask(OR_KEY) is False
    assert ask(OR_KEY) is True
    # Same bare model id behind another provider: a fresh agent.
    assert ask(DS_KEY) is False
    assert ask(DS_KEY) is True
    assert len(h.created) == 2
    assert [provider for _m, _t, provider in h.applied] == [
        "openrouter", "openrouter", "deepseek", "deepseek",
    ]


def test_cost_uses_the_bare_model_id(provider_harness):
    _run_id, frames = provider_harness.run(model=KIT_KEY)
    done = frames[-1]["data"]
    assert done["model"] == MODEL
    assert done["cost"] == format_cost_usd(estimate_cost_usd(MODEL, 100, 20))
    assert done["cost"]


def test_apply_model_settings_matches_the_signature(monkeypatch):
    option = meta.ModelOption(key=OR_KEY, provider="openrouter",
                              model="deepseek-v4-pro", name="x", price=None)
    calls: list[tuple] = []

    monkeypatch.setattr(chat_controls, "apply_chat_settings",
                        lambda model, temperature: calls.append((model, temperature)))
    runs.apply_model_settings(option, 0.5)

    def aware(model, temperature, provider="kit"):
        calls.append((model, temperature, provider))

    monkeypatch.setattr(chat_controls, "apply_chat_settings", aware)
    runs.apply_model_settings(option, 0.7)
    assert calls == [("deepseek-v4-pro", 0.5), ("deepseek-v4-pro", 0.7, "openrouter")]

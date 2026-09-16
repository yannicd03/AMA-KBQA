"""The provider-aware half of chat_controls: ``available_choices`` and friends.

The KIT half (auto-discovery, filtering, the offline fallback) is covered by
``test_chat_controls.py`` and must keep behaving exactly as it did — these
additions are meant to be purely additive.

What is pinned here: OpenRouter entries come from ``[[frontend.chat_models]]``
and are shown only when the key is set and the live catalog still knows them;
the free-text entry follows the same key rule; and ``apply_chat_settings``
exports the *provider* as well as the model, which is what carries the pick
across the MCP subprocess boundary.
"""

from __future__ import annotations

import os

import pytest

import ama_kbqa.config as cfg_module
from ama_kbqa import pricing
from ama_kbqa.frontend.utils import chat_controls

KIT_MODELS = ["kit.mistral-small-4-119b-a8b", "kit.glm-5.3"]

PRO = "deepseek/deepseek-v4-pro"
FLASH = "deepseek/deepseek-v4.1-flash"

CONFIG_ENTRIES = [
    {"provider": "openrouter", "id": PRO, "name": "DeepSeek V4 Pro",
     "prompt_usd_per_m": None, "completion_usd_per_m": None, "default": False},
    {"provider": "openrouter", "id": FLASH, "name": "DeepSeek V4.1 Flash",
     "prompt_usd_per_m": None, "completion_usd_per_m": None, "default": False},
]

CATALOG = {
    PRO: {"name": "DeepSeek V4 Pro", "prompt": 1.0e-6,
          "completion": 3.0e-6, "supports_tools": True},
    FLASH: {"name": "DeepSeek V4.1 Flash", "prompt": 1.0e-7,
            "completion": 3.0e-7, "supports_tools": True},
}


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    """No network, no ambient key, no leaked runtime prices."""
    monkeypatch.setattr(chat_controls, "available_models", lambda: list(KIT_MODELS))
    monkeypatch.setattr(chat_controls, "get_frontend_chat_models",
                        lambda: [dict(e) for e in CONFIG_ENTRIES])
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog",
                        lambda: {k: dict(v) for k, v in CATALOG.items()})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    saved = dict(pricing._RUNTIME_PRICING)
    yield
    pricing._RUNTIME_PRICING.clear()
    pricing._RUNTIME_PRICING.update(saved)


def keys(choices) -> list[str]:
    return [c.key for c in choices]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def test_kit_first_then_presets_then_the_custom_entry():
    choices, notices = chat_controls.available_choices()
    assert keys(choices) == [
        "kit:kit.mistral-small-4-119b-a8b",
        "kit:kit.glm-5.3",
        f"openrouter:{PRO}",
        f"openrouter:{FLASH}",
        chat_controls.CUSTOM_CHOICE_KEY,
    ]
    assert notices == []
    # Every KIT model the picker offered before is still offered.
    assert [c.model for c in choices if c.provider == "kit"] == KIT_MODELS


def test_the_two_shipped_presets_are_deepseek_on_openrouter():
    choices, _ = chat_controls.available_choices()
    presets = [c for c in choices if c.provider == "openrouter" and not c.custom]
    assert [(c.model, c.name) for c in presets] == [
        (PRO, "DeepSeek V4 Pro"),
        (FLASH, "DeepSeek V4.1 Flash"),
    ]
    # Routed through OpenRouter, never a direct [deepseek] provider.
    assert all(c.provider == "openrouter" for c in presets)


def test_the_custom_entry_is_a_placeholder_not_a_model():
    choices, _ = chat_controls.available_choices()
    custom = choices[-1]
    assert custom.custom is True
    assert custom.model == chat_controls.CUSTOM_MODEL_SENTINEL
    assert custom.key == "openrouter:__custom__"
    # A sentinel that can never collide with a real OpenRouter id.
    assert "/" not in custom.model


# ---------------------------------------------------------------------------
# The key gate
# ---------------------------------------------------------------------------

def test_without_the_key_every_openrouter_entry_is_hidden_once(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    choices, notices = chat_controls.available_choices()

    assert keys(choices) == ["kit:kit.mistral-small-4-119b-a8b", "kit:kit.glm-5.3"]
    # One notice for the provider, not one per configured model.
    assert notices == ["OpenRouter models hidden: OPENROUTER_API_KEY not set"]


def test_the_custom_entry_follows_the_same_key_rule(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    choices, _ = chat_controls.available_choices()
    assert all(not c.custom for c in choices)


# ---------------------------------------------------------------------------
# Live-catalog validation
# ---------------------------------------------------------------------------

def test_an_id_the_catalog_does_not_know_is_hidden_with_a_reason(monkeypatch):
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog",
                        lambda: {PRO: dict(CATALOG[PRO])})
    choices, notices = chat_controls.available_choices()
    assert f"openrouter:{FLASH}" not in keys(choices)
    assert notices == [f"OpenRouter model {FLASH} is not in the live catalog and was hidden"]


def test_a_model_that_cannot_call_tools_is_hidden(monkeypatch):
    catalog = {k: dict(v) for k, v in CATALOG.items()}
    catalog[FLASH]["supports_tools"] = False
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", lambda: catalog)
    choices, notices = chat_controls.available_choices()
    # Every agent here answers by calling tools; such a model would fail on
    # the first question.
    assert f"openrouter:{FLASH}" not in keys(choices)
    assert notices == [f"OpenRouter model {FLASH} cannot call tools and was hidden"]


def test_an_unreachable_catalog_keeps_the_entries_unvalidated(monkeypatch):
    def boom():
        raise RuntimeError("openrouter down")

    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", boom)
    choices, notices = chat_controls.available_choices()
    # A flaky network must not empty the picker.
    assert f"openrouter:{PRO}" in keys(choices)
    assert f"openrouter:{FLASH}" in keys(choices)
    assert notices == []
    # Unpriced rather than wrongly priced.
    assert all(c.prompt_usd_per_token is None
               for c in choices if c.provider == "openrouter" and not c.custom)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def test_catalog_prices_are_used_and_registered_for_the_cost_footer():
    choices, _ = chat_controls.available_choices()
    pro = next(c for c in choices if c.model == PRO)
    assert pro.prompt_usd_per_token == pytest.approx(1.0e-6)
    assert pro.completion_usd_per_token == pytest.approx(3.0e-6)
    # Registered, so the per-answer estimate works for a billed model whose id
    # is not in the shipped KIT pricing table.
    assert pricing.estimate_cost_usd(PRO, 1_000_000, 0) == pytest.approx(1.0)


def test_a_config_price_overrides_the_catalog(monkeypatch):
    entries = [dict(CONFIG_ENTRIES[0], prompt_usd_per_m=2.0, completion_usd_per_m=8.0)]
    monkeypatch.setattr(chat_controls, "get_frontend_chat_models", lambda: entries)
    choices, _ = chat_controls.available_choices()
    pro = next(c for c in choices if c.model == PRO)
    # Config quotes per 1M tokens; everything else works per token.
    assert pro.prompt_usd_per_token == pytest.approx(2.0e-6)
    assert pro.completion_usd_per_token == pytest.approx(8.0e-6)


def test_price_caption_marks_billed_models_and_skips_the_custom_entry():
    choices, _ = chat_controls.available_choices()
    pro = next(c for c in choices if c.model == PRO)
    caption = chat_controls.price_caption_for(pro)
    assert "(billed)" in caption
    assert r"\$" in caption  # escaped for Streamlit markdown

    # KIT keeps the illustrative table and never says "billed".
    kit = next(c for c in choices if c.provider == "kit")
    assert "(billed)" not in (chat_controls.price_caption_for(kit) or "")

    # Nothing is known about an id nobody has typed yet.
    assert chat_controls.price_caption_for(choices[-1]) is None


# ---------------------------------------------------------------------------
# Labels and the default
# ---------------------------------------------------------------------------

def test_display_choice_names_the_endpoint():
    choices, _ = chat_controls.available_choices()
    pro = next(c for c in choices if c.model == PRO)
    assert chat_controls.display_choice(pro) == "DeepSeek V4 Pro · OpenRouter"
    # The custom entry already reads as its own label.
    assert chat_controls.display_choice(choices[-1]) == "OpenRouter (custom)"


def test_default_stays_on_the_free_kit_model():
    choices, _ = chat_controls.available_choices()
    default = chat_controls.default_choice(choices)
    # The demo opens on a fast, free model and only spends money on request.
    assert default.provider == "kit"
    assert default.model == "kit.mistral-small-4-119b-a8b"


def test_the_custom_placeholder_is_never_chosen_as_a_fallback():
    only_custom = [chat_controls.custom_choice()]
    default = chat_controls.default_choice(only_custom)
    # It carries no model id, so a build opening on it could answer nothing.
    assert default.custom is False
    assert default.model == chat_controls.DEFAULT_MODEL_PREFERENCE[0]


def test_an_explicit_config_default_still_wins(monkeypatch):
    entries = [dict(CONFIG_ENTRIES[0], default=True)]
    monkeypatch.setattr(chat_controls, "get_frontend_chat_models", lambda: entries)
    choices, _ = chat_controls.available_choices()
    assert chat_controls.default_choice(choices).model == PRO


# ---------------------------------------------------------------------------
# apply_chat_settings: the subprocess channel
# ---------------------------------------------------------------------------

@pytest.fixture
def fixed_config(monkeypatch):
    base = {"llm": {"chat_provider": "kit", "chat_temperature": 1.0},
            "kit": {"chat_model": "old"}, "openrouter": {"chat_model": "old"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    monkeypatch.setattr(cfg_module, "load_config", lambda: cfg_module._config_cache)
    for var in (cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR,
                cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR,
                cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR):
        monkeypatch.delenv(var, raising=False)
    return base


def test_a_non_kit_pick_exports_the_provider_for_the_mcp_subprocesses(fixed_config):
    """The trap this guards: MCP tool servers are spawned with
    ``env=os.environ.copy()`` and load their *own* config.toml, so mutating
    this process's config alone would leave every specialist on KIT while the
    parent talked to OpenRouter — and the question would still answer."""
    chat_controls.apply_chat_settings(PRO, 0.5, provider="openrouter")

    assert os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] == "openrouter"
    assert os.environ[cfg_module.CHAT_MODEL_OVERRIDE_ENV_VAR] == PRO
    assert os.environ[cfg_module.CHAT_TEMPERATURE_OVERRIDE_ENV_VAR] == "0.5"

    # What a freshly-spawned subprocess would actually resolve.
    assert cfg_module.get_chat_provider() == "openrouter"
    assert cfg_module.get_chat_model_name() == PRO

    # And the in-memory config of this process agrees.
    assert fixed_config["llm"]["chat_provider"] == "openrouter"
    assert fixed_config["openrouter"]["chat_model"] == PRO


def test_the_default_still_pins_kit_for_two_argument_callers(fixed_config):
    # The Streamlit page and the smoke script call with two arguments.
    chat_controls.apply_chat_settings("kit.glm-5.3", 1.0)
    assert os.environ[cfg_module.CHAT_PROVIDER_OVERRIDE_ENV_VAR] == "kit"
    assert cfg_module.get_chat_provider() == "kit"
    assert fixed_config["llm"]["chat_provider"] == "kit"
    assert fixed_config["kit"]["chat_model"] == "kit.glm-5.3"


def test_switching_back_to_kit_clears_the_openrouter_pick(fixed_config):
    chat_controls.apply_chat_settings(PRO, 1.0, provider="openrouter")
    chat_controls.apply_chat_settings("kit.glm-5.3", 1.0, provider="kit")
    # No stale override left pointing the subprocesses at OpenRouter.
    assert cfg_module.get_chat_provider() == "kit"
    assert cfg_module.get_chat_model_name() == "kit.glm-5.3"

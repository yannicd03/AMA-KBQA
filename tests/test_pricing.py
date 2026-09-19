"""Tests for the illustrative cost-estimation helpers in ama_kbqa.pricing."""

from __future__ import annotations

import json

import pytest

from ama_kbqa import pricing


@pytest.fixture(autouse=True)
def _clear_cache():
    """The pricing table is lru_cached; reset it around each test."""
    pricing._load.cache_clear()
    yield
    pricing._load.cache_clear()


def test_known_model_priced_from_data_file():
    """A model present in the shipped data file gets a positive cost."""
    table = pricing._load()
    priced = [
        m for m, e in table.get("models", {}).items()
        if e.get("prompt_usd_per_token") is not None
    ]
    assert priced, "expected at least one priced model in model_pricing.json"
    model = priced[0]
    entry = table["models"][model]
    cost = pricing.estimate_cost_usd(model, 1000, 500)
    expected = 1000 * entry["prompt_usd_per_token"] + 500 * entry["completion_usd_per_token"]
    assert cost == pytest.approx(expected)
    assert cost > 0


def test_unknown_model_returns_none():
    assert pricing.estimate_cost_usd("kit.does-not-exist", 1000, 1000) is None


def test_none_model_returns_none():
    assert pricing.estimate_cost_usd(None, 1000, 1000) is None
    assert pricing.get_model_pricing(None) is None


def test_estimate_uses_separate_prompt_and_completion_rates(tmp_path, monkeypatch):
    """Prompt and completion tokens are billed at their own rates."""
    data = {
        "models": {
            "test.model": {
                "openrouter_id": "vendor/test",
                "prompt_usd_per_token": 1e-6,
                "completion_usd_per_token": 2e-6,
                "matched": True,
            }
        }
    }
    path = tmp_path / "model_pricing.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(pricing, "_PRICING_PATH", path)
    pricing._load.cache_clear()

    # 1000 * 1e-6 + 500 * 2e-6 = 0.001 + 0.001 = 0.002
    assert pricing.estimate_cost_usd("test.model", 1000, 500) == pytest.approx(0.002)


def test_missing_data_file_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "_PRICING_PATH", tmp_path / "nope.json")
    pricing._load.cache_clear()
    assert pricing._load() == {"models": {}}
    assert pricing.estimate_cost_usd("anything", 1, 1) is None


class TestRuntimePricingRegistry:
    """The booth picker registers real prices for OpenRouter/DeepSeek models,
    whose ids are absent from the shipped KIT table."""

    @pytest.fixture(autouse=True)
    def _empty_registry(self, monkeypatch):
        monkeypatch.setattr(pricing, "_RUNTIME_PRICING", {})

    def test_registered_model_is_priced(self):
        pricing.register_runtime_pricing("vendor/model", 1e-6, 2e-6)
        assert pricing.estimate_cost_usd("vendor/model", 1000, 500) == pytest.approx(0.002)

    def test_registry_beats_the_static_table(self):
        table = pricing._load()
        model = next(
            m for m, e in table["models"].items()
            if e.get("prompt_usd_per_token") is not None
        )
        pricing.register_runtime_pricing(model, 1e-3, 1e-3)
        assert pricing.get_model_pricing(model)["prompt_usd_per_token"] == pytest.approx(1e-3)

    def test_unregistered_model_still_returns_none(self):
        pricing.register_runtime_pricing("vendor/model", 1e-6, 2e-6)
        assert pricing.get_model_pricing("vendor/other") is None
        assert pricing.estimate_cost_usd("vendor/other", 1000, 1000) is None

    def test_half_known_price_is_ignored(self):
        # A partial registration must not shadow the static table with an
        # entry the estimator cannot use.
        pricing.register_runtime_pricing("vendor/model", 1e-6, None)
        assert pricing.get_model_pricing("vendor/model") is None

    def test_known_models_is_unaffected(self):
        before = pricing.known_models()
        pricing.register_runtime_pricing("vendor/model", 1e-6, 2e-6)
        assert pricing.known_models() == before
        assert "vendor/model" not in pricing.known_models()


@pytest.mark.parametrize(
    "cost,expected",
    [
        (None, None),
        (0, "$0.00"),
        (-1.0, "$0.00"),
        (0.0001234, "$0.0001"),
        (0.0233, "$0.023"),
        (0.5, "$0.500"),
        (12.345, "$12.35"),
        (1234.5, "$1,234.50"),
    ],
)
def test_format_cost_usd(cost, expected):
    assert pricing.format_cost_usd(cost) == expected

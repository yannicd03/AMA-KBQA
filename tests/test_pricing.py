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

"""Tests for AMA_RETRIEVAL_* environment overrides on the [retrieval] config.

The frontend sets these env vars so per-session retrieval toggles cross the
process boundary into MCP server subprocesses (which inherit the environment
at launch). Env must win over config.toml; unset vars leave toml values.
"""
from __future__ import annotations

import pytest

import ama_kbqa.config as cfg_module
from ama_kbqa.config import (
    RETRIEVAL_ENV_PREFIX,
    get_fusion,
    get_hybrid_enabled,
    get_prefetch_limit,
    get_rerank_threshold,
    get_reranker_enabled,
)

ALL_ENV_VARS = [
    RETRIEVAL_ENV_PREFIX + key
    for key in (
        "HYBRID_ENABLED",
        "FUSION",
        "PREFETCH_LIMIT",
        "RERANKER_ENABLED",
        "RERANKER_MODEL",
        "RERANK_CANDIDATES",
        "RERANK_THRESHOLD",
    )
]


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    """Pin a known [retrieval] config and clear all override env vars."""
    base = {"retrieval": {"hybrid_enabled": False, "fusion": "rrf"}}
    monkeypatch.setattr(cfg_module, "_config_cache", base)
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def test_no_env_vars_returns_toml_values():
    assert get_hybrid_enabled() is False
    assert get_fusion() == "rrf"
    assert get_reranker_enabled() is False


@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("1", True), ("yes", True), ("on", True), ("TRUE", True),
    ("false", False), ("0", False), ("no", False), ("off", False), ("", False),
])
def test_bool_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "HYBRID_ENABLED", raw)
    assert get_hybrid_enabled() is expected


def test_env_overrides_toml(monkeypatch):
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "HYBRID_ENABLED", "true")
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "FUSION", "dbsf")
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "RERANKER_ENABLED", "true")
    assert get_hybrid_enabled() is True
    assert get_fusion() == "dbsf"
    assert get_reranker_enabled() is True


def test_int_and_float_env_overrides(monkeypatch):
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "PREFETCH_LIMIT", "50")
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "RERANK_THRESHOLD", "0.5")
    assert get_prefetch_limit() == 50
    assert get_rerank_threshold() == pytest.approx(0.5)


def test_invalid_numeric_env_falls_back_to_toml(monkeypatch):
    cfg_module._config_cache["retrieval"]["prefetch_limit"] = 30
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "PREFETCH_LIMIT", "not-a-number")
    assert get_prefetch_limit() == 30


def test_env_overlay_does_not_mutate_config_cache(monkeypatch):
    monkeypatch.setenv(RETRIEVAL_ENV_PREFIX + "HYBRID_ENABLED", "true")
    get_hybrid_enabled()
    # The cached dict must stay pristine: the overlay works on a copy.
    assert cfg_module._config_cache["retrieval"]["hybrid_enabled"] is False

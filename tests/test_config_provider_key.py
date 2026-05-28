"""Regression tests for the provider API-key startup assertion.

Background: the system must not hard-require OPENROUTER_API_KEY (a deployment
may run exclusively against KIT). assert_provider_api_key_present() only
requires that *at least one* supported provider key is set, and fails fast
with a clear RuntimeError when none are.
"""

from __future__ import annotations

import pytest

from ama_kbqa.config import assert_provider_api_key_present


def test_kit_key_only_is_accepted(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("KIT_API_KEY", "kit-secret")
    # Should not raise.
    assert_provider_api_key_present()


def test_openrouter_key_only_is_accepted(monkeypatch):
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    # Should not raise.
    assert_provider_api_key_present()


def test_no_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("KIT_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="at least one"):
        assert_provider_api_key_present()

"""Keep the frontend tests off the network.

Rendering the chat page populates the model picker, which fetches two live
``/models`` catalogs (KIT, and OpenRouter since the booth build). With real
keys in .env those are real HTTP calls: slow, order-dependent through
``st.cache_data``, and enough on a bad day to blow the 30s AppTest timeout and
fail a test that has nothing to do with model discovery. Force both fetches to
fail so every test sees the deterministic offline path; tests that need a
catalog monkeypatch these same attributes with their own fixture.
"""

from __future__ import annotations

import pytest

from ama_kbqa.frontend.utils import chat_controls


@pytest.fixture(autouse=True)
def _offline_model_catalogs(monkeypatch):
    def _unreachable(*_args, **_kwargs):
        raise RuntimeError("model catalog fetch disabled in tests")

    monkeypatch.setattr(chat_controls, "_fetch_kit_models_meta", _unreachable)
    monkeypatch.setattr(chat_controls, "_fetch_openrouter_catalog", _unreachable)

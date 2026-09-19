"""Streamlit AppTest smoke test for the multi-provider model picker.

Runs `chat.py` headless with the shipped config and NO provider API keys, the
state a booth laptop is in before the keys are pasted into .env. The page must
still load, list only KIT models, and say once per provider why the billed
entries are missing. Nothing here calls a backend: the initial (no-messages,
no-live-run) view stops before any agent is constructed, and the KIT catalog
fetch fails without a key and falls back to the static list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

CHAT_PY = Path(__file__).resolve().parents[2] / "ama_kbqa" / "frontend" / "chat.py"


@pytest.fixture
def no_provider_keys(monkeypatch):
    """No provider keys at all, as on a booth laptop before .env is filled in.

    (The chat override env vars the page run writes are restored by the
    session-wide autouse fixture in tests/conftest.py.)
    """
    from ama_kbqa.frontend.utils import chat_controls

    for var in ("KIT_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # The KIT fetch is cached by st.cache_data across AppTest runs; force the
    # offline fallback rather than depending on cache state or the network.
    monkeypatch.setattr(
        chat_controls,
        "_fetch_kit_models_meta",
        lambda: (_ for _ in ()).throw(RuntimeError("no KIT key")),
    )


def _run_app():
    at = AppTest.from_file(str(CHAT_PY))
    at.run(timeout=30)
    return at


class TestModelPickerWithoutProviderKeys:
    def test_page_loads_without_exception(self, no_provider_keys):
        at = _run_app()
        assert not at.exception

    def test_picker_lists_only_kit_fallback_models(self, no_provider_keys):
        from ama_kbqa.frontend.utils import chat_controls

        at = _run_app()
        picker = next(s for s in at.selectbox if s.key == "chat_model_select")
        # AppTest exposes the *formatted* labels; every one must be a KIT entry.
        assert all(label.endswith(" · KIT") for label in picker.options)
        assert len(picker.options) == len(chat_controls._OFFLINE_FALLBACK_MODELS)
        # The selected value is the provider-qualified key.
        assert picker.value.startswith("kit:")
        assert picker.value[len("kit:"):] in chat_controls._OFFLINE_FALLBACK_MODELS

    def test_both_missing_key_notices_are_shown(self, no_provider_keys):
        at = _run_app()
        captions = [c.value for c in at.caption]
        assert "OpenRouter models hidden: OPENROUTER_API_KEY not set" in captions
        assert "DeepSeek models hidden: DEEPSEEK_API_KEY not set" in captions

"""Streamlit AppTest smoke test for the agent picker on the Chat page.

Runs `chat.py` headless (no live backend / MCP servers) and checks the picker
offers exactly the four expected entries, defaulting to "Orchestrator
(Router)". Nothing here calls a backend: the initial (no-messages, no-live-run)
view stops before any agent is constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

CHAT_PY = Path(__file__).resolve().parents[2] / "ama_kbqa" / "frontend" / "chat.py"


def _run_app():
    at = AppTest.from_file(str(CHAT_PY))
    at.run(timeout=30)
    return at


class TestAgentPickerSmoke:
    def test_page_loads_without_exception(self):
        at = _run_app()
        assert not at.exception

    def test_picker_offers_exactly_four_entries(self):
        at = _run_app()
        picker_buttons = [b for b in at.button if b.key and b.key.startswith("agentpick_")]
        labels = {b.label for b in picker_buttons}
        assert labels == {
            "Orchestrator (Router)",
            "Orchestrator (Federated)",
            "KQAPro",
            "SciQA",
        }

    def test_defaults_to_orchestrator_router(self):
        at = _run_app()
        # The picker's popover label mirrors the current selection; before
        # any explicit pick it must read the Router default.
        popover_blocks = at.get("popover")
        assert popover_blocks, "no st.popover rendered"
        assert popover_blocks[0].proto.popover.label == "Orchestrator (Router)"

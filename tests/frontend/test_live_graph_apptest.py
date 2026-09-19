"""Page-level tests for the live graph panel.

Two things are checked here that no unit test can see:

1. **The flag really gates the page.** `chat.py` grew a sidebar toggle and a
   two-column body; with `AMA_FRONTEND_LIVE_GRAPH=0` the page must render
   exactly as it did before the feature existed, toggle included. The env
   override (rather than a patched config cache) is deliberate: it is what a
   deployment with a read-only mounted config.toml actually uses, and in
   `get_live_graph_enabled` it wins over the cache, so it cannot be defeated
   by another test having already loaded config.toml.
2. **The frozen panel draws what it claims.** AppTest cannot exercise the
   live two-column state without a backend (constructing an agent needs MCP
   servers and KIT credentials), so the frozen half is driven directly and
   the HTML handed to `components.html` is inspected: the nodes the answer
   mentions must arrive at the template as highlighted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import ama_kbqa.config as cfg
from ama_kbqa.frontend.utils import live_graph_panel

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

CHAT_PY = Path(__file__).resolve().parents[2] / "ama_kbqa" / "frontend" / "chat.py"


def _run_app():
    at = AppTest.from_file(str(CHAT_PY))
    at.run(timeout=30)
    return at


def _toggle_keys(at) -> set:
    return {t.key for t in at.toggle}


class TestFlagGatesThePage:
    def test_flag_on_renders_the_sidebar_toggle(self, monkeypatch):
        monkeypatch.setenv(cfg.FRONTEND_LIVE_GRAPH_ENV, "1")
        at = _run_app()
        assert not at.exception
        assert "live_graph_view" in _toggle_keys(at)

    def test_flag_off_renders_no_toggle(self, monkeypatch):
        monkeypatch.setenv(cfg.FRONTEND_LIVE_GRAPH_ENV, "0")
        at = _run_app()
        assert not at.exception
        assert "live_graph_view" not in _toggle_keys(at)
        # The pre-feature toggle is untouched either way.
        assert "simplified_view" in _toggle_keys(at)

    def test_landing_view_stays_single_column_with_the_flag_on(self, monkeypatch):
        # No messages and no completed trace: there is nothing explored yet,
        # so the page must not split into columns.
        monkeypatch.setenv(cfg.FRONTEND_LIVE_GRAPH_ENV, "1")
        at = _run_app()
        assert not at.exception
        assert not at.get("column")


class TestFrozenPanelRendering:
    TRACE = {
        "trace_id": "trace-1",
        "agent": "KQAPro",
        # Mentions Q937 but not the neighbour's label, so the highlight set is
        # a real discrimination rather than "everything lights up".
        "answer": "Albert Einstein (Q937) developed the theory of relativity.",
        "events": [],
        "journal_snapshots": [
            {
                "trigger": "after:GetNodeSummary",
                "state": {
                    "visited_nodes": {
                        "Q937": "Albert Einstein",
                        "Q169470": "physicist",
                    },
                    "verified_facts": [
                        {
                            "subject": "Q937",
                            "relation": "occupation",
                            "related_id": "Q169470",
                        }
                    ],
                },
            }
        ],
    }

    def _capture(self, monkeypatch, trace) -> str:
        captured: dict = {}

        def _fake_html(html, **kwargs):
            captured["html"] = html
            captured["kwargs"] = kwargs

        monkeypatch.setattr(live_graph_panel.components, "html", _fake_html)
        live_graph_panel.render_live_graph_panel(
            live_run=None, trace=trace, key_prefix="chat",
        )
        return captured

    def _payload(self, html: str) -> dict:
        match = re.search(r"const payload = (\{.*\});", html)
        assert match, "no payload spliced into the template"
        # The template escapes "</" as "<\/" so an HTML parser cannot end the
        # inline script early; JSON parses that back to the original text.
        return json.loads(match.group(1).replace("<\\/", "</"))

    def test_answer_nodes_reach_the_template_as_highlighted(self, monkeypatch):
        captured = self._capture(monkeypatch, self.TRACE)
        payload = self._payload(captured["html"])

        assert set(payload["highlight"]) == {"Q937"}
        highlighted = {n["id"] for n in payload["nodes"] if n["highlighted"]}
        assert highlighted == {"Q937"}
        # The whole explored subgraph is drawn, not only the answer.
        assert {"Q937", "Q169470"} <= {n["id"] for n in payload["nodes"]}
        assert payload["edges"], "the occupation edge is missing"
        assert captured["kwargs"]["height"] == live_graph_panel.IFRAME_HEIGHT_PX

    def test_empty_trace_draws_no_iframe(self, monkeypatch):
        captured = self._capture(
            monkeypatch,
            {"trace_id": "empty", "agent": "KQAPro", "events": [],
             "journal_snapshots": []},
        )
        assert "html" not in captured

    def test_missing_trace_never_raises(self, monkeypatch):
        captured = self._capture(monkeypatch, None)
        assert "html" not in captured

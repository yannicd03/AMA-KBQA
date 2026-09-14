"""Tests for the vis-network HTML template (frontend/utils/graph_html.py).

The template is a string with a JSON payload spliced into it, so the things
that can silently break it are not type errors: a dropped group name (nodes
render in the wrong colour), a lost localStorage key (the graph re-scatters
on every one-second tick), a leftover dark-theme colour (the panel reads as a
hole in a light page), or a label with a quote in it corrupting the payload.
Each of those gets an assertion here.

`journal_to_graph` is covered too: it is the compatibility shim the post-hoc
scrubber in `graph_panel.py` still calls, so its list-of-dicts shape has to
survive the rewrite of the normaliser underneath it.
"""

from __future__ import annotations

import json
import re

from ama_kbqa.frontend.utils.graph_html import build_graph_html, journal_to_graph
from ama_kbqa.frontend.utils.live_graph_data import (
    graph_from_journal_state,
    to_vis_payload,
)


def _payload_of(html_doc: str) -> dict:
    """Pull the spliced JSON payload back out of the rendered template."""
    match = re.search(r"const payload = (\{.*?\});\n", html_doc, re.DOTALL)
    assert match, "payload literal not found in template"
    return json.loads(match.group(1))


def _demo_graph():
    state = {
        "visited_nodes": {"Q1": "Berlin"},
        "found_values": {"Q1": {"population": "3700000"}},
        "verified_facts": [{"subject": "Q1", "predicate": "country", "object": "Q2"}],
    }
    return graph_from_journal_state(state, source="kqapro")


class TestTemplateContents:
    def test_all_four_groups_are_styled(self):
        doc = build_graph_html([], [])
        for group in ("kqapro", "sciqa", "literal", "candidate"):
            assert f"{group}:" in doc or f'"{group}"' in doc, group

    def test_legend_names_the_groups_and_the_answer_highlight(self):
        doc = build_graph_html([], [])
        for entry in ("KQAPro", "SciQA", "literal", "candidate", "in answer"):
            assert f">{entry}</div>" in doc, entry

    def test_no_dark_theme_colours_remain(self):
        doc = build_graph_html([], [])
        for dark in ("#0d1117", "#30363d", "#c9d1d9", "#e6edf3", "#58a6ff", "#a371f7"):
            assert dark not in doc, dark
        assert "#fafafa" in doc and "#3f3f46" in doc

    def test_position_persistence_keys_are_wired(self):
        doc = build_graph_html([], [], view_state_key="k")
        assert 'const posKey = stateKey + ":pos";' in doc
        assert "network.getPositions()" in doc
        assert 'network.once("stabilized"' in doc
        assert 'network.on("dragEnd"' in doc

    def test_stabilisation_only_fits_when_there_are_no_saved_positions(self):
        doc = build_graph_html([], [])
        assert "stabilization: { iterations: 80, fit: !hasSavedPositions }" in doc

    def test_reset_clears_both_localstorage_keys(self):
        doc = build_graph_html([], [], reset=True)
        assert _payload_of(doc)["reset"] is True
        assert "localStorage.removeItem(stateKey)" in doc
        assert "localStorage.removeItem(posKey)" in doc

    def test_candidate_nodes_are_dashed_and_literals_are_boxes(self):
        doc = build_graph_html([], [])
        assert 'borderDashes: group === "candidate" ? [4, 3] : false' in doc
        assert 'shape: group === "literal" ? "box" : "dot"' in doc

    def test_answer_nodes_get_the_orange_fill_and_a_thicker_border(self):
        doc = build_graph_html([], [])
        assert '#ffedd5' in doc and '#f97316' in doc
        assert "borderWidth: isAnswer ? 4" in doc

    def test_cdn_pin_is_unchanged(self):
        assert "vis-network@9.1.9" in build_graph_html([], [])


class TestPayloadEmbedding:
    def test_payload_survives_the_round_trip(self):
        graph = _demo_graph()
        vis = to_vis_payload(graph, highlight={"Q2"}, new_ids={"Q1"})
        doc = build_graph_html(
            vis["nodes"], vis["edges"],
            view_state_key="ama_kbqa_graph::chat::T1",
            highlight_node_ids=["Q2"],
            new_node_ids=["Q1"],
        )
        payload = _payload_of(doc)
        assert payload["nodes"] == vis["nodes"]
        assert payload["edges"] == vis["edges"]
        assert payload["highlight"] == ["Q2"]
        assert payload["new"] == ["Q1"]
        assert payload["view_state_key"] == "ama_kbqa_graph::chat::T1"
        assert payload["reset"] is False

    def test_quotes_and_markup_in_labels_do_not_corrupt_the_payload(self):
        nodes = [{
            "id": "Q1",
            "label": 'He said "hi" </script>',
            "title": "<b>a &amp; b</b>",
            "group": "kqapro",
            "highlighted": False,
            "new": False,
        }]
        doc = build_graph_html(nodes, [])
        # A literal "</" inside the inline script ends the script element in
        # an HTML parser, JS string or not, so it must be escaped on the way
        # in and unescaped by the JSON parse on the way out.
        raw = re.search(r"const payload = (\{.*?\});\n", doc, re.DOTALL).group(1)
        assert "</" not in raw
        payload = _payload_of(doc)
        assert payload["nodes"][0]["label"] == 'He said "hi" </script>'
        assert payload["nodes"][0]["title"] == "<b>a &amp; b</b>"

    def test_height_is_applied_to_the_container(self):
        assert "height: 480px;" in build_graph_html([], [], height_px=480)


class TestJournalToGraphCompatibility:
    def test_returns_the_old_list_of_dicts_shape(self):
        nodes, edges = journal_to_graph({
            "visited_nodes": {"Q1": "Berlin", "Q2": "Germany"},
            "verified_facts": [
                {"subject": "Q1", "predicate": "country", "object": "Q2", "source": "tool"}
            ],
        })
        assert {n["id"] for n in nodes} == {"Q1", "Q2"}
        assert set(nodes[0]) == {"id", "label", "title", "group"}
        assert len(edges) == 1
        assert set(edges[0]) == {"id", "from", "to", "label", "title"}
        assert edges[0]["from"] == "Q1" and edges[0]["to"] == "Q2"
        assert edges[0]["label"] == "country"

    def test_literals_keep_the_literal_group_the_scrubber_filters_on(self):
        nodes, _ = journal_to_graph({
            "visited_nodes": {"Q1": "Berlin"},
            "verified_facts": [{"subject": "Q1", "predicate": "population", "object": "3700000"}],
        })
        literals = [n for n in nodes if n["group"] == "literal"]
        assert [n["label"] for n in literals] == ["3700000"]

    def test_kg_name_drives_the_group_colour(self):
        nodes, _ = journal_to_graph({
            "visited_nodes": {"R1": "A paper"}, "kg_name": "ORKG",
        })
        assert nodes[0]["group"] == "sciqa"

    def test_empty_and_malformed_states_yield_empty_lists(self):
        assert journal_to_graph({}) == ([], [])
        assert journal_to_graph(None) == ([], [])

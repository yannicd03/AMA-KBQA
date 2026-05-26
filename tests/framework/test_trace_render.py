"""Tests for trace-rendering and graph helpers used by the frontend."""

from __future__ import annotations

from ama_kbqa.frontend.utils.trace_render import (
    build_tree,
    format_duration,
    span_button_label,
    summarise,
    render_summary_html,
    render_tree_html,
)
from ama_kbqa.frontend.utils.graph_html import journal_to_graph


def _ev(span_id, parent, kind, name, duration_ms=10.0, is_event=False, attrs=None, status="ok"):
    return {
        "trace_id": "T",
        "span_id": span_id,
        "parent_span_id": parent,
        "kind": kind,
        "name": name,
        "start_time_unix_nano": 0,
        "end_time_unix_nano": int(duration_ms * 1e6),
        "duration_ms": duration_ms,
        "status": status,
        "is_event": is_event,
        "attributes": attrs or {},
        "payload": {},
        "error": None,
    }


class TestSpanButtonLabel:
    def test_span_label_has_kind_badge_name_and_duration(self):
        label = span_button_label(_ev("a", None, "tool_call", "FindNode", duration_ms=42.0), depth=0)
        assert ":green-background[tool_call]" in label
        assert "FindNode" in label
        assert ":gray[42ms]" in label

    def test_depth_indents_with_emspaces(self):
        flat = span_button_label(_ev("a", None, "llm_call", "x"), depth=0)
        nested = span_button_label(_ev("a", None, "llm_call", "x"), depth=2)
        assert " " not in flat
        assert nested.startswith("    ")  # 2 em-spaces per depth

    def test_event_marker_and_no_duration(self):
        label = span_button_label(_ev("a", None, "loop_detected", "x", is_event=True), depth=1)
        assert "•" in label
        assert "ms]" not in label  # events carry no duration badge

    def test_error_status_appends_marker(self):
        label = span_button_label(_ev("a", None, "tool_call", "x", status="error"), depth=0)
        assert ":red[●]" in label

    def test_unknown_kind_falls_back_to_gray(self):
        label = span_button_label(_ev("a", None, "mystery_kind", "x"), depth=0)
        assert ":gray-background[mystery_kind]" in label


class TestBuildTree:
    def test_simple_tree_structure(self):
        events = [
            _ev("a", None, "agent_run", "ask"),
            _ev("b", "a", "llm_call", "gpt-4o", attrs={"prompt_tokens": 10, "completion_tokens": 5}),
            _ev("c", "a", "tool_call", "FindNode"),
        ]
        tree = build_tree(events)
        assert len(tree["roots"]) == 1
        assert tree["roots"][0]["span_id"] == "a"
        assert {c["span_id"] for c in tree["children"]["a"]} == {"b", "c"}

    def test_orphans_become_roots(self):
        events = [
            _ev("a", "missing-parent", "tool_call", "Foo"),
        ]
        tree = build_tree(events)
        assert len(tree["roots"]) == 1
        assert tree["roots"][0]["span_id"] == "a"


class TestSummarise:
    def test_aggregate_counts_and_tokens(self):
        events = [
            _ev("a", None, "agent_run", "ask"),
            _ev("b", "a", "llm_call", "gpt-4o", attrs={"prompt_tokens": 10, "completion_tokens": 5}),
            _ev("c", "a", "tool_call", "FindNode"),
            _ev("d", "a", "tool_call", "Bar", status="error"),
        ]
        s = summarise(events)
        assert s["n_llm_calls"] == 1
        assert s["n_tool_calls"] == 2
        assert s["n_errors"] == 1
        assert s["total_tokens"] == 15

    def test_empty_summary(self):
        s = summarise([])
        assert s["total_duration_ms"] == 0
        assert s["n_llm_calls"] == 0
        assert s["n_tool_calls"] == 0
        assert s["n_errors"] == 0


class TestRenderHTML:
    def test_render_tree_returns_string_with_kinds(self):
        events = [
            _ev("a", None, "agent_run", "ask"),
            _ev("b", "a", "llm_call", "gpt-4o"),
        ]
        out = render_tree_html(events, selected_span_id="b")
        assert "trace-tree" in out
        assert "agent_run" in out
        assert "llm_call" in out
        assert "selected" in out

    def test_render_summary_html_includes_stats(self):
        events = [
            _ev("a", None, "agent_run", "ask"),
            _ev("b", "a", "llm_call", "gpt-4o", attrs={"prompt_tokens": 7, "completion_tokens": 3}),
        ]
        out = render_summary_html(events)
        assert "trace-summary" in out
        assert "10" in out  # 7 + 3 = 10 total tokens

    def test_format_duration_buckets(self):
        assert format_duration(0.4).endswith("µs")
        assert format_duration(50).endswith("ms")
        assert format_duration(2500).endswith("s")
        assert "m" in format_duration(125_000)


class TestJournalToGraph:
    def test_visited_nodes_become_graph_nodes(self):
        state = {
            "visited_nodes": {"Q1": "Berlin", "Q2": "Germany"},
            "verified_facts": [
                {"subject": "Q1", "predicate": "country", "object": "Q2", "source": "tool"}
            ],
        }
        nodes, edges = journal_to_graph(state)
        ids = {n["id"] for n in nodes}
        assert ids == {"Q1", "Q2"}
        assert len(edges) == 1
        assert edges[0]["from"] == "Q1" and edges[0]["to"] == "Q2"
        assert edges[0]["label"] == "country"

    def test_unvisited_endpoint_creates_pseudo_node(self):
        state = {
            "visited_nodes": {"Q1": "Berlin"},
            "verified_facts": [
                {"subject": "Q1", "predicate": "population", "object": "3700000", "source": "tool"}
            ],
        }
        nodes, edges = journal_to_graph(state)
        ids = {n["id"]: n for n in nodes}
        assert "3700000" in ids
        # Literal value should be classified as 'literal'.
        assert ids["3700000"]["group"] == "literal"

    def test_found_values_appear_in_node_tooltip(self):
        state = {
            "visited_nodes": {"Q1": "Berlin"},
            "found_values": {"Q1": {"population": 3700000}},
            "verified_facts": [],
        }
        nodes, _ = journal_to_graph(state)
        node = next(n for n in nodes if n["id"] == "Q1")
        assert "population" in node["title"]

    def test_empty_state_yields_empty_lists(self):
        nodes, edges = journal_to_graph({})
        assert nodes == [] and edges == []

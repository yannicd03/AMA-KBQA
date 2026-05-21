"""Tests for span-kind → lifecycle-node mapping."""

from __future__ import annotations

import pytest

from ama_kbqa.frontend.utils.lifecycle_mapping import (
    TOOLS_A,
    TOOLS_B,
    span_to_node_ids,
)


@pytest.mark.parametrize("kind,name,phase,expected", [
    ("agent_run", "ask", "open",  ["user_query"]),
    ("classify",  "classify", "open",  ["pre_classifier", "pre_extractor"]),
    ("classify",  "classify", "close", ["pre_strategy_inject"]),
    ("tool_loop_iter", "iter 0", "event", ["main_llm_reason"]),
    ("llm_call", "gpt-4o", "open",  ["main_llm_reason"]),
    ("llm_call", "gpt-4o", "close", ["main_llm_reason"]),
    ("loop_detected", "consecutive_tool_repeat", "event", ["main_loop_detect"]),
    ("intervention", "zero_tool_call_retry", "event", ["main_more"]),
    ("context_trim", "truncate", "event", ["main_more"]),
    ("synthesis", "final", "open",  ["post_synthesis"]),
    ("synthesis", "final", "close", ["post_answer", "post_response"]),
    ("agent_run", "ask", "close", ["post_evaluate", "post_lessons"]),
])
def test_mapping(kind, name, phase, expected):
    assert span_to_node_ids(kind, name, phase=phase) == expected


class TestToolsSplit:
    def test_tools_a_traversal_lights_tools_a(self):
        for tool in ("FindNode", "FindResource", "GetResourceDetails"):
            assert span_to_node_ids("tool_call", tool, phase="open") == ["main_tools_a"]

    def test_tools_b_summary_lights_tools_b(self):
        for tool in ("GetNodeSummary", "VerifyFact", "ManageJournal"):
            assert span_to_node_ids("tool_call", tool, phase="open") == ["main_tools_b"]

    def test_journal_snapshot_tool_lights_journal_state(self):
        assert span_to_node_ids("tool_call", "GetJournalStateJSON", phase="open") == [
            "main_journal_state"
        ]

    def test_tools_a_and_b_are_disjoint(self):
        assert TOOLS_A.isdisjoint(TOOLS_B)

    def test_unknown_tool_falls_back_to_journal_state(self):
        # We never want the figure to go fully dark during an unknown tool call.
        assert span_to_node_ids("tool_call", "WeirdNewTool", phase="open") == [
            "main_journal_state"
        ]


class TestDelegateAndUnknowns:
    def test_delegate_is_noop(self):
        assert span_to_node_ids("delegate", "kqapro", phase="open") == []
        assert span_to_node_ids("delegate", "kqapro", phase="close") == []

    def test_completely_unknown_kind_is_noop(self):
        assert span_to_node_ids("brand_new_kind", "x", phase="open") == []

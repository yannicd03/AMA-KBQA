"""Tests for span-kind → lifecycle-node mapping."""

from __future__ import annotations

import pytest

from ama_kbqa.frontend.utils.lifecycle_mapping import (
    LOOP_BACK_EDGE_ID,
    ORCH_SUBAGENTS,
    TOOLS_A,
    TOOLS_B,
    lifecycle_node_phase,
    orchestrator_span_to_node_ids,
    span_to_edge_ids,
    span_to_node_ids,
)


@pytest.mark.parametrize("kind,name,phase,expected", [
    ("agent_run", "ask", "open",  ["agent_invocation"]),
    ("classify",  "classify", "open",  ["pre_classifier", "pre_extractor"]),
    ("classify",  "classify", "close", ["pre_strategy_inject"]),
    ("tool_loop_iter", "iter 0", "event", ["main_llm_reason"]),
    ("llm_call", "gpt-4o", "open",  ["main_llm_reason"]),
    ("llm_call", "gpt-4o", "close", ["main_llm_reason"]),
    ("loop_detected", "consecutive_tool_repeat", "event", ["main_done"]),
    ("intervention", "zero_tool_call_retry", "event", ["main_done"]),
    ("context_trim", "truncate", "event", ["main_done"]),
    ("synthesis", "final", "open",  ["post_synthesis"]),
    ("synthesis", "final", "close", ["post_synthesis"]),
    ("agent_run", "ask", "close", ["post_evaluate", "post_lessons"]),
])
def test_mapping(kind, name, phase, expected):
    assert span_to_node_ids(kind, name, phase=phase) == expected


class TestToolsSplit:
    def test_traversal_tools_light_tool_call(self):
        for tool in ("FindNode", "FindResource", "GetResourceDetails"):
            assert span_to_node_ids("tool_call", tool, phase="open") == ["main_tool_call"]

    def test_summary_tools_light_tool_call(self):
        for tool in ("GetNodeSummary", "VerifyFact", "ManageJournal"):
            assert span_to_node_ids("tool_call", tool, phase="open") == ["main_tool_call"]

    def test_journal_snapshot_tool_lights_scratchpad(self):
        assert span_to_node_ids("tool_call", "GetJournalStateJSON", phase="open") == [
            "main_scratchpad"
        ]

    def test_tools_a_and_b_are_disjoint(self):
        assert TOOLS_A.isdisjoint(TOOLS_B)

    def test_unknown_tool_falls_back_to_tool_call(self):
        # We never want the figure to go fully dark during an unknown tool call.
        assert span_to_node_ids("tool_call", "WeirdNewTool", phase="open") == [
            "main_tool_call"
        ]


class TestDelegateAndUnknowns:
    def test_delegate_is_noop(self):
        assert span_to_node_ids("delegate", "kqapro", phase="open") == []
        assert span_to_node_ids("delegate", "kqapro", phase="close") == []

    def test_completely_unknown_kind_is_noop(self):
        assert span_to_node_ids("brand_new_kind", "x", phase="open") == []


class TestEdgeMapping:
    def test_loop_back_lights_only_from_second_iteration(self):
        # Iteration 1 is the initial entry into the loop, not a loop-back.
        assert span_to_edge_ids(
            "tool_loop_iter", "iter:1", phase="event", attributes={"iteration": 1}
        ) == []
        # Iteration 2+ is a genuine new ReAct cycle → light the loop-back arrow.
        assert span_to_edge_ids(
            "tool_loop_iter", "iter:2", phase="event", attributes={"iteration": 2}
        ) == [LOOP_BACK_EDGE_ID]
        assert span_to_edge_ids(
            "tool_loop_iter", "iter:7", phase="event", attributes={"iteration": 7}
        ) == [LOOP_BACK_EDGE_ID]

    def test_loop_back_tolerates_missing_or_bad_iteration(self):
        assert span_to_edge_ids("tool_loop_iter", "iter", phase="event") == []
        assert span_to_edge_ids(
            "tool_loop_iter", "iter", phase="event", attributes={"iteration": "nope"}
        ) == []

    def test_other_kinds_light_no_edges(self):
        assert span_to_edge_ids("llm_call", "gpt-4o", phase="open") == []
        assert span_to_edge_ids("tool_call", "FindNode", phase="open") == []
        assert span_to_edge_ids("agent_run", "ask", phase="close") == []


class TestOrchestratorMapping:
    def test_orchestrator_root_lights_user_then_combine(self):
        assert orchestrator_span_to_node_ids(
            "agent_run", "ORCHESTRATOR", phase="open"
        ) == ["orch_user"]
        assert orchestrator_span_to_node_ids(
            "agent_run", "ORCHESTRATOR", phase="close"
        ) == ["orch_combine"]

    def test_route_lights_probe_and_dispatch(self):
        assert orchestrator_span_to_node_ids(
            "classify", "route", phase="open"
        ) == ["orch_probe", "orch_dispatch"]
        # Closing the route span re-lights nothing (the boxes settle to visited).
        assert orchestrator_span_to_node_ids("classify", "route", phase="close") == []

    def test_delegate_lights_its_container_on_open(self):
        assert orchestrator_span_to_node_ids(
            "delegate", "kqapro_agent", phase="open",
            attributes={"sub_agent": "kqapro_agent"},
        ) == ["sub_kqapro"]
        assert orchestrator_span_to_node_ids(
            "delegate", "sciqa_agent", phase="open",
            attributes={"sub_agent": "sciqa_agent"},
        ) == ["sub_sciqa"]
        # On close the container settles (the runner pops it off the stack).
        assert orchestrator_span_to_node_ids(
            "delegate", "kqapro_agent", phase="close",
            attributes={"sub_agent": "kqapro_agent"},
        ) == []

    def test_unknown_delegate_target_lights_nothing(self):
        assert orchestrator_span_to_node_ids(
            "delegate", "mystery_agent", phase="open",
            attributes={"sub_agent": "mystery_agent"},
        ) == []

    def test_sub_agent_internal_spans_do_not_light_orchestrator_figure(self):
        # A sub-agent's own classify/llm/tool spans drive its detail figure,
        # not the orchestrator figure.
        assert orchestrator_span_to_node_ids("classify", "classify", phase="open") == []
        assert orchestrator_span_to_node_ids("llm_call", "gpt-4o", phase="open") == []
        assert orchestrator_span_to_node_ids("tool_call", "FindNode", phase="open") == []

    def test_subagent_registry_matches_container_ids(self):
        assert ORCH_SUBAGENTS["kqapro_agent"] == ("KQAPro", "sub_kqapro")
        assert ORCH_SUBAGENTS["sciqa_agent"] == ("SciQA", "sub_sciqa")

    def test_node_phase_mapping(self):
        assert lifecycle_node_phase("agent_invocation") == "pre"
        assert lifecycle_node_phase("pre_strategy_inject") == "pre"
        assert lifecycle_node_phase("main_llm_reason") == "main"
        assert lifecycle_node_phase("main_done") == "main"
        assert lifecycle_node_phase("post_synthesis") == "post"
        assert lifecycle_node_phase("nonexistent") is None

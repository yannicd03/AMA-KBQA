"""Tests for `_loop_exempt_tools`: workhorse exploration tools (e.g.
Wikidata's RunSPARQL) are EXPECTED to repeat with different arguments —
that's the intended workflow, not a loop. They must be exempt from the
name-frequency detectors (2: oscillation, 3: same-tool-5-of-6), but
Detection 1 (identical args 3x in a row) must still catch their true loops.
"""

from __future__ import annotations

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.config import KnowledgeGraphConfig


class _ConcreteAgent(BaseKBQAAgent):
    def get_config(self) -> KnowledgeGraphConfig:  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def _bare_agent(loop_exempt_tools=None):
    """An agent instance without the heavy __init__ (only loop-detection
    state is needed for _detect_loops)."""
    agent = object.__new__(_ConcreteAgent)
    agent.tool_call_history = []
    agent.tool_sequence = []
    agent.tool_call_counts = {}
    if loop_exempt_tools is not None:
        agent._loop_exempt_tools = loop_exempt_tools
    return agent


def test_exempt_tool_called_five_times_in_six_does_not_trigger_detection_3():
    agent = _bare_agent(loop_exempt_tools={"RunSPARQL"})

    detected = False
    reason = ""
    # Five distinct RunSPARQL calls (differing args) plus one other tool.
    calls = [
        ("RunSPARQL", {"query": "SELECT ?a WHERE { ?a a wd:Q1 }"}),
        ("RunSPARQL", {"query": "SELECT ?b WHERE { ?b a wd:Q2 }"}),
        ("GetEntityProperties", {"entity_id": "Q1"}),
        ("RunSPARQL", {"query": "SELECT ?c WHERE { ?c a wd:Q3 }"}),
        ("RunSPARQL", {"query": "SELECT ?d WHERE { ?d a wd:Q4 }"}),
        ("RunSPARQL", {"query": "SELECT ?e WHERE { ?e a wd:Q5 }"}),
    ]
    for name, args in calls:
        detected, reason = agent._detect_loops(name, args)

    assert detected is False, reason


def test_non_exempt_tool_called_five_times_in_six_still_triggers_detection_3():
    agent = _bare_agent(loop_exempt_tools={"RunSPARQL"})

    detected = False
    reason = ""
    calls = [
        ("GetEntityProperties", {"entity_id": "Q1"}),
        ("GetEntityProperties", {"entity_id": "Q2"}),
        ("RunSPARQL", {"query": "SELECT ?a WHERE { ?a a wd:Q1 }"}),
        ("GetEntityProperties", {"entity_id": "Q3"}),
        ("GetEntityProperties", {"entity_id": "Q4"}),
        ("GetEntityProperties", {"entity_id": "Q5"}),
    ]
    for name, args in calls:
        detected, reason = agent._detect_loops(name, args)

    assert detected is True
    assert "GetEntityProperties" in reason
    assert "5 times" in reason


def test_exempt_tool_oscillation_ab_does_not_trigger_detection_2():
    agent = _bare_agent(loop_exempt_tools={"RunSPARQL"})

    detected = False
    reason = ""
    # RunSPARQL <-> OtherTool A-B-A-B-A-B oscillation, with distinct args so
    # Detection 1 (identical repeat) never fires either.
    calls = [
        ("RunSPARQL", {"query": "Q1"}),
        ("OtherTool", {"x": 1}),
        ("RunSPARQL", {"query": "Q2"}),
        ("OtherTool", {"x": 2}),
        ("RunSPARQL", {"query": "Q3"}),
        ("OtherTool", {"x": 3}),
    ]
    for name, args in calls:
        detected, reason = agent._detect_loops(name, args)

    assert detected is False, reason


def test_non_exempt_oscillation_ab_still_triggers_detection_2():
    agent = _bare_agent(loop_exempt_tools={"RunSPARQL"})

    detected = False
    reason = ""
    calls = [
        ("ToolA", {"x": 1}),
        ("ToolB", {"x": 1}),
        ("ToolA", {"x": 2}),
        ("ToolB", {"x": 2}),
        ("ToolA", {"x": 3}),
        ("ToolB", {"x": 3}),
    ]
    for name, args in calls:
        detected, reason = agent._detect_loops(name, args)

    assert detected is True
    assert "Oscillating" in reason


def test_identical_args_three_times_still_triggers_for_exempt_tools():
    """Detection 1 (identical call repeated 3x) is not affected by the
    exemption — a real loop on an exempt tool must still be caught."""
    agent = _bare_agent(loop_exempt_tools={"RunSPARQL"})

    same_args = {"query": "SELECT ?a WHERE { ?a a wd:Q1 }"}
    detected, reason = agent._detect_loops("RunSPARQL", same_args)
    assert detected is False
    detected, reason = agent._detect_loops("RunSPARQL", same_args)
    assert detected is False
    detected, reason = agent._detect_loops("RunSPARQL", same_args)

    assert detected is True
    assert "Identical call repeated 3 times" in reason
    assert "RunSPARQL" in reason


def test_no_exemption_set_defaults_to_empty_and_behaves_as_before():
    """Without `_loop_exempt_tools` set, exemption defaults to an empty set
    (getattr default), so name-frequency detectors apply to every tool."""
    agent = _bare_agent(loop_exempt_tools=None)

    detected = False
    reason = ""
    calls = [
        ("RunSPARQL", {"query": "Q1"}),
        ("RunSPARQL", {"query": "Q2"}),
        ("GetEntityProperties", {"entity_id": "Q1"}),
        ("RunSPARQL", {"query": "Q3"}),
        ("RunSPARQL", {"query": "Q4"}),
        ("RunSPARQL", {"query": "Q5"}),
    ]
    for name, args in calls:
        detected, reason = agent._detect_loops(name, args)

    assert detected is True
    assert "RunSPARQL" in reason

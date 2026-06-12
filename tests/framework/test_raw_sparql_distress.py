"""Tests for the raw-SPARQL distress intervention in the base agent loop."""

from __future__ import annotations

from ama_kbqa.framework.base_agent import BaseKBQAAgent


class _Recorder:
    def __init__(self):
        self.events = []

    def event(self, kind, name, attributes=None):
        self.events.append((kind, name, attributes or {}))


class DistressAgent(BaseKBQAAgent):
    """Test double exposing only the state the intervention reads."""

    def __init__(self, counts=None, threshold=None):
        self._messages = []
        self.tool_call_counts = dict(counts or {})
        self.recorder = _Recorder()
        if threshold is not None:
            self._raw_sparql_distress_threshold = threshold

    def get_config(self):
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass


def test_below_threshold_no_intervention():
    agent = DistressAgent(counts={"RunSPARQL": 3})
    assert agent._maybe_inject_raw_sparql_distress(5) is False
    assert agent._messages == []


def test_at_threshold_injects_once():
    agent = DistressAgent(counts={"RunSPARQL": 4})
    assert agent._maybe_inject_raw_sparql_distress(6) is True
    assert len(agent._messages) == 1
    msg = agent._messages[0]
    assert msg["role"] == "user"
    assert "RAW SPARQL DISTRESS" in msg["content"]
    assert "4 raw SPARQL queries" in msg["content"]
    assert "RunSPARQL" in msg["content"]
    assert agent.recorder.events == [
        ("intervention", "raw_sparql_distress",
         {"raw_calls": 4, "threshold": 4, "iteration": 6}),
    ]
    # Second crossing does not re-fire.
    agent.tool_call_counts["RunSPARQL"] = 9
    assert agent._maybe_inject_raw_sparql_distress(7) is False
    assert len(agent._messages) == 1


def test_counts_sum_across_both_raw_tools():
    agent = DistressAgent(counts={"RunSPARQL": 2, "RunORKGSPARQL": 2})
    assert agent._maybe_inject_raw_sparql_distress(4) is True
    assert "RunORKGSPARQL, RunSPARQL" in agent._messages[0]["content"]


def test_threshold_is_overridable():
    agent = DistressAgent(counts={"RunORKGSPARQL": 2}, threshold=2)
    assert agent._maybe_inject_raw_sparql_distress(3) is True


def test_wrapped_tool_calls_do_not_count():
    agent = DistressAgent(counts={"FindNode": 10, "QueryComparisonRows": 6})
    assert agent._maybe_inject_raw_sparql_distress(8) is False
    assert agent._messages == []


def test_reset_rearms_intervention():
    agent = DistressAgent(counts={"RunSPARQL": 4})
    assert agent._maybe_inject_raw_sparql_distress(5) is True
    agent._raw_sparql_intervention_done = False  # what reset() does
    agent.tool_call_counts = {"RunSPARQL": 4}
    assert agent._maybe_inject_raw_sparql_distress(2) is True

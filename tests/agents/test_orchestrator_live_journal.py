"""Tests for Orchestrator.live_journal_snapshots().

Why a separate accessor exists at all: `self.journal_snapshots` only gains a
specialist's snapshots once `_run_specialist` hoists them, which happens
after that specialist has finished. The live graph panel needs them while the
run is still in flight, so it reads the loaded specialists directly.

Two contracts are load-bearing and both are tested here: every snapshot is
tagged with its `source_agent` (a federated run merges two specialists'
journals and they must stay attributable), and the call never raises, because
it runs on the Streamlit thread against lists the agent thread is still
appending to.

Same hermetic style as tests/agents/test_orchestrator_routing.py: the
Orchestrator is built via __new__ so __init__ (API keys, LLM client) is
bypassed, and the sub-agents are stubs.
"""

from __future__ import annotations

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator


class _FakeAgent:
    """Minimal sub-agent stub with just the journal surface the accessor reads."""

    def __init__(self, snapshots):
        self.journal_snapshots = snapshots


def _make_orchestrator(agents=None):
    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.journal_snapshots = []
    o._agents = agents if agents is not None else {}
    return o


class TestLiveJournalSnapshots:
    def test_no_agents_loaded_yields_nothing(self):
        assert _make_orchestrator().live_journal_snapshots() == []

    def test_snapshots_are_tagged_with_their_specialist(self):
        o = _make_orchestrator({
            "kqapro_agent": _FakeAgent([
                {"ts": 1.0, "trigger": "after:FindNode", "state": {"visited_nodes": {"Q1": "Berlin"}}},
            ]),
            "sciqa_agent": _FakeAgent([
                {"ts": 2.0, "trigger": "after:FindResource", "state": {"visited_nodes": {"R1": "Paper"}}},
            ]),
        })
        snapshots = o.live_journal_snapshots()
        assert sorted(s["source_agent"] for s in snapshots) == ["kqapro_agent", "sciqa_agent"]
        assert {s["trigger"] for s in snapshots} == {"after:FindNode", "after:FindResource"}

    def test_original_snapshot_dicts_are_not_mutated(self):
        snapshot = {"ts": 1.0, "trigger": "t", "state": {}}
        o = _make_orchestrator({"kqapro_agent": _FakeAgent([snapshot])})
        o.live_journal_snapshots()
        assert "source_agent" not in snapshot

    def test_returns_snapshots_before_the_specialist_finished(self):
        """The hoist in _run_specialist has not happened yet, so the
        orchestrator's own list is still empty; the accessor must still see
        what the running specialist has journalled."""
        o = _make_orchestrator({"kqapro_agent": _FakeAgent([{"ts": 1.0, "state": {}}])})
        assert o.journal_snapshots == []
        assert len(o.live_journal_snapshots()) == 1

    def test_later_appends_are_picked_up_on_the_next_call(self):
        agent = _FakeAgent([{"ts": 1.0, "state": {}}])
        o = _make_orchestrator({"kqapro_agent": agent})
        assert len(o.live_journal_snapshots()) == 1
        agent.journal_snapshots.append({"ts": 2.0, "state": {}})
        assert len(o.live_journal_snapshots()) == 2

    def test_agent_without_journal_snapshots_is_skipped(self):
        class _Bare:
            pass

        o = _make_orchestrator({
            "kqapro_agent": _Bare(),
            "sciqa_agent": _FakeAgent([{"ts": 1.0, "state": {}}]),
        })
        assert len(o.live_journal_snapshots()) == 1

    def test_non_dict_snapshots_are_skipped(self):
        o = _make_orchestrator({"kqapro_agent": _FakeAgent(["junk", None, {"ts": 1.0}])})
        assert o.live_journal_snapshots() == [{"ts": 1.0, "source_agent": "kqapro_agent"}]

    def test_a_raising_agent_does_not_take_down_the_caller(self):
        class _Exploding:
            @property
            def journal_snapshots(self):
                raise RuntimeError("boom")

        o = _make_orchestrator({
            "kqapro_agent": _Exploding(),
            "sciqa_agent": _FakeAgent([{"ts": 1.0, "state": {}}]),
        })
        assert len(o.live_journal_snapshots()) == 1

    def test_missing_agents_attribute_does_not_raise(self):
        o = Orchestrator.__new__(Orchestrator)
        assert o.live_journal_snapshots() == []

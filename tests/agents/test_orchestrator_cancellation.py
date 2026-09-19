"""Cooperative cancellation on the Orchestrator path.

The Orchestrator is a separate implementation with no tool loop, so its
checkpoints bracket the billed steps instead: routing, dispatch, and fusion.
The federated fan-out needs the token in BOTH branches — each specialist runs
in its own asyncio task and must unwind through its own code so its MCP
teardown stays in the task that opened it.

No network calls, no API keys, no MCP subprocess.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ama_kbqa.agents.orchestrator_agent.agent import Orchestrator
from ama_kbqa.framework.cancellation import CANCELLED_ANSWER, CancellationToken
from ama_kbqa.framework.trace import TraceRecorder


def _run(coro):
    return asyncio.run(coro)


class _FakeMcp:
    def __init__(self):
        self.closed = False

    async def call_tool(self, name, args):
        return "{}"

    async def close(self):
        self.closed = True


class _FakeAgent:
    """Sub-agent stub matching the surface `_run_specialist` touches."""

    def __init__(self, name, answer="answer", on_ask=None, error=None):
        self.name = name
        self._answer = answer
        self._on_ask = on_ask
        self._error = error
        self.journal_snapshots = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.recorder = None
        self._parent_span_id_override = None
        self.ask_kwargs = None
        self.close_tasks = []

    async def ask(self, query, cancel_token=None):
        self.ask_kwargs = {"cancel_token": cancel_token}
        if self._on_ask is not None:
            self._on_ask()
        await asyncio.sleep(0)
        if self._error is not None:
            raise self._error
        return self._answer

    async def close(self):
        self.close_tasks.append(asyncio.current_task())


class _TokenlessAgent(_FakeAgent):
    """A sub-agent whose ask() predates the token, to prove the kwarg is only
    passed when a token is actually in play."""

    async def ask(self, query):
        self.ask_kwargs = {}
        return self._answer


def _make_orchestrator(mcp=None):
    """Instantiate Orchestrator without running __init__ (mirrors the pattern
    in test_orchestrator_federation_mode.py)."""
    from chatkit import TransientRetry

    o = Orchestrator.__new__(Orchestrator)
    o.name = "TEST_ORCHESTRATOR"
    o.mcp = mcp if mcp is not None else _FakeMcp()
    o.client = MagicMock()
    o.model = "test-model"
    o.last_routing_reason = None
    o.last_routing_evidence = None
    o.recorder = TraceRecorder()
    o.journal_snapshots = []
    o.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    o._agents = {}
    o._retry = TransientRetry()
    o._agent_config = {
        "kqapro_agent": {
            "module": "ama_kbqa.agents.kqapro_agent.agent",
            "class": "KQAProAgent",
            "description": "General world knowledge.",
        },
        "sciqa_agent": {
            "module": "ama_kbqa.agents.sciqa_agent.agent",
            "class": "SciQAAgent",
            "description": "Scholarly knowledge.",
        },
    }

    async def _noop_init_mcp():
        return None

    o._init_mcp = _noop_init_mcp
    return o


def _cancelled(reason=""):
    token = CancellationToken()
    token.cancel(reason)
    return token


# ---------------------------------------------------------------------------
# ask() checkpoints
# ---------------------------------------------------------------------------

class TestAskCheckpoints:

    def test_cancelled_before_routing_skips_the_routing_llm_call(self):
        o = _make_orchestrator()

        async def _must_not_route(query):
            raise AssertionError("routing ran on a cancelled run")

        o._route_autonomously = _must_not_route

        answer = _run(o.ask("Who directed Inception?", cancel_token=_cancelled()))

        assert answer == CANCELLED_ANSWER
        # The orchestrator's own MCP connection was still closed, on this task.
        assert o.mcp.closed is True

    def test_cancelled_during_routing_skips_dispatch(self):
        o = _make_orchestrator()
        token = CancellationToken()

        async def _route(query):
            token.cancel("user pressed stop")
            return ["kqapro_agent"]

        async def _must_not_dispatch(*args, **kwargs):
            raise AssertionError("dispatch ran on a cancelled run")

        o._route_autonomously = _route
        o._delegate = _must_not_dispatch
        o._federate = _must_not_dispatch
        o._fallback_kqapro = _must_not_dispatch

        answer = _run(o.ask("Who directed Inception?", cancel_token=token))

        assert answer.startswith(CANCELLED_ANSWER)
        assert o.mcp.closed is True

    def test_without_a_token_ask_dispatches_normally(self):
        o = _make_orchestrator()
        dispatched = []

        async def _route(query):
            return ["kqapro_agent"]

        async def _delegate(agent_name, query, cancel_token=None):
            dispatched.append((agent_name, cancel_token))
            return "the answer"

        o._route_autonomously = _route
        o._delegate = _delegate

        assert _run(o.ask("Who directed Inception?")) == "the answer"
        assert dispatched == [("kqapro_agent", None)]


# ---------------------------------------------------------------------------
# _run_specialist
# ---------------------------------------------------------------------------

class TestRunSpecialist:

    def test_cancelled_token_skips_loading_and_running_the_specialist(self):
        o = _make_orchestrator()

        def _must_not_load(name):
            raise AssertionError("specialist was loaded on a cancelled run")

        o._load_agent = _must_not_load

        record = _run(
            o._run_specialist("kqapro_agent", "q", cancel_token=_cancelled("stopped"))
        )

        assert record["agent"] == "kqapro_agent"
        assert record["answer"].startswith(CANCELLED_ANSWER)
        assert record["scratchpad"] is None

    def test_token_is_forwarded_to_the_sub_agent(self):
        o = _make_orchestrator()
        agent = _FakeAgent("kqapro_agent")
        o._load_agent = lambda name: agent
        token = CancellationToken()

        _run(o._run_specialist("kqapro_agent", "q", cancel_token=token))

        assert agent.ask_kwargs == {"cancel_token": token}

    def test_no_kwarg_is_passed_when_there_is_no_token(self):
        o = _make_orchestrator()
        agent = _TokenlessAgent("kqapro_agent")
        o._load_agent = lambda name: agent

        result = _run(o._run_specialist("kqapro_agent", "q"))

        assert result["answer"] == "answer"
        assert agent.ask_kwargs == {}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatchCancellation:

    def test_delegate_does_not_fall_back_to_kqapro_when_cancelled(self):
        o = _make_orchestrator()
        token = CancellationToken()

        async def _run_specialist(agent_name, query, cancel_token=None, **kwargs):
            token.cancel()
            raise RuntimeError("specialist unwound")

        async def _must_not_fall_back(*args, **kwargs):
            raise AssertionError("fallback ran on a cancelled run")

        o._run_specialist = _run_specialist
        o._fallback_kqapro = _must_not_fall_back

        answer = _run(o._delegate("sciqa_agent", "q", cancel_token=token))

        assert answer == CANCELLED_ANSWER

    def test_federated_fan_out_carries_the_token_into_both_branches(self):
        o = _make_orchestrator()
        token = CancellationToken()
        # Both specialists are already running when the cancellation lands
        # (the second one flips the token), so both branches must have carried
        # the token in and both must unwind through their own task.
        agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="A"),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="B", on_ask=token.cancel),
        }
        o._load_agent = lambda name: agents[name]

        async def _must_not_fuse(*args, **kwargs):
            raise AssertionError("fusion ran on a cancelled run")

        o._fuse_answers = _must_not_fuse

        answer = _run(
            o._federate(["kqapro_agent", "sciqa_agent"], "q", cancel_token=token)
        )

        assert answer == CANCELLED_ANSWER
        # Both specialists saw the token ...
        for agent in agents.values():
            assert agent.ask_kwargs == {"cancel_token": token}
            # ... and each closed its MCP inside its own task, which is what
            # keeps anyio's cancel scope in the task that opened it.
            assert len(agent.close_tasks) == 1
        assert agents["kqapro_agent"].close_tasks != agents["sciqa_agent"].close_tasks

    def test_cancelling_inside_the_first_branch_skips_the_second_specialist(self):
        """The fan-out's second branch checks the token before it starts, so a
        cancellation during the first specialist saves a whole agent run."""
        o = _make_orchestrator()
        token = CancellationToken()
        agents = {
            "kqapro_agent": _FakeAgent("kqapro_agent", answer="A", on_ask=token.cancel),
            "sciqa_agent": _FakeAgent("sciqa_agent", answer="B"),
        }
        o._load_agent = lambda name: agents[name]

        async def _must_not_fuse(*args, **kwargs):
            raise AssertionError("fusion ran on a cancelled run")

        o._fuse_answers = _must_not_fuse

        answer = _run(
            o._federate(["kqapro_agent", "sciqa_agent"], "q", cancel_token=token)
        )

        assert answer == CANCELLED_ANSWER
        assert agents["kqapro_agent"].ask_kwargs == {"cancel_token": token}
        assert len(agents["kqapro_agent"].close_tasks) == 1
        # The second specialist never ran at all.
        assert agents["sciqa_agent"].ask_kwargs is None
        assert agents["sciqa_agent"].close_tasks == []

    def test_cancelling_before_the_fan_out_skips_every_specialist(self):
        o = _make_orchestrator()

        def _must_not_load(name):
            raise AssertionError("specialist was loaded on a cancelled run")

        o._load_agent = _must_not_load

        async def _must_not_fuse(*args, **kwargs):
            raise AssertionError("fusion ran on a cancelled run")

        o._fuse_answers = _must_not_fuse

        answer = _run(
            o._federate(["kqapro_agent", "sciqa_agent"], "q", cancel_token=_cancelled())
        )

        assert answer == CANCELLED_ANSWER

    def test_federation_without_a_token_still_fuses(self):
        o = _make_orchestrator()
        agents = {
            "kqapro_agent": _TokenlessAgent("kqapro_agent", answer="A"),
            "sciqa_agent": _TokenlessAgent("sciqa_agent", answer="B"),
        }
        o._load_agent = lambda name: agents[name]
        fused = []

        async def _fuse(query, answers):
            fused.append([a["answer"] for a in answers])
            return "fused"

        o._fuse_answers = _fuse

        answer = _run(o._federate(["kqapro_agent", "sciqa_agent"], "q"))

        assert answer == "fused"
        assert fused == [["A", "B"]]


@pytest.mark.parametrize(
    "method", ["ask", "_run_specialist", "_delegate", "_federate", "_fallback_kqapro"]
)
def test_cancel_token_is_optional_on_every_orchestrator_entry_point(method):
    import inspect

    param = inspect.signature(getattr(Orchestrator, method)).parameters["cancel_token"]
    assert param.default is None

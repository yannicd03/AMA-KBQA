"""Smoke tests for the background lifecycle runner."""

from __future__ import annotations

import queue
import threading
import time

from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    live_graph_snapshot,
    make_listener,
    orchestrator_visited_node_ids,
    reconstruct_orchestrator,
    start_run,
)
from ama_kbqa.frontend.utils.live_graph_data import GraphData, GraphNode
from ama_kbqa.server import kqapro_server as kqa


class _StubAgent:
    """Minimal agent that emits the canonical span set then returns an answer."""

    def __init__(self, *, fail: bool = False) -> None:
        self.recorder = TraceRecorder()
        self.fail = fail

    async def ask(self, question: str) -> str:
        async with self.recorder.span("agent_run", "ask"):
            async with self.recorder.span("classify", "classify"):
                pass
            async with self.recorder.span("llm_call", "gpt-4o"):
                async with self.recorder.span("tool_call", "FindNode"):
                    pass
                async with self.recorder.span("tool_call", "VerifyFact"):
                    pass
            self.recorder.event("loop_detected", "x")
            async with self.recorder.span("synthesis", "final"):
                pass
        if self.fail:
            raise RuntimeError("boom")
        return f"answer to: {question}"


class _StubMultiturnAgent:
    """Agent double that mimics the real history-preserving reset + ask so the
    multiturn continuation path can be driven end-to-end through start_run."""

    def __init__(self) -> None:
        self.recorder = TraceRecorder()
        self._messages = [{"role": "system", "content": "sys"}]
        self._catalog_injected = False
        self.mcp = object()  # pretend an MCP connection is open
        self.reset_calls: list[dict] = []

    async def reset(self, keep_mcp_open: bool = False, keep_history: bool = False) -> None:
        self.reset_calls.append({"keep_mcp_open": keep_mcp_open, "keep_history": keep_history})
        if not keep_history:
            self._messages = [{"role": "system", "content": "sys"}]
            self._catalog_injected = False
        self.recorder = TraceRecorder()  # fresh trace each turn
        if not keep_mcp_open and self.mcp is not None:
            self.mcp = None

    async def ask(self, question: str) -> str:
        if self.mcp is None:  # mimic _init_mcp rebuilding on this loop
            self.mcp = object()
        if not self._catalog_injected:  # one-time catalog injection
            self._messages.append({"role": "system", "content": "CATALOG"})
            self._catalog_injected = True
        self._messages.append({"role": "user", "content": question})
        async with self.recorder.span("agent_run", "ask"):
            pass
        answer = f"answer to: {question}"
        self._messages.append({"role": "assistant", "content": answer})
        return answer


def _wait_until_terminal(q: queue.Queue, state: LiveLifecycleState, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if drain_into(q, state):
            return
        time.sleep(0.02)
    raise AssertionError(f"runner did not terminate; state={state.status}")


# Real event stream captured from a deployed KQAPro run of
# "Who is the director of Inception?" (fast-path attempt that fails and falls
# back to a 3-iteration loop). Each entry is (phase, kind, name, span_id,
# is_event). span_id pairs open/close; events carry "".
_GROUND_TRUTH = [
    ("open",  "agent_run",      "kqapro_agent",     "ar", False),
    ("open",  "classify",       "kit.gpt-oss-120b", "cl", False),
    ("close", "classify",       "kit.gpt-oss-120b", "cl", False),
    ("open",  "fast_path",      "QueryAttr",        "fp", False),
    ("open",  "tool_call",      "FindNode",         "t1", False),
    ("close", "tool_call",      "FindNode",         "t1", False),
    ("open",  "tool_call",      "GetAttributeDetails", "t2", False),
    ("close", "tool_call",      "GetAttributeDetails", "t2", False),
    ("open",  "tool_call",      "GetNodeSummary",   "t3", False),
    ("close", "tool_call",      "GetNodeSummary",   "t3", False),
    ("close", "fast_path",      "QueryAttr",        "fp", False),  # <-- fast-path FAILS here
    ("event", "tool_loop_iter", "iter:1",           "",   True),
    ("open",  "llm_call",       "kit.gpt-oss-120b", "l1", False),
    ("close", "llm_call",       "kit.gpt-oss-120b", "l1", False),
    ("open",  "tool_call",      "FindNode",         "t4", False),
    ("close", "tool_call",      "FindNode",         "t4", False),
    ("event", "tool_loop_iter", "iter:2",           "",   True),
    ("open",  "llm_call",       "kit.gpt-oss-120b", "l2", False),
    ("close", "llm_call",       "kit.gpt-oss-120b", "l2", False),
    ("open",  "tool_call",      "GetJournalSummary","t5", False),
    ("close", "tool_call",      "GetJournalSummary","t5", False),
    ("event", "tool_loop_iter", "iter:3",           "",   True),
    ("open",  "llm_call",       "kit.gpt-oss-120b", "l3", False),
    ("close", "llm_call",       "kit.gpt-oss-120b", "l3", False),
    ("close", "agent_run",      "kqapro_agent",     "ar", False),
]


def _drain_events(state: LiveLifecycleState, events: list) -> None:
    q: queue.Queue = queue.Queue()
    for phase, kind, name, sid, is_ev in events:
        q.put((phase, {
            "kind": kind, "name": name, "span_id": sid,
            "parent_span_id": None, "status": "ok",
            "is_event": is_ev, "attributes": {},
        }))
    drain_into(q, state)


class TestGroundTruthReplay:
    """Replays a real captured agent run to lock in the two reported figure
    bugs: scratchpad must light during the run, and Answer Synthesis must not
    flash mid-run (only settle at the end)."""

    def _split_at_fastpath_close(self):
        idx = next(i for i, e in enumerate(_GROUND_TRUTH)
                   if e[0] == "close" and e[1] == "fast_path")
        return _GROUND_TRUTH[: idx + 1], _GROUND_TRUTH[idx + 1:]

    def test_synthesis_does_not_flash_when_fastpath_fails(self):
        # Drain everything up to and including the fast-path close. The old
        # mapping lit post_synthesis here (the "randomly lights up" bug).
        through_fastpath, _rest = self._split_at_fastpath_close()
        state = LiveLifecycleState()
        _drain_events(state, through_fastpath)
        assert "post_synthesis" not in state.render_active_node_ids()

    def test_scratchpad_lights_during_the_run(self):
        # By the time the first KG tool calls have closed, the scratchpad must
        # have lit (it never did before — the core complaint).
        through_fastpath, _rest = self._split_at_fastpath_close()
        state = LiveLifecycleState()
        _drain_events(state, through_fastpath)
        assert "main_scratchpad" in state.render_active_node_ids()
        assert "main_scratchpad" in state.visited_node_ids

    def test_synthesis_settles_only_at_the_end(self):
        state = LiveLifecycleState()
        _drain_events(state, _GROUND_TRUTH)
        # agent_run close flashes the full post band, including Answer Synthesis.
        assert "post_synthesis" in state.render_active_node_ids()


class TestRunner:
    def test_successful_run_transitions_to_done(self):
        agent = _StubAgent()
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        t = start_run(agent=agent, question="hi", q=q)
        assert isinstance(t, threading.Thread)

        _wait_until_terminal(q, state)

        assert state.status == "done"
        assert state.answer == "answer to: hi"
        # Tool calls (traversal + summary) all light the single Tool Call box.
        assert "main_tool_call" in state.visited_node_ids
        assert "main_llm_reason" in state.visited_node_ids
        assert "post_synthesis" in state.visited_node_ids
        assert "agent_invocation" in state.visited_node_ids
        # No spans should be active after completion.
        assert state.active_node_ids == set()

    def test_failing_run_transitions_to_error(self):
        agent = _StubAgent(fail=True)
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        start_run(agent=agent, question="hi", q=q)

        _wait_until_terminal(q, state)

        assert state.status == "error"
        assert state.error is not None and "boom" in state.error
        assert state.active_node_ids == set()

    def test_continuation_preserves_and_accumulates_history(self):
        agent = _StubMultiturnAgent()

        # Turn 1: fresh conversation, no reset.
        q1: queue.Queue = queue.Queue()
        s1 = LiveLifecycleState()
        start_run(
            agent=agent,
            question="Who directed Inception?",
            q=q1,
            is_continuation=False,
        )
        _wait_until_terminal(q1, s1)

        assert s1.status == "done"
        rec_after_t1 = agent.recorder
        # Live trace_id was emitted to the main thread via the queue.
        assert s1.trace_id == rec_after_t1.trace_id
        assert agent.reset_calls == []  # first turn never resets
        assert sum(1 for m in agent._messages if m["content"] == "CATALOG") == 1

        # Turn 2: continuation on the SAME instance.
        q2: queue.Queue = queue.Queue()
        s2 = LiveLifecycleState()
        start_run(
            agent=agent,
            question="Where was he born?",
            q=q2,
            is_continuation=True,
        )
        _wait_until_terminal(q2, s2)

        assert s2.status == "done"
        # Reset ran once, preserving history and not closing MCP across loops.
        assert agent.reset_calls == [{"keep_mcp_open": True, "keep_history": True}]
        # MCP was orphaned then rebuilt.
        assert agent.mcp is not None
        # Fresh recorder + trace_id propagated to the live panel.
        assert agent.recorder is not rec_after_t1
        assert s2.trace_id == agent.recorder.trace_id
        assert s2.trace_id != s1.trace_id
        # History ACCUMULATED across turns.
        contents = [m["content"] for m in agent._messages]
        assert "Who directed Inception?" in contents
        assert "Where was he born?" in contents
        # Catalog was NOT duplicated onto the preserved stack.
        assert sum(1 for m in agent._messages if m["content"] == "CATALOG") == 1

    def _emit(self, q, phase, kind, name, *, span_id="s1", is_event=False, attributes=None):
        q.put((phase, {
            "kind": kind,
            "name": name,
            "span_id": span_id,
            "parent_span_id": None,
            "status": "ok",
            "is_event": is_event,
            "attributes": attributes or {},
        }))

    def test_minimum_lightup_holds_node_after_span_closes(self):
        # A span that opens and closes within a single drain (faster than the
        # ~0.4s UI tick) must still render active for at least the hold window.
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._emit(q, "open", "llm_call", "gpt-4o", span_id="s1")
        self._emit(q, "close", "llm_call", "gpt-4o", span_id="s1")
        drain_into(q, state)

        # The span is off the live stack immediately...
        assert "main_llm_reason" not in state.active_node_ids
        # ...but the hold keeps it in the *rendered* active set.
        until = state._node_lit_until["main_llm_reason"]
        assert "main_llm_reason" in state.render_active_node_ids(until - 0.01)
        # Once the window elapses it drops out.
        assert "main_llm_reason" not in state.render_active_node_ids(until + 0.01)

    def test_held_node_unions_with_live_stack(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        # A still-open span (agent_invocation) plus a closed-but-held one.
        self._emit(q, "open", "agent_run", "ask", span_id="root")
        self._emit(q, "open", "llm_call", "gpt-4o", span_id="s1")
        self._emit(q, "close", "llm_call", "gpt-4o", span_id="s1")
        drain_into(q, state)

        rendered = state.render_active_node_ids(state._node_lit_until["main_llm_reason"] - 0.01)
        assert "agent_invocation" in rendered  # live on the stack
        assert "main_llm_reason" in rendered    # held after close

    def test_new_cycle_lights_loop_back_edge(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        # First iteration: no loop-back.
        self._emit(q, "event", "tool_loop_iter", "iter:1", is_event=True,
                   attributes={"iteration": 1})
        drain_into(q, state)
        assert state.render_active_edge_ids() == set()

        # Second iteration: a new ReAct cycle lights the loop-back arrow.
        self._emit(q, "event", "tool_loop_iter", "iter:2", is_event=True,
                   attributes={"iteration": 2})
        drain_into(q, state)
        until = state._edge_lit_until["loop_back"]
        assert "loop_back" in state.render_active_edge_ids(until - 0.01)
        assert "loop_back" not in state.render_active_edge_ids(until + 0.01)

    def _put(self, q, phase, kind, name, span_id, parent=None,
             attrs=None, is_event=False, status="ok"):
        q.put((phase, {
            "kind": kind, "name": name, "span_id": span_id,
            "parent_span_id": parent, "status": status,
            "is_event": is_event, "attributes": attrs or {},
        }))

    def _drive_orchestrator_until_delegate_open(self, q, state):
        """Emit the orchestrator span prefix up to (and including) the open of
        the kqapro delegate + sub-agent root + an llm_call — i.e. mid-run."""
        self._put(q, "open", "agent_run", "ORCHESTRATOR", "root")
        self._put(q, "open", "classify", "route", "route1", "root")
        self._put(q, "close", "classify", "route", "route1", "root")
        self._put(q, "open", "delegate", "kqapro_agent", "deleg", "root",
                  attrs={"sub_agent": "kqapro_agent"})
        self._put(q, "open", "agent_run", "kqapro_agent", "subroot", "deleg")
        self._put(q, "open", "llm_call", "gpt-4o", "llm", "subroot")
        drain_into(q, state)

    def test_delegate_lights_container_and_creates_subagent(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)

        # The dispatched specialist is registered and running.
        assert "kqapro_agent" in state.subagents
        sub = state.subagents["kqapro_agent"]
        assert sub.status == "running"
        assert sub.display == "KQAPro" and sub.container_id == "sub_kqapro"
        # Its container is on the orchestrator stack → active while delegating.
        assert "sub_kqapro" in state.orch.active_node_ids
        # User Query stays lit for the whole run (root span still open).
        assert "orch_user" in state.orch.active_node_ids

    def test_subagent_internal_spans_drive_detail_figure_not_orchestrator(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        sub = state.subagents["kqapro_agent"]

        # The sub-agent's own spans light ITS detail figure …
        assert "agent_invocation" in sub.visited_node_ids   # its agent_run open
        assert "main_llm_reason" in sub.visited_node_ids     # its llm_call
        # … and never leak into the orchestrator figure's node set.
        assert "main_llm_reason" not in state.orch.visited_node_ids
        assert "agent_invocation" not in state.orch.visited_node_ids

    def test_running_subagent_mini_tracks_current_phase(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        # llm_call is open → main phase in flight → the Main mini lights.
        active = state.render_orchestrator_active_node_ids()
        assert "sub_kqapro_main" in active
        assert "sub_kqapro" in active

    def test_only_dispatched_specialist_appears(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        # SciQA was never dispatched: no detail figure, container stays idle.
        assert "sciqa_agent" not in state.subagents
        assert "sub_sciqa" not in state.render_orchestrator_visited_node_ids()

    def test_delegate_close_marks_subagent_done(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        # Close the sub-agent and the delegate.
        self._put(q, "close", "llm_call", "gpt-4o", "llm", "subroot")
        self._put(q, "close", "agent_run", "kqapro_agent", "subroot", "deleg")
        self._put(q, "close", "delegate", "kqapro_agent", "deleg", "root")
        drain_into(q, state)

        sub = state.subagents["kqapro_agent"]
        assert sub.status == "done"
        # Container is no longer on the orchestrator stack (delegate closed).
        assert "sub_kqapro" not in state.orch.active_node_ids
        # But it remains in the visited set for the figure.
        assert "sub_kqapro" in state.render_orchestrator_visited_node_ids()

    def test_error_marks_running_subagents_error(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        q.put(("__error__", {"message": "RuntimeError: boom"}))
        assert drain_into(q, state) is True
        assert state.status == "error"
        assert state.subagents["kqapro_agent"].status == "error"

    def test_reconstruct_orchestrator_from_events(self):
        events = [
            {"kind": "agent_run", "name": "ORCHESTRATOR", "span_id": "root",
             "parent_span_id": None, "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "classify", "name": "route", "span_id": "r1",
             "parent_span_id": "root", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "delegate", "name": "sciqa_agent", "span_id": "d1",
             "parent_span_id": "root", "is_event": False,
             "attributes": {"sub_agent": "sciqa_agent"}, "status": "ok"},
            {"kind": "agent_run", "name": "sciqa_agent", "span_id": "sr",
             "parent_span_id": "d1", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "llm_call", "name": "gpt-4o", "span_id": "l1",
             "parent_span_id": "sr", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "tool_call", "name": "FindResource", "span_id": "t1",
             "parent_span_id": "l1", "is_event": False, "attributes": {}, "status": "ok"},
        ]
        orch_visited, subs = reconstruct_orchestrator(events)

        # Orchestrator-level nodes reconstructed.
        assert {"orch_user", "orch_probe", "orch_dispatch", "orch_combine",
                "sub_sciqa"} <= orch_visited
        # The dispatched specialist (SciQA) reconstructed with its lifecycle.
        assert list(subs.keys()) == ["sciqa_agent"]
        sub = subs["sciqa_agent"]
        assert sub.display == "SciQA"
        assert {"agent_invocation", "main_llm_reason", "main_tool_call"} <= sub.visited_node_ids

        # The combined figure visited set includes the container + minis.
        full = orchestrator_visited_node_ids(orch_visited, subs)
        assert "sub_sciqa" in full and "sub_sciqa_main" in full

    def test_reconstruct_empty_trace_is_safe(self):
        orch_visited, subs = reconstruct_orchestrator([])
        assert orch_visited == set()
        assert subs == {}

    # -- Federated dispatch: two interleaved sibling delegate spans ---------
    #
    # Mirrors `Orchestrator._federate` (feature/federated-retrieval @
    # 98882ac): both specialists are dispatched via asyncio.gather, so their
    # `delegate` spans open and interleave before either closes, and a
    # `synthesis`/`fuse` span (kind="synthesis", name="fuse") runs after both
    # return. Delegate span attributes carry `sub_agent`; the fuse span
    # carries `agents` (comma-joined names) — see agent.py:_run_specialist /
    # _fuse_answers on that branch.

    def _drive_federated_dispatch(self, q, state):
        """Open both delegates (interleaved), open+close both sub-agents'
        internal spans, close both delegates, then open the fuse span."""
        self._put(q, "open", "agent_run", "ORCHESTRATOR", "root")
        self._put(q, "open", "classify", "route", "route1", "root")
        self._put(q, "close", "classify", "route", "route1", "root",
                  attrs={"selected_agent": "kqapro_agent, sciqa_agent",
                         "route_mode": "federated"})
        # Both delegates open before either closes (concurrent gather).
        self._put(q, "open", "delegate", "kqapro_agent", "deleg_k", "root",
                  attrs={"sub_agent": "kqapro_agent"})
        self._put(q, "open", "delegate", "sciqa_agent", "deleg_s", "root",
                  attrs={"sub_agent": "sciqa_agent"})
        self._put(q, "open", "agent_run", "kqapro_agent", "subroot_k", "deleg_k")
        self._put(q, "open", "agent_run", "sciqa_agent", "subroot_s", "deleg_s")
        self._put(q, "open", "llm_call", "gpt-4o", "llm_k", "subroot_k")
        self._put(q, "open", "llm_call", "gpt-4o", "llm_s", "subroot_s")
        drain_into(q, state)

    def test_two_interleaved_delegate_spans_create_two_active_panes(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_federated_dispatch(q, state)

        assert set(state.subagents.keys()) == {"kqapro_agent", "sciqa_agent"}
        kqapro = state.subagents["kqapro_agent"]
        sciqa = state.subagents["sciqa_agent"]
        assert kqapro.status == "running"
        assert sciqa.status == "running"

        # Both containers active in the orchestrator figure at once.
        assert "sub_kqapro" in state.orch.active_node_ids
        assert "sub_sciqa" in state.orch.active_node_ids

        # Each sub-agent's own spans drove only ITS detail figure (correct
        # attribution via parent_span_id → nearest delegate ancestor).
        assert "main_llm_reason" in kqapro.visited_node_ids
        assert "main_llm_reason" in sciqa.visited_node_ids
        assert "main_llm_reason" not in state.orch.visited_node_ids

        active = state.render_orchestrator_active_node_ids()
        assert "sub_kqapro_main" in active
        assert "sub_sciqa_main" in active

    def test_fusion_span_lights_answer_combination_on_open(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_federated_dispatch(q, state)
        self._put(q, "close", "llm_call", "gpt-4o", "llm_k", "subroot_k")
        self._put(q, "close", "llm_call", "gpt-4o", "llm_s", "subroot_s")
        self._put(q, "close", "agent_run", "kqapro_agent", "subroot_k", "deleg_k")
        self._put(q, "close", "agent_run", "sciqa_agent", "subroot_s", "deleg_s")
        self._put(q, "close", "delegate", "kqapro_agent", "deleg_k", "root")
        self._put(q, "close", "delegate", "sciqa_agent", "deleg_s", "root")
        drain_into(q, state)

        assert state.subagents["kqapro_agent"].status == "done"
        assert state.subagents["sciqa_agent"].status == "done"
        assert "orch_combine" not in state.orch.active_node_ids

        self._put(q, "open", "synthesis", "fuse", "fuse1", "root",
                  attrs={"model": "gpt-4o", "agents": "kqapro_agent, sciqa_agent"})
        drain_into(q, state)

        assert "orch_combine" in state.orch.render_active_node_ids()
        assert "orch_combine" in state.render_orchestrator_visited_node_ids()

    def test_redispatch_of_same_agent_resets_pane_to_running(self):
        """A second `delegate` open for an agent that already has a pane
        (e.g. a KQAPro fallback re-running after an earlier dispatch) must
        reopen that pane as live instead of leaving it frozen done/error."""
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._drive_orchestrator_until_delegate_open(q, state)
        # Close out the first KQAPro dispatch.
        self._put(q, "close", "llm_call", "gpt-4o", "llm", "subroot")
        self._put(q, "close", "agent_run", "kqapro_agent", "subroot", "deleg")
        self._put(q, "close", "delegate", "kqapro_agent", "deleg", "root")
        drain_into(q, state)
        assert state.subagents["kqapro_agent"].status == "done"

        # Re-dispatch the SAME agent (e.g. fallback retry).
        self._put(q, "open", "delegate", "kqapro_agent", "deleg2", "root",
                  attrs={"sub_agent": "kqapro_agent"})
        drain_into(q, state)

        # Same pane object, reopened as running (not a duplicate).
        assert set(state.subagents.keys()) == {"kqapro_agent"}
        assert state.subagents["kqapro_agent"].status == "running"

        # New activity on the re-dispatch lights its detail figure again.
        self._put(q, "open", "agent_run", "kqapro_agent", "subroot2", "deleg2")
        self._put(q, "open", "llm_call", "gpt-4o", "llm2", "subroot2")
        drain_into(q, state)
        assert "sub_kqapro" in state.orch.active_node_ids
        assert state.subagents["kqapro_agent"].status == "running"

    def test_reconstruct_orchestrator_with_two_delegates_and_fusion(self):
        events = [
            {"kind": "agent_run", "name": "ORCHESTRATOR", "span_id": "root",
             "parent_span_id": None, "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "classify", "name": "route", "span_id": "r1",
             "parent_span_id": "root", "is_event": False,
             "attributes": {"route_mode": "federated"}, "status": "ok"},
            {"kind": "delegate", "name": "kqapro_agent", "span_id": "dk",
             "parent_span_id": "root", "is_event": False,
             "attributes": {"sub_agent": "kqapro_agent"}, "status": "ok"},
            {"kind": "delegate", "name": "sciqa_agent", "span_id": "ds",
             "parent_span_id": "root", "is_event": False,
             "attributes": {"sub_agent": "sciqa_agent"}, "status": "ok"},
            {"kind": "agent_run", "name": "kqapro_agent", "span_id": "sk",
             "parent_span_id": "dk", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "llm_call", "name": "gpt-4o", "span_id": "lk",
             "parent_span_id": "sk", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "agent_run", "name": "sciqa_agent", "span_id": "ss",
             "parent_span_id": "ds", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "llm_call", "name": "gpt-4o", "span_id": "ls",
             "parent_span_id": "ss", "is_event": False, "attributes": {}, "status": "ok"},
            {"kind": "synthesis", "name": "fuse", "span_id": "fu",
             "parent_span_id": "root", "is_event": False,
             "attributes": {"agents": "kqapro_agent, sciqa_agent"}, "status": "ok"},
        ]
        orch_visited, subs = reconstruct_orchestrator(events)

        assert {"orch_user", "orch_probe", "orch_dispatch", "orch_combine",
                "sub_kqapro", "sub_sciqa"} <= orch_visited
        assert set(subs.keys()) == {"kqapro_agent", "sciqa_agent"}
        assert subs["kqapro_agent"].display == "KQAPro"
        assert subs["sciqa_agent"].display == "SciQA"
        assert "main_llm_reason" in subs["kqapro_agent"].visited_node_ids
        assert "main_llm_reason" in subs["sciqa_agent"].visited_node_ids

        full = orchestrator_visited_node_ids(orch_visited, subs)
        assert "sub_kqapro_main" in full and "sub_sciqa_main" in full

    def test_drain_into_is_idempotent_after_terminal(self):
        agent = _StubAgent()
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        start_run(agent=agent, question="hi", q=q)

        _wait_until_terminal(q, state)
        snapshot_active = set(state.active_node_ids)
        snapshot_visited = set(state.visited_node_ids)

        # Subsequent drains on an empty queue must not mutate state.
        assert drain_into(q, state) is True
        assert state.active_node_ids == snapshot_active
        assert state.visited_node_ids == snapshot_visited


# ---------------------------------------------------------------------------
# Live graph deltas
# ---------------------------------------------------------------------------
#
# The graph half of the runner has two jobs the lifecycle half does not:
# attributing a tool result to the specialist that produced it (so a federated
# run can colour each graph's nodes), and telling the panel when NOT to
# redraw. Both are invisible in the UI when they go wrong (wrong colour, or a
# 1 Hz canvas remount), so they are pinned here.


def _search_result_json(node_id: str = "Q937", name: str = "Albert Einstein") -> str:
    """A SearchResponse payload, built from the real server model so a schema
    change breaks this test instead of silently emptying the panel."""
    return kqa.SearchResponse(
        matches=[
            kqa.NodeMatch(
                original_id=node_id,
                name=name,
                node_type="entity",
                relevance_score=0.91,
            )
        ],
        result_count=1,
    ).model_dump_json()


class _FakeJournalAgent:
    """Single-agent double: just the append-only list the runner reads."""

    def __init__(self, snapshots: list) -> None:
        self.journal_snapshots = snapshots


class _FakeOrchestratorAgent:
    """Orchestrator double exposing the tagged live accessor (agent.py:536)."""

    def __init__(self, snapshots: list) -> None:
        self._snapshots = snapshots

    def live_journal_snapshots(self) -> list:
        return list(self._snapshots)


def _graph_delta(node_id: str, label: str) -> GraphData:
    """A one-node delta, as the worker thread would hand it over (source "")."""
    return GraphData(nodes={node_id: GraphNode(id=node_id, label=label)})


class TestGraphDeltaAttribution:
    def _put(self, q, phase, kind, name, span_id, parent=None, attrs=None):
        q.put((phase, {
            "kind": kind, "name": name, "span_id": span_id,
            "parent_span_id": parent, "status": "ok",
            "is_event": False, "attributes": attrs or {},
        }))

    def _put_graph(self, q, span_id, parent, graph, tool="FindNode"):
        q.put(("__graph__", {
            "span_id": span_id, "parent_span_id": parent,
            "tool": tool, "graph": graph,
        }))

    def test_single_agent_delta_uses_the_run_wide_source(self):
        # No delegate span: the owner is None and the source comes from the
        # agent the user picked in the sidebar.
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState(graph_source_default="kqapro")
        self._put(q, "open", "tool_call", "FindNode", "t1")
        self._put(q, "close", "tool_call", "FindNode", "t1")
        self._put_graph(q, "t1", None, _graph_delta("Q937", "Einstein"))
        drain_into(q, state)

        assert set(state.graph.by_owner) == {None}
        node = state.graph.by_owner[None].nodes["Q937"]
        assert node.source == "kqapro"
        assert state.graph.version == 1

    def test_router_delta_is_attributed_to_the_delegated_specialist(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()  # orchestrator pick: no default source
        self._put(q, "open", "agent_run", "ORCHESTRATOR", "root")
        self._put(q, "open", "delegate", "kqapro_agent", "deleg", "root",
                  attrs={"sub_agent": "kqapro_agent"})
        self._put(q, "open", "agent_run", "kqapro_agent", "subroot", "deleg")
        self._put(q, "open", "tool_call", "FindNode", "t1", "subroot")
        self._put(q, "close", "tool_call", "FindNode", "t1", "subroot")
        self._put_graph(q, "t1", "subroot", _graph_delta("Q937", "Einstein"))
        drain_into(q, state)

        assert set(state.graph.by_owner) == {"kqapro_agent"}
        assert state.graph.by_owner["kqapro_agent"].nodes["Q937"].source == "kqapro"

    def test_federated_deltas_land_in_separate_owner_buckets(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._put(q, "open", "agent_run", "ORCHESTRATOR", "root")
        self._put(q, "open", "delegate", "kqapro_agent", "deleg_k", "root",
                  attrs={"sub_agent": "kqapro_agent"})
        self._put(q, "open", "delegate", "sciqa_agent", "deleg_s", "root",
                  attrs={"sub_agent": "sciqa_agent"})
        self._put(q, "open", "agent_run", "kqapro_agent", "sub_k", "deleg_k")
        self._put(q, "open", "agent_run", "sciqa_agent", "sub_s", "deleg_s")
        # Interleaved tool calls, one per specialist.
        self._put(q, "open", "tool_call", "FindNode", "tk", "sub_k")
        self._put(q, "open", "tool_call", "FindResource", "ts", "sub_s")
        self._put(q, "close", "tool_call", "FindNode", "tk", "sub_k")
        self._put_graph(q, "tk", "sub_k", _graph_delta("Q937", "Einstein"))
        self._put(q, "close", "tool_call", "FindResource", "ts", "sub_s")
        self._put_graph(q, "ts", "sub_s", _graph_delta("R123", "Deep Learning"),
                        tool="FindResource")
        drain_into(q, state)

        assert set(state.graph.by_owner) == {"kqapro_agent", "sciqa_agent"}
        assert state.graph.by_owner["kqapro_agent"].nodes["Q937"].source == "kqapro"
        assert state.graph.by_owner["sciqa_agent"].nodes["R123"].source == "sciqa"
        # The merged view is what the panel draws: both specialists at once.
        merged = state.graph.merged()
        assert set(merged.nodes) == {"Q937", "R123"}

    def test_version_bumps_only_on_new_content(self):
        # The agent re-checking a node it already looked at must not make the
        # panel remount its canvas.
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState(graph_source_default="kqapro")
        self._put_graph(q, "t1", None, _graph_delta("Q937", "Einstein"))
        drain_into(q, state)
        assert state.graph.version == 1

        self._put_graph(q, "t2", None, _graph_delta("Q937", "Einstein"))
        drain_into(q, state)
        assert state.graph.version == 1

        self._put_graph(q, "t3", None, _graph_delta("Q42", "Douglas Adams"))
        drain_into(q, state)
        assert state.graph.version == 2

    def test_empty_delta_is_ignored(self):
        q: queue.Queue = queue.Queue()
        state = LiveLifecycleState()
        self._put_graph(q, "t1", None, GraphData())
        drain_into(q, state)
        assert state.graph.by_owner == {}
        assert state.graph.version == 0


class TestMakeListener:
    """The worker-side half: which span closes turn into a graph delta."""

    def _drain(self, q: queue.Queue) -> list:
        items = []
        while True:
            try:
                items.append(q.get_nowait())
            except queue.Empty:
                return items

    def test_allow_listed_tool_close_enqueues_a_graph_delta(self):
        q: queue.Queue = queue.Queue()
        recorder = TraceRecorder()
        recorder.add_listener(make_listener(q))

        with recorder.span_sync(
            "tool_call", "FindNode",
            attributes={"tool_name": "FindNode"},
            payload={"arguments": {"semantic_node_name": "Einstein"}},
        ) as span:
            span.set_payload("result", _search_result_json())

        items = self._drain(q)
        phases = [phase for phase, _ in items]
        # The regular open/close notifications are unchanged and still first.
        assert phases == ["open", "close", "__graph__"]
        assert "payload" not in items[1][1]  # close is still stripped
        graph_info = items[2][1]
        assert graph_info["tool"] == "FindNode"
        assert graph_info["span_id"] == items[1][1]["span_id"]
        assert "Q937" in graph_info["graph"].nodes
        # The worker leaves the source blank; drain_into stamps it.
        assert graph_info["graph"].nodes["Q937"].source == ""

    def test_non_allow_listed_tool_enqueues_no_delta(self):
        q: queue.Queue = queue.Queue()
        recorder = TraceRecorder()
        recorder.add_listener(make_listener(q))

        with recorder.span_sync(
            "tool_call", "VerifyFact",
            attributes={"tool_name": "VerifyFact"},
            payload={"arguments": {}},
        ) as span:
            span.set_payload("result", _search_result_json())

        assert [phase for phase, _ in self._drain(q)] == ["open", "close"]

    def test_unparseable_result_is_swallowed(self):
        q: queue.Queue = queue.Queue()
        recorder = TraceRecorder()
        recorder.add_listener(make_listener(q))

        with recorder.span_sync(
            "tool_call", "FindNode",
            attributes={"tool_name": "FindNode"},
            payload={"arguments": {}},
        ) as span:
            span.set_payload("result", "Error: node not found")

        assert [phase for phase, _ in self._drain(q)] == ["open", "close"]


class TestLiveGraphSnapshot:
    _JOURNAL = {
        "visited_nodes": {"Q937": "Albert Einstein"},
        "verified_facts": [
            {"subject": "Q937", "relation": "occupation", "related_id": "Q169470"}
        ],
    }

    def test_merges_journal_state_with_tool_deltas(self):
        state = LiveLifecycleState(graph_source_default="kqapro")
        state.graph.by_owner[None] = _graph_delta("Q42", "Douglas Adams")
        agent = _FakeJournalAgent([{"trigger": "after:FindNode", "state": self._JOURNAL}])

        graph, key = live_graph_snapshot(state, agent)

        assert {"Q42", "Q937", "Q169470"} <= set(graph.nodes)
        assert graph.nodes["Q937"].source == "kqapro"
        assert key == (0, (("", 1),))

    def test_version_key_tracks_both_sources_of_change(self):
        state = LiveLifecycleState(graph_source_default="kqapro")
        agent = _FakeJournalAgent([])
        _graph, first = live_graph_snapshot(state, agent)

        agent.journal_snapshots.append({"state": self._JOURNAL})
        _graph, second = live_graph_snapshot(state, agent)
        assert second != first

        state.graph.by_owner[None] = _graph_delta("Q42", "Douglas Adams")
        state.graph.version += 1
        _graph, third = live_graph_snapshot(state, agent)
        assert third != second

    def test_orchestrator_uses_the_last_snapshot_per_specialist(self):
        state = LiveLifecycleState()  # orchestrator: no run-wide source
        agent = _FakeOrchestratorAgent([
            # An earlier, smaller snapshot of the same specialist is superseded.
            {"source_agent": "kqapro_agent", "state": {"visited_nodes": {"Q1": "old"}}},
            {"source_agent": "kqapro_agent", "state": self._JOURNAL},
            {"source_agent": "sciqa_agent",
             "state": {"visited_nodes": {"R123": "Deep Learning"}}},
        ])

        graph, key = live_graph_snapshot(state, agent)

        assert "Q1" not in graph.nodes           # superseded snapshot dropped
        assert graph.nodes["Q937"].source == "kqapro"
        assert graph.nodes["R123"].source == "sciqa"
        assert key == (0, (("kqapro_agent", 2), ("sciqa_agent", 1)))

    def test_agent_without_journal_attributes_never_raises(self):
        state = LiveLifecycleState()
        graph, key = live_graph_snapshot(state, object())
        assert graph.is_empty()
        assert key == (0, (("", 0),))

    def test_broken_journal_state_degrades_instead_of_raising(self):
        state = LiveLifecycleState(graph_source_default="kqapro")
        state.graph.by_owner[None] = _graph_delta("Q42", "Douglas Adams")
        agent = _FakeJournalAgent([{"state": "not a dict"}])

        graph, _key = live_graph_snapshot(state, agent)
        assert set(graph.nodes) == {"Q42"}

"""Smoke tests for the background lifecycle runner."""

from __future__ import annotations

import queue
import threading
import time

from ama_kbqa.framework.trace import TraceRecorder
from ama_kbqa.frontend.utils.lifecycle_runner import (
    LiveLifecycleState,
    drain_into,
    orchestrator_visited_node_ids,
    reconstruct_orchestrator,
    start_run,
)


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

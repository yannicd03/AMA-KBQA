"""Run lifecycle through the HTTP API with a fake agent.

Covers POST /api/runs, the SSE stream (snapshots, done, error), the run
record, the trace endpoint, session reset, the multiturn rules copied from
chat.py, demo limits, eviction, and stdout isolation between two concurrent
runs.
"""

from __future__ import annotations

import sys
import threading
import time

from ama_kbqa.api import runs
from ama_kbqa.pricing import estimate_cost_usd, format_cost_usd

from .conftest import MODEL, OTHER_MODEL, read_sse


def _events(frames, name):
    return [f for f in frames if f.get("event") == name]


def _release_later(gate: threading.Event, seconds: float = 0.3) -> None:
    timer = threading.Timer(seconds, gate.set)
    timer.daemon = True
    timer.start()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_run_streams_snapshots_then_done(harness):
    gate = harness.new_gate()
    resp = harness.post()
    assert resp.status_code == 201
    assert resp.json()["continuation"] is False
    run_id = resp.json()["run_id"]

    _release_later(gate)
    frames = read_sse(harness.client, run_id)

    snapshots = _events(frames, "snapshot")
    assert snapshots, frames
    assert frames[-1]["event"] == "done"
    # Frame ids are the monotonically increasing sequence.
    ids = [f["id"] for f in frames if "id" in f]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)

    first = snapshots[0]["data"]
    assert set(first) == {
        "seq", "status", "stage", "elapsed_s", "span_count",
        "figure", "subagents", "log_html", "graph",
    }
    assert first["status"] == "running"
    assert first["figure"]["kind"] == "lifecycle"
    assert first["figure"]["svg"].startswith("<svg")
    assert first["subagents"] == []
    # First frame on a connection always carries the log and the graph.
    assert isinstance(first["log_html"], str)
    assert first["graph"] is not None

    # While gated the agent is silent: later heartbeats carry null for the
    # unchanged log and graph.
    assert any(s["data"]["log_html"] is None and s["data"]["graph"] is None
               for s in snapshots[1:])

    # The live graph picked up the tool result, painted with KQAPro's group.
    graphs = [s["data"]["graph"] for s in snapshots if s["data"]["graph"]]
    live_nodes = {n["id"]: n for g in graphs for n in g["nodes"]}
    assert live_nodes["Q25191"]["group"] == "kqapro"
    assert any("Q25191" in g["new_ids"] for g in graphs)
    assert all(set(g["stats"]) >= {"nodes", "edges", "truncated"} for g in graphs)

    done = frames[-1]["data"]
    agent = harness.created[0]
    assert done["status"] == "done"
    assert done["run_id"] == run_id
    assert done["agent"] == "KQAPro"
    assert done["question"] == "Who is the director of Inception?"
    assert done["answer"] == "Inception was directed by Christopher Nolan."
    assert done["duration_s"] > 0
    assert done["tokens"] == {"prompt": 100, "completion": 20, "total": 120}
    assert done["model"] == MODEL
    assert done["cost"] == format_cost_usd(estimate_cost_usd(MODEL, 100, 20))
    assert done["trace_id"] == agent.recorder.trace_id
    # ANSI colour turned into a server-escaped span.
    assert '<span style="color:#3fb950">[agent0] start</span>' in done["log_html"]
    assert "[agent0] line 2" in done["log_html"]
    # Frozen figure: all lifecycle nodes visited, "completed in" caption.
    assert done["figure"]["kind"] == "lifecycle"
    assert "completed in" in done["figure"]["svg"]
    assert done["subagents"] == []
    # Frozen graph highlights the node the answer names.
    graph = done["graph"]
    assert graph["highlight"] == ["Q25191"]
    assert graph["answer_nodes"] == ["Christopher Nolan"]
    node = next(n for n in graph["nodes"] if n["id"] == "Q25191")
    assert node["highlighted"] is True and node["group"] == "kqapro"
    assert graph["stats"]["nodes"] == len(graph["nodes"])

    # The run record carries the done fields too.
    record = harness.client.get(f"/api/runs/{run_id}").json()
    assert record["status"] == "done"
    assert record["session_id"] == "s1"
    assert record["answer"] == done["answer"]
    assert record["continuation"] is False
    assert record["started_at"]

    # Reconnecting to a finished run: its final event, immediately, nothing else.
    again = read_sse(harness.client, run_id)
    assert [f["event"] for f in again] == ["done"]
    assert again[0]["data"] == done


def test_error_path(harness):
    harness.options["fail"] = True
    run_id, frames = harness.run()

    assert frames[-1]["event"] == "error"
    err = frames[-1]["data"]
    assert err == {
        "run_id": run_id,
        "status": "error",
        "message": "RuntimeError: boom",
        "log_html": err["log_html"],
    }
    assert "[agent0] line 0" in err["log_html"]

    record = harness.client.get(f"/api/runs/{run_id}").json()
    assert record["status"] == "error"
    assert record["message"] == "RuntimeError: boom"
    # A failed run is finished: its (partial) trace is available.
    trace = harness.client.get(f"/api/runs/{run_id}/trace")
    assert trace.status_code == 200
    assert trace.json()["events"]


def test_agent_creation_failure_becomes_an_error_event(harness):
    harness.create_error = RuntimeError("no MCP")
    run_id, frames = harness.run()
    assert [f["event"] for f in frames] == ["error"]
    assert frames[0]["data"]["message"] == "Could not create the agent: RuntimeError: no MCP"

    # The session was released: the next question is accepted.
    harness.create_error = None
    _run_id, frames = harness.run()
    assert frames[-1]["event"] == "done"


def test_live_graph_disabled_sends_no_graph(harness, monkeypatch):
    monkeypatch.setattr(runs, "get_live_graph_enabled", lambda: False)
    _run_id, frames = harness.run()
    assert all(f["data"]["graph"] is None for f in _events(frames, "snapshot"))
    assert frames[-1]["data"]["graph"] is None


# ---------------------------------------------------------------------------
# Concurrency and sessions
# ---------------------------------------------------------------------------

def test_second_run_in_same_session_is_409(harness):
    gate = harness.new_gate()
    first = harness.post(session="s1")
    assert first.status_code == 201
    run_id = first.json()["run_id"]

    busy = harness.post(session="s1")
    assert busy.status_code == 409
    assert busy.json() == {"detail": "This session already has a run in flight."}
    assert harness.client.post("/api/sessions/s1/reset").status_code == 409
    assert harness.client.get(f"/api/runs/{run_id}/trace").status_code == 409
    assert harness.client.get(f"/api/runs/{run_id}").json()["status"] == "running"

    # Another session may run at the same time.
    other = harness.post(session="s2")
    assert other.status_code == 201

    gate.set()
    assert read_sse(harness.client, run_id)[-1]["event"] == "done"
    assert read_sse(harness.client, other.json()["run_id"])[-1]["event"] == "done"
    # Finished: the session accepts the next question again.
    _run_id, frames = harness.run(session="s1")
    assert frames[-1]["event"] == "done"


def test_specialist_followup_reuses_the_agent(harness):
    run1, _ = harness.run(question="Who directed Inception?")
    second = harness.post(question="Where was he born?", temperature=0.5)
    assert second.status_code == 201
    assert second.json()["continuation"] is True
    run2 = second.json()["run_id"]
    read_sse(harness.client, run2)

    assert len(harness.created) == 1
    agent = harness.created[0]
    assert agent.asked == ["Who directed Inception?", "Where was he born?"]
    # start_run's continuation path: history kept, MCP rebuilt on the new loop.
    assert agent.reset_calls == [{"keep_mcp_open": True, "keep_history": True}]
    # Settings applied before every run, under the build lock.
    assert harness.applied == [(MODEL, 1.0), (MODEL, 0.5)]

    # Each run froze its own trace, although the agent's recorder was reset.
    def questions(run_id):
        events = harness.client.get(f"/api/runs/{run_id}/trace").json()["events"]
        return {e["attributes"].get("question") for e in events if e["kind"] == "agent_run"}

    assert questions(run1) == {"Who directed Inception?"}
    assert questions(run2) == {"Where was he born?"}

    # Another model: a fresh agent, not a follow-up.
    third = harness.post(model=OTHER_MODEL)
    assert third.json()["continuation"] is False
    read_sse(harness.client, third.json()["run_id"])
    # Another specialist: fresh again.
    fourth = harness.post(agent="SciQA")
    assert fourth.json()["continuation"] is False
    read_sse(harness.client, fourth.json()["run_id"])
    # Reset drops the stored agent: the same specialist starts over.
    assert harness.client.post("/api/sessions/s1/reset").status_code == 204
    fifth = harness.post(agent="SciQA")
    assert fifth.json()["continuation"] is False
    read_sse(harness.client, fifth.json()["run_id"])
    assert len(harness.created) == 4


def test_orchestrator_is_always_fresh_and_drops_the_stored_agent(harness):
    harness.run(agent="KQAPro")
    # Gate the first orchestrator run so the stream is guaranteed to open
    # while it is still live (an ungated fake run can finish before the
    # client connects, and then only its done event is sent).
    gate = harness.new_gate()
    first = harness.post(agent="Orchestrator (Router)")
    assert first.json()["continuation"] is False
    _release_later(gate)
    frames = read_sse(harness.client, first.json()["run_id"])
    harness.options.pop("gate")
    second = harness.post(agent="Orchestrator (Router)")
    assert second.json()["continuation"] is False
    read_sse(harness.client, second.json()["run_id"])
    # The orchestrator pick dropped the KQAPro agent: no follow-up now.
    third = harness.post(agent="KQAPro")
    assert third.json()["continuation"] is False
    read_sse(harness.client, third.json()["run_id"])

    assert len(harness.created) == 4
    assert len({id(a) for a in harness.created}) == 4
    assert all(a.reset_calls == [] for a in harness.created)

    snapshot = _events(frames, "snapshot")[0]["data"]
    assert snapshot["figure"]["kind"] == "orchestrator"
    assert "orchestrator-svg" in snapshot["figure"]["svg"]
    done = frames[-1]["data"]
    assert done["figure"]["kind"] == "orchestrator"
    assert "completed in" in done["figure"]["svg"]
    assert done["subagents"] == []


def test_reset_of_unknown_session_is_204(harness):
    assert harness.client.post("/api/sessions/nobody/reset").status_code == 204


def test_stdout_is_routed_per_run(harness):
    gate = harness.new_gate()
    harness.options.update(prints=15, delay=0.005)
    run_a = harness.post(session="a").json()["run_id"]
    run_b = harness.post(session="b").json()["run_id"]
    # A write from outside any run goes to the real stdout, not into a log.
    print("MAIN-THREAD-NOISE")
    time.sleep(0.2)
    gate.set()

    log_a = read_sse(harness.client, run_a)[-1]["data"]["log_html"]
    log_b = read_sse(harness.client, run_b)[-1]["data"]["log_html"]
    assert "[agent0] line 14" in log_a and "[agent1]" not in log_a
    assert "[agent1] line 14" in log_b and "[agent0]" not in log_b
    assert "MAIN-THREAD-NOISE" not in log_a + log_b


def test_a_run_reinstalls_the_proxy_if_stdout_was_replaced(harness):
    # pytest itself swaps sys.stdout between test phases, after the app's
    # lifespan installed the proxy: the run must still get its own log.
    from ama_kbqa.api.stdout_router import RoutingStdout

    _run_id, frames = harness.run()
    assert isinstance(sys.stdout, RoutingStdout)
    assert "[agent0] line 0" in frames[-1]["data"]["log_html"]


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def test_trace_truncates_payload_strings(harness):
    harness.options["big_payload"] = True
    run_id, _ = harness.run()

    resp = harness.client.get(f"/api/runs/{run_id}/trace")
    assert resp.status_code == 200
    body = resp.json()
    agent = harness.created[0]
    assert body["trace_id"] == agent.recorder.trace_id
    assert len(body["events"]) == len(agent.recorder.events)
    assert body["summary"]["n_tool_calls"] == 1
    assert body["summary"]["n_llm_calls"] == 1
    assert body["summary"]["total_tokens"] == 120

    tool = next(e for e in body["events"] if e["kind"] == "tool_call")
    suffix = runs.TRUNCATION_SUFFIX
    assert tool["payload"]["blob"] == "x" * 20_000 + suffix
    assert tool["payload"]["nested"]["items"] == ["y" * 20_000 + suffix, "short"]
    assert tool["payload"]["arguments"] == {"node_id": "Q25191"}
    # The span-tree fields the frontend needs are all there.
    assert {"span_id", "parent_span_id", "kind", "name", "status", "duration_ms",
            "start_time_unix_nano", "attributes", "payload", "is_event"} <= set(tool)


# ---------------------------------------------------------------------------
# Validation, limits, eviction
# ---------------------------------------------------------------------------

def test_invalid_requests_are_400(harness):
    cases = [
        ({"agent": "Nope"}, "Unknown agent: Nope"),
        ({"model": "kit.not-a-model"}, "Unknown model: kit.not-a-model"),
        ({"temperature": 2.5}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"question": ""}, "question"),
        ({"question": "   "}, "The question must not be empty."),
        ({"question": "x" * 4001}, "question"),
    ]
    for override, needle in cases:
        resp = harness.post(**override)
        assert resp.status_code == 400, (override, resp.text)
        assert needle in resp.json()["detail"], (override, resp.json())

    missing = harness.client.post("/api/runs", json={"session_id": "s1"})
    assert missing.status_code == 400
    assert isinstance(missing.json()["detail"], str)
    assert harness.created == []


def test_unknown_run_is_404(harness):
    for path in ("/api/runs/nope", "/api/runs/nope/trace", "/api/runs/nope/events"):
        resp = harness.client.get(path)
        assert resp.status_code == 404, path
        assert "detail" in resp.json()


def test_demo_query_limit_is_429(harness, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_QUERIES_PER_SESSION", "1")
    monkeypatch.setenv("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "0")
    harness.run()
    resp = harness.post()
    assert resp.status_code == 429
    assert resp.json()["detail"] == (
        "Demo limit reached: 1 queries per session. Refresh the page to start a new session."
    )


def test_demo_rate_limit_is_429(harness, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MIN_SECONDS_BETWEEN_QUERIES", "60")
    harness.run()
    resp = harness.post()
    assert resp.status_code == 429
    assert resp.json()["detail"].startswith("Please wait ")
    assert resp.json()["detail"].endswith("s before the next query.")


def test_oldest_finished_runs_are_evicted(harness, monkeypatch):
    monkeypatch.setattr(runs, "MAX_RUNS", 2)
    first, _ = harness.run(session="a")
    second, _ = harness.run(session="b")
    third, _ = harness.run(session="c")
    assert harness.client.get(f"/api/runs/{first}").status_code == 404
    assert harness.client.get(f"/api/runs/{second}").status_code == 200
    assert harness.client.get(f"/api/runs/{third}").status_code == 200


def test_idle_sessions_are_evicted():
    manager = runs.RunManager()
    manager.sessions["idle"] = runs.Session("idle", last_seen=0.0)
    manager.sessions["busy"] = runs.Session("busy", last_seen=0.0, active_run_id="r1")
    manager.sessions["fresh"] = runs.Session("fresh", last_seen=runs.SESSION_IDLE_SECONDS)
    manager.evict_idle_sessions(now=runs.SESSION_IDLE_SECONDS + 10)
    assert set(manager.sessions) == {"busy", "fresh"}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_log_tail_is_capped_on_a_line_boundary():
    raw = "".join(f"line {i:06d}\n" for i in range(20_000))  # ~240 KB
    html = runs.log_tail_html(raw)
    assert html.startswith('<span style="color:#6e7681">… earlier output truncated …</span>\n')
    body = html.split("</span>\n", 1)[1]
    assert body.startswith("line ") and body.endswith("line 019999\n")
    assert len(body) <= runs.LOG_TAIL_CHARS


def test_short_log_is_converted_unchanged():
    assert runs.log_tail_html("\x1b[31mred\x1b[0m <b>") == (
        '<span style="color:#ff7b72">red</span> &lt;b&gt;'
    )


def test_truncate_trace_events_only_touches_payload():
    events = [{
        "kind": "tool_call",
        "attributes": {"note": "a" * 30},
        "payload": {"s": "b" * 30, "n": 3, "obj": object()},
    }]
    out = runs.truncate_trace_events(events, limit=10)
    assert out[0]["attributes"]["note"] == "a" * 30
    assert out[0]["payload"]["s"] == "b" * 10 + runs.TRUNCATION_SUFFIX
    assert out[0]["payload"]["n"] == 3
    assert isinstance(out[0]["payload"]["obj"], str)  # made JSON-safe

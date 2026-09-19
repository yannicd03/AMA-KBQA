"""Cancelling a run through the HTTP API.

The agent half is covered by ``tests/frontend/test_lifecycle_cancellation.py``
and ``tests/agents/test_orchestrator_cancellation.py``. This is the API half:
the cancel route, the ``cancelled`` terminal event, and what cancelling does to
the session that owned the run.

The fake agent spins in its gate loop until the token is flipped, so every
"mid-run" cancel here is genuinely mid-run: the token is always set before the
agent leaves ``ask()``.
"""

from __future__ import annotations

from ama_kbqa.api import runs
from ama_kbqa.framework.cancellation import CANCELLED_ANSWER

from .conftest import read_sse


def _cancel(harness, run_id: str) -> dict:
    resp = harness.client.post(f"/api/runs/{run_id}/cancel")
    # 202: the stop was accepted, the agent has not necessarily noticed yet.
    assert resp.status_code == 202, resp.text
    return resp.json()


def _manager(harness) -> runs.RunManager:
    return harness.client.app.state.manager


# ---------------------------------------------------------------------------
# The terminal event
# ---------------------------------------------------------------------------

def test_cancel_mid_run_ends_with_a_cancelled_event(harness):
    harness.new_gate()
    resp = harness.post()
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    assert _cancel(harness, run_id) == {"run_id": run_id, "cancelling": True}

    frames = read_sse(harness.client, run_id)
    assert frames[-1]["event"] == "cancelled"
    payload = frames[-1]["data"]
    assert payload["run_id"] == run_id
    assert payload["status"] == "cancelled"
    assert payload["answer"].startswith(CANCELLED_ANSWER)
    assert payload["duration_s"] >= 0
    assert "[agent0] line 0" in payload["log_html"]
    # Never the error event: reusing `error` would lose the distinction the
    # client needs, and a cancelled run did not fail.
    assert not any(f.get("event") == "error" for f in frames)
    # The token really reached the agent.
    assert harness.created[0].seen_token is not None

    record = harness.client.get(f"/api/runs/{run_id}").json()
    assert record["status"] == "cancelled"
    assert record["answer"].startswith(CANCELLED_ANSWER)


def test_the_status_comes_from_the_token_not_the_answer_text(harness):
    """An uncancelled run that happens to answer with the cancelled wording is
    still a normal ``done``."""
    harness.options["answer"] = CANCELLED_ANSWER
    _run_id, frames = harness.run()
    assert frames[-1]["event"] == "done"
    assert frames[-1]["data"]["status"] == "done"


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

def test_cancelling_unlocks_the_session_at_once(harness):
    harness.new_gate()
    first = harness.post(session="s1")
    assert first.status_code == 201
    run_id = first.json()["run_id"]
    # Locked while the run is in flight: this is what Stop is for.
    assert harness.post(session="s1").status_code == 409
    assert harness.client.post("/api/sessions/s1/reset").status_code == 409

    _cancel(harness, run_id)

    # Free immediately, without waiting for the old run to drain.
    assert _manager(harness).sessions["s1"].active_run_id is None
    assert harness.client.post("/api/sessions/s1/reset").status_code == 204

    harness.options.pop("gate")
    second = harness.post(session="s1")
    assert second.status_code == 201
    assert read_sse(harness.client, second.json()["run_id"])[-1]["event"] == "done"
    assert read_sse(harness.client, run_id)[-1]["event"] == "cancelled"


def test_cancelling_drops_the_stored_agent(harness):
    # A finished specialist run leaves its agent behind for the next turn.
    harness.run(question="Who directed Inception?")
    assert _manager(harness).sessions["s1"].stored is not None

    # The follow-up reuses that instance, which was built before this gate
    # existed, so gate it directly: options only reach agents built later, and
    # an ungated fake would race to its answer before the cancel lands.
    harness.created[0].gate = harness.new_gate()
    second = harness.post(question="Where was he born?")
    assert second.json()["continuation"] is True
    second_id = second.json()["run_id"]
    _cancel(harness, second_id)
    assert _manager(harness).sessions["s1"].stored is None
    assert read_sse(harness.client, second_id)[-1]["event"] == "cancelled"

    # v1 rule: a cancelled agent is never reused, so the next question builds a
    # fresh one instead of continuing the conversation.
    harness.options.pop("gate")
    third = harness.post(question="And his birth year?")
    assert third.json()["continuation"] is False
    read_sse(harness.client, third.json()["run_id"])
    assert len(harness.created) == 2


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_cancelling_an_unknown_or_finished_run_is_idempotent(harness):
    # Unknown or already evicted: accepted, never a 404 and never a 500.
    assert _cancel(harness, "nope") == {"run_id": "nope", "cancelling": False}

    run_id, frames = harness.run()
    assert frames[-1]["event"] == "done"
    assert _cancel(harness, run_id) == {"run_id": run_id, "cancelling": False}
    # Cancelling a finished run does not rewrite it.
    assert harness.client.get(f"/api/runs/{run_id}").json()["status"] == "done"

    # Twice on a live run: the second call must not double-count the session.
    harness.new_gate()
    live = harness.post(session="s2").json()["run_id"]
    _cancel(harness, live)
    _cancel(harness, live)
    assert _manager(harness).sessions["s2"].detached_runs <= 1
    assert read_sse(harness.client, live)[-1]["event"] == "cancelled"
    assert _manager(harness).sessions["s2"].detached_runs == 0


# ---------------------------------------------------------------------------
# The run does not wedge, and the overlap is bounded
# ---------------------------------------------------------------------------

def test_a_cancelled_run_finishes_and_is_evictable(harness, monkeypatch):
    monkeypatch.setattr(runs, "MAX_RUNS", 1)
    harness.new_gate()
    run_id = harness.post(session="a").json()["run_id"]
    _cancel(harness, run_id)
    assert read_sse(harness.client, run_id)[-1]["event"] == "cancelled"

    manager = _manager(harness)
    assert manager.runs[run_id].finished is True
    assert manager.runs[run_id].status == "cancelled"
    # Released on both counters, so nothing keeps the session busy.
    assert manager.sessions["a"].active_run_id is None
    assert manager.sessions["a"].detached_runs == 0
    # Finished, so its partial trace is served rather than 409'd…
    assert harness.client.get(f"/api/runs/{run_id}/trace").status_code == 200

    # …and the eviction sweep may drop it, which it never does to a run still
    # stuck in "running".
    harness.options.pop("gate")
    harness.run(session="b")
    assert harness.client.get(f"/api/runs/{run_id}").status_code == 404


def test_a_session_with_too_many_draining_runs_refuses_a_new_one(harness):
    manager = _manager(harness)
    manager.sessions["busy"] = runs.Session(
        "busy", detached_runs=runs.MAX_DETACHED_RUNS_PER_SESSION
    )
    resp = harness.post(session="busy")
    assert resp.status_code == 409
    assert "winding down" in resp.json()["detail"]
    # Refused before any agent was built, so no overlapping apply/create pair.
    assert harness.created == []


def test_a_draining_session_is_not_evicted_as_idle():
    manager = runs.RunManager()
    manager.sessions["draining"] = runs.Session("draining", last_seen=0.0, detached_runs=1)
    manager.sessions["idle"] = runs.Session("idle", last_seen=0.0)
    manager.evict_idle_sessions(now=runs.SESSION_IDLE_SECONDS + 10)
    assert set(manager.sessions) == {"draining"}

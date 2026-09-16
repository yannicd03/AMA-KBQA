"""The React client's half of the cancellation contract.

There is no JS test runner in this repo, so these are static checks on the
sources rather than behavioural tests. They earn their place because the bug
they guard against is invisible at runtime: ``EventSource`` silently ignores an
event name it has no listener for, and ``api.ts``'s ``recover()`` treats any
status that is not terminal as "still running" and reconnects every 2 seconds
forever. Dropping either half of the cancelled wiring therefore ships an
infinite reconnect loop that reads as a hang, with nothing in the console.
"""

from __future__ import annotations

from pathlib import Path

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"
API_TS = (WEB_SRC / "api.ts").read_text(encoding="utf-8")
STATE_TS = (WEB_SRC / "state.ts").read_text(encoding="utf-8")

CANCELLED_LISTENER = 'es.addEventListener("cancelled"'


def test_the_sse_stream_has_a_cancelled_listener():
    assert CANCELLED_LISTENER in API_TS
    body = API_TS[API_TS.index(CANCELLED_LISTENER):]
    body = body[: body.index("});")]
    # It reaches its own handler and is not folded into the error path.
    assert "h.onCancelled(" in body
    assert "h.onError(" not in body


def test_recover_treats_cancelled_as_terminal():
    start = API_TS.index("const recover =")
    recover = API_TS[start: API_TS.index("const connect =", start)]
    assert 'rec.status === "cancelled"' in recover
    branch = recover[recover.index('rec.status === "cancelled"'):]
    # finish() is what closes the stream and cancels the 2s retry timer.
    assert "finish();" in branch
    assert "h.onCancelled(" in branch


def test_the_record_and_view_types_know_the_status():
    assert '"running" | "done" | "error" | "cancelled"' in API_TS
    run_status = STATE_TS.split("export type RunStatus")[1].split("\n")[0]
    assert '"cancelled"' in run_status
    assert "export function applyCancelled" in STATE_TS


def test_the_client_can_ask_the_server_to_cancel():
    assert "cancelRun(runId: string): Promise<void>;" in API_TS
    assert "/cancel" in API_TS

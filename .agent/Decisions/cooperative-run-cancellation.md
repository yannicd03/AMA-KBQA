---
type: decision
status: accepted
date: 2026-09-16
summary: A run is stopped by flipping a flag the agent checks at its own checkpoints, never by killing a thread or cancelling a task; the cancel route answers 202, cancelled is a real SSE event, overlapping runs are bounded not blocked, and every previously unbounded wait (SPARQL, MCP tool calls) is now bounded.
addresses: [issue/no-way-to-stop-a-run, issue/unbounded-waits-sparql-and-mcp]
affects: [branch/demo-v2-int, feature/cooperative-run-cancellation, issue/apply-chat-settings-process-global-concurrency]
relates: [react-frontend-second-ui, federated-dispatch-and-fusion, architecture-audit-2026-07-05]
evidence:
  - "commit 34c2314"
  - "commit 69cd5a9"
  - "ama_kbqa/framework/cancellation.py:37"
  - "ama_kbqa/api/runs.py:81"
  - "ama_kbqa/api/runs.py:579"
  - "ama_kbqa/framework/sparql_client.py"
---

# ADR: Cooperative Run Cancellation, and Bounding Every Wait

**Status:** Accepted (shipped 2026-09-16 on `demo-v2-int`, commits `34c2314`
framework + orchestrator + bounded waits, `69cd5a9` HTTP/UI surface).

**Narrows:** [architecture-audit-2026-07-05.md](./architecture-audit-2026-07-05.md)
items B1/B2 ("always exit through synthesis"). That invariant now has exactly
one deliberate exception, argued in Decision 1 below.

## Related Docs
- [System/agent_framework.md](../System/agent_framework.md) — the checkpoints, the token, and the bounded-wait mechanics
- [System/demo_react_frontend.md](../System/demo_react_frontend.md) — the cancel route, the `cancelled` SSE event, detached runs
- [Decisions/react-frontend-second-ui.md](./react-frontend-second-ui.md) — the API this surfaces through
- [Decisions/federated-dispatch-and-fusion.md](./federated-dispatch-and-fusion.md) — the "close MCP in the task that opened it" constraint this decision rests on
- [SOP/running_react_demo_locally.md](../SOP/running_react_demo_locally.md) — Stop-button behaviour when running it

---

## Context

The demo had no way to stop a run. A visitor who asked an expensive question
waited it out, and the session stayed locked behind `409` for the whole
duration — both the next question and "New chat" were refused until the agent
finished on its own.

The obvious implementations are all unavailable here, for one structural
reason: **a run executes on a daemon worker thread that owns its own event
loop** (`ama_kbqa/frontend/utils/lifecycle_runner.py::start_run`), and its MCP
connection is bound to *that* loop.

- The thread is a daemon thread with no kill primitive, so it cannot be stopped
  from outside.
- Cancelling the asyncio consumer task (`RunManager._consume`) would stop only
  the SSE stream. The agent would run on, and keep billing, invisibly.
- Tearing MCP down out-of-band is not merely untidy, it is impossible: anyio
  raises `Attempted to exit cancel scope in a different task` when a connection
  is closed anywhere but the task that opened it. This is the same constraint
  that already makes `lifecycle_runner` orphan a stale connection rather than
  close it, and that makes `Orchestrator._run_specialist` close each specialist
  inside its own task (see `federated-dispatch-and-fusion.md`). Out-of-band
  teardown therefore *cannot* reclaim the subprocess.

There was also a correctness bug in the pre-existing cancel path: it never
called `run.finish`, so a stopped run reported `running` forever and was never
evicted.

Separately, cancellation can only promise a time bound if every wait inside a
run is itself bounded — and two were not (see Decision 6).

## Decision 1 — Cancellation is cooperative, and a cancelled run skips synthesis

The caller flips a `CancellationToken` (`ama_kbqa/framework/cancellation.py`, a
thread-safe `threading.Event` wrapper, because the setter runs on the API's
event loop and the reader on the worker thread). The agent notices it at one of
its own checkpoints and **returns normally through its own code**, so every
`finally` on the way out — MCP teardown included — runs on the worker's own
loop, exactly as for a run that finished by itself.

Checkpoints (`ama_kbqa/framework/base_agent.py`):

1. **After MCP init, before classification** — a run stopped here has not made a
   single billed LLM call.
2. **At the top of the tool loop**, once per iteration.

The Orchestrator is a separate implementation and gets its own checkpoints
(`ama_kbqa/agents/orchestrator_agent/agent.py`), including both branches of the
federated fan-out and the KQAPro fallback path.

**The tool-loop exit deliberately does not run synthesis.** Synthesis is a
second billed LLM call; a visitor who pressed Stop must not pay for it. This is
the one documented exception to the B1/B2 "always exit through synthesis"
invariant — every other terminal path still funnels through it. The exit still
appends the answer to `self._messages`, so the stack stays answer-terminated and
a follow-up turn on that instance would replay a valid conversation.

Everything is opt-in: `cancel_token=None` is the default at every entry point
and `is_cancelled(None)` is `False`, so the Streamlit page and the benchmark
runners are behaviourally unchanged.

## Decision 2 — No checkpoint inside tool execution

Cancelling in the middle of `_execute_tool_calls` would leave an assistant
message carrying `tool_calls` whose matching `role=tool` results were never
appended. Replaying that stack makes providers reject the request with a 400.

The top of the loop is the only point where the message stack is guaranteed
consistent, so there is deliberately no finer-grained checkpoint. The cost is
stop latency: a long tool call runs to completion before the flag is seen.

## Decision 3 — The cancel route answers `202 Accepted`

`POST /api/runs/{run_id}/cancel` returns **202**, not 204. The route flips the
flag and returns while the agent is still running to its next checkpoint, which
is literally "accepted, processing not complete". 204 would claim the work was
already over.

**Unknown, evicted and already-finished runs also get 202**, with
`{"cancelling": false}`. Stopping something that is already stopped is the
outcome the caller wanted; a 404 would make the UI render a successful Stop as a
failure.

## Decision 4 — Overlapping runs are bounded, not blocked

Cancelling frees the session slot **immediately** (`RunManager.cancel_run`),
because freeing the session is the entire user-visible point of the feature —
before this, a stopped run left both the next question and "New chat" returning
409. The consequence is that a new run can overlap a still-draining one.

That overlap is bounded by `Session.detached_runs` and
`MAX_DETACHED_RUNS_PER_SESSION = 2` (`ama_kbqa/api/runs.py`) rather than
forbidden. Blocking the second run would defeat the feature.

Why the overlap is safe: `self.model` is frozen at agent construction, so a new
run cannot retarget an already-draining agent. The residual window is env-var
contamination flowing *into* the cancelled run (the process-global override
described in `react-frontend-second-ui.md` trade-off #2) — and that run's output
is discarded anyway.

A cancelled session's stored agent is dropped (`session.stored = None`): its
message stack was cut mid-turn and its MCP connection belongs to a worker loop
that is still unwinding, so the next question builds a fresh agent rather than
continuing that conversation.

## Decision 5 — `cancelled` is a real SSE event, not a reused `error`

The terminal event kind is decided by the token, never by string-matching the
answer text. Two reasons it is not folded into `error`:

- The React client treats any status that is not `done`/`error` as still
  running and reconnects every 2 seconds, forever. A cancelled run that never
  emitted a terminal event it recognises would present as a hang, with nothing
  in the console — `EventSource` silently ignores event names it has no
  listener for.
- Rendering a deliberate stop as a failure is simply wrong in the UI.

## Decision 6 — Every previously unbounded wait is now bounded

Cancellation can only promise a time limit if the run cannot block forever
between checkpoints. Two waits were unbounded:

- **SPARQL.** `GraphConfig.timeout_ms` (30s) had been plumbed through
  `framework/adapters/resolve.py` into both adapters since the adapter refactor
  and then never consumed — the servers built a bare `SPARQLWrapper(endpoint)`,
  leaving `urlopen` with no timeout at all. All clients now build through
  `framework/sparql_client.py::make_sparql_client`, which applies the timeout
  once at construction so ~60 call sites inherit it untouched. Expiry becomes
  `SparqlTimeout`, whose message names the budget and says how to narrow the
  query, so the agent reads an actionable tool error instead of an opaque socket
  exception. This covered a third construction site in `ama_kbqa/postprocessing.py`
  that the original plumbing had missed.
- **MCP tool calls** had no timeout whatsoever. They now default to 180s
  (`[agent] mcp_tool_timeout_seconds`, env
  `AMA_KBQA_MCP_TOOL_TIMEOUT_SECONDS`), implemented with the SDK's own
  `read_timeout_seconds` rather than an `asyncio.wait_for` wrapper — the SDK
  cancels the pending request *inside* the session's own task, instead of
  abandoning an await from outside it, which is the same anyio constraint that
  shapes Decision 1. A value `<= 0` restores the old unbounded behaviour for a
  deliberately slow workload.

## Trade-offs accepted

- **Stop is not instant, and the LLM call dominates its latency.** The LLM call
  is synchronous and not interruptible, so a run waiting on a model response
  finishes that response before it can notice the flag. A fast-path run already
  in flight has no checkpoint at all.
- **A cancelled run keeps consuming resources until its next checkpoint**, which
  is what Decision 4 exists to bound.
- **The answer text of a cancelled run is not a real answer** (`"Run cancelled."`,
  plus a reason when one was given). Callers that need to distinguish it should
  consult their own token rather than string-match `CANCELLED_ANSWER`.

## Known gaps (unverified as of 2026-09-16)

These are recorded as gaps, not as results. Nothing below was observed working.

- **The client half is covered by static source assertions only, never executed.**
  There is no JS test runner in this repo. `tests/api/test_web_client_contract.py`
  reads `web/src/api.ts` and `web/src/state.ts` as text and asserts the
  `cancelled` listener exists, that `recover()` treats `cancelled` as terminal,
  and that the status types know the value. `tsc -b` passes, but **no client code
  was run**, and the infinite-reconnect scenario this guards against was never
  reproduced nor observed fixed. This is the riskiest unverified line in the
  feature.
- **The Stop button has never been rendered.** No browser check was performed —
  Chrome automation is broken on this machine (a pre-existing regression already
  recorded in `System/demo_react_frontend.md`).
- **The detached-run cap is tested by injecting session state**, not by driving
  two genuinely overlapping real runs.
- The three commits were not individually test-verified in isolation, so the
  history may not bisect cleanly.

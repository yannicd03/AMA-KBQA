# ADR: KIT Outage / Hang Detection Watchdog

**Date:** 2026-05-08
**Status:** Accepted
**Commit:** `d01d029`
**Files:**
- `ama_kbqa/config.py` — `_create_client`: structured `httpx.Timeout` replacing bulk `timeout=60.0`
- `ama_kbqa/benchmark_agents.py` — `run_benchmark_for_model_agent`: consecutive infra-failure counter + `[FATAL]` abort

## Related Docs

- [Decisions/text-mode-tool-calls.md](./text-mode-tool-calls.md) — general KIT endpoint quirk handling (minimax-m2.7 prose output)
- [Decisions/truncated-tool-call-retry.md](./truncated-tool-call-retry.md) — same theme: defense against KIT-side failure modes corrupting benchmark data
- [System/agent_system.md](../System/agent_system.md) — agent lifecycle; loop-detection mechanisms (this watchdog operates at the benchmark runner level, not the agent loop level)

---

## Context: Two Empirical Failures

### Failure A — KIT bursty 404s corrupting minimax accuracy

Run `benchmark_results/minimax-kqapro-2026-05-06-seed43` (minimax-m2.7-kit, seed=43):

- Questions 70–93 (24 consecutive) hit bursty KIT 404s.
- Without a watchdog, each question was logged as `INCORRECT` — the agent loop saw an API error, fell through to synthesis with no journal data, and produced garbage output.
- Raw accuracy: 0.88 → 0.67 (−21 pp).
- The corruption was undetected until the pattern was noticed by hand during the seed=43 sample analysis.

### Failure B — gemma hang lasting 24 hours

Run `gemma-kqapro-2026-05-07-seed44` (first attempt):

- Process hung at iteration 17 of question 4.
- CPU was at 0%; no exception raised.
- Root cause: the bulk `timeout=60.0` passed to the OpenAI client is interpreted as a single-phase read timeout. When a KIT socket accepted the connection but then stalled at the socket level (TCP ESTABLISHED, zero bytes returned), the read deadline never fired — the socket sat idle indefinitely.

---

## Decision

### Fix 1 — Structured `httpx.Timeout` in `_create_client`

Replace:
```python
client = openai.AsyncOpenAI(..., timeout=60.0)
```
With:
```python
import httpx
client = openai.AsyncOpenAI(
    ...,
    timeout=httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0)
)
```

**Why this works:** `httpx.Timeout` with per-phase budgets maps to TCP-level socket timeouts. A socket that accepts the TCP connection but never returns bytes hits the `read=60.0` deadline and raises `httpx.ReadTimeout`. This surfaces as a normal `httpx.TimeoutError` exception, which the existing `asyncio.wait_for(timeout=600)` wrapping `process_single_question` can catch and propagate cleanly.

**Phase budget rationale:**

| Phase | Budget | Rationale |
|-------|--------|-----------|
| `connect` | 20 s | KIT connections are local network; >20 s means the endpoint is down |
| `read` | 60 s | Generous for streaming LLM output; covers long completions |
| `write` | 10 s | Request body is small; >10 s means socket is stalled |
| `pool` | 5 s | httpx connection pool acquisition; should be near-instant |

### Fix 2 — Consecutive infra-failure watchdog in `run_benchmark_for_model_agent`

**Patterns matched (`INFRA_ERROR_PATTERNS`):**
- `NotFoundError` / `404`
- `'NoneType' object has no attribute 'choices'`
- `litellm.NotFoundError`
- `Hosted_vllmException`
- `ConnectError` / `ReadTimeout` / `WriteTimeout` / `PoolTimeout` / `ConnectTimeout`
- `RemoteProtocolError`

**Mechanism:**
```python
consecutive_infra_failures = 0
INFRA_FAILURE_ABORT_THRESHOLD = 5  # configurable constant

for question in questions:
    result = await process_single_question(...)
    if any(pat in result["error"] for pat in INFRA_ERROR_PATTERNS):
        consecutive_infra_failures += 1
    else:
        consecutive_infra_failures = 0  # reset on any non-infra result

    if consecutive_infra_failures >= INFRA_FAILURE_ABORT_THRESHOLD:
        log.error("[FATAL] 5 consecutive infrastructure failures — aborting run")
        break  # partial output is preserved; saved results are written
```

**Why 5 (not lower, not higher):**
- A threshold of 1–2 would abort on transient flakes (e.g., a momentary KIT restart).
- A threshold of 10+ would allow a 10-question corruption window before aborting.
- 5 is empirically derived: Failure A showed 24-question corruption starting from iteration 70. At 5, the run would have aborted after question 74 — preserving 73 clean results and flagging the outage.
- Adjustable via `INFRA_FAILURE_ABORT_THRESHOLD` constant in `benchmark_agents.py`.

**Partial output preservation:** The abort path uses `break`, not `sys.exit`. All results collected up to that point are written to disk in the normal output path (results.json, summary.json). The partial run is re-runnable with `--resume` once the endpoint recovers.

---

## Alternatives Considered

| Option | Rejected reason |
|--------|-----------------|
| Raise the `asyncio.wait_for` timeout to 3600 s | Would turn a hang into a 1-hour wait per question; doesn't detect socket-level stalls |
| Retry individual questions on infra error | A bursty outage would retry all 24 questions, still burning time against a down endpoint |
| Per-request retry with backoff | Better than no retry, but still doesn't abort a sustained outage — adds complexity; out of scope for this fix |
| Alert via external webhook | Overkill for a research benchmark; the `[FATAL]` log line is sufficient for manual monitoring |

---

## Invariants

- `INFRA_FAILURE_ABORT_THRESHOLD = 5` — change only with empirical justification.
- The counter resets to 0 on **any** non-infra result (including agent-logic failures like max-iterations). Only consecutive infra failures trigger the abort.
- The `httpx.Timeout` phases must be kept proportional: `connect` < `read`, `pool` ≪ `connect`. Do not equalize them — that defeats the purpose of per-phase granularity.
- Do not increase `read` timeout to suppress hang symptoms. If reads consistently stall at 60 s, the endpoint is malfunctioning — the abort watchdog should fire.

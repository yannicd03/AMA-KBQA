# ADR: WikiKGQA Tool-Call Budget as Binding Limit + Provider-Error Resilience

**Date:** 2026-06-30
**Branch:** `worktree-wikikgqa-2026`
**Status:** Settled — implemented and validated (199 tests passing; commitment fix confirmed on live Qwen3.5-397B re-run 2026-07-01, see Validation)

## Related Docs
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — full current-state mechanism reference for everything described here
- [Decisions/wikikgqa-2026-adaptation.md](./wikikgqa-2026-adaptation.md) — original ADR for the WikiKGQA subsystem (agent generates SPARQL, not NL)
- [Decisions/kit-outage-watchdog.md](./kit-outage-watchdog.md) — prior precedent for KIT-proxy transient-failure handling (structured `httpx.Timeout` + consecutive-failure watchdog in the benchmark runner); this ADR adds a complementary retry at the LLM-call layer

---

## Problem

Running the `AgentSparqlGenerator` (full agentic tool loop) against the larger
model **Qwen3.5-397B (KIT)** surfaced two failure modes that were silently
losing questions:

**(a) Over-exploration timeout cancel-with-nothing.** The model kept
re-running `RunSPARQL` (often the same broad query, or minor variations)
instead of committing to an answer. The only stopping condition was the
generator's `agent_timeout` wall-clock cap (180s). When that fired, the call
was hard-cancelled (`asyncio.wait_for` raises `TimeoutError`, caught by
`generate()`'s broad `except Exception`) and the question got **no SPARQL
query at all** — not even the model's best partial attempt. The model had
typically validated a working query early in the trace and then kept
"polishing" past the point of diminishing returns.

**(b) Transient KIT proxy errors aborting the whole question.** KIT
intermittently returns `"Open WebUI: Server Connection Error"` (surfaced as an
HTTP 400) or similar transient failures from `chat.completions.create`. Prior
to this change, any exception from that call propagated up and aborted the
entire question — a single flaky proxy response threw away a question that
otherwise would have succeeded.

## Decision 1 — Tool-call budget becomes the binding limit, not the wall-clock

**Chosen:** Lower `WikidataAgent`'s `max_tool_calls` from 30 to **20** and
`max_iterations` from its prior value to **24**, and treat the framework's
existing "force synthesis at `max_tool_calls`" mechanism
(`base_agent.py` ~line 1483-1498) as the *primary* stopping condition for this
agent. Raise `AgentSparqlGenerator.agent_timeout` from 180s to **280s** and
recharacterize it as a rare safety net rather than the primary limiter.

**Rejected:** Keeping the wall-clock as the primary limiter and just raising
its value. A longer timeout delays the failure but does not fix it — an
over-exploring model will eventually hit any fixed wall-clock and still get
cancelled with nothing. The actual bug is that the *cancel path* discards the
model's already-validated query.

**Rationale:** The framework already force-synthesizes (calls
`_run_synthesis()` to emit the best validated answer) the moment
`total_tool_calls_made >= max_tool_calls` — this mechanism exists in the
shared framework and is used by KQAPro/SciQA, but WikidataAgent's tool-call
cap (30) was set high enough, and the wall-clock low enough (180s), that the
wall-clock was firing *first* on the over-exploration cases. Lowering the
tool-call cap to 20 makes it the limit that actually binds in practice, so the
agent now always exits through `_run_synthesis()` — which emits a real SPARQL
query — rather than through the timeout cancellation path, which emits
nothing. Raising the wall-clock to 280s gives the (now-binding) tool-call cap
room to be reached normally before the safety net would ever fire.

**Implications:**
- `ama_kbqa/agents/wikidata_agent/agent.py`: `max_tool_calls: 20`,
  `max_iterations: 24`.
- `ama_kbqa/wikikgqa/generator.py`: `AgentSparqlGenerator.agent_timeout`
  default `180.0` → `280.0`.
- The model needs to know about the budget to plan around it — see Decision 3.

---

## Decision 2 — Per-tool SPARQL timeout with a steering message, not a hang or generic error

**Chosen:** `RunSPARQL` in `wikidata_server.py` enforces
`_TOOL_SPARQL_TIMEOUT = 30` seconds via `execute(probe, timeout=...)`. On
timeout it returns a message that both names the failure and tells the model
what to do differently:

> `TIMEOUT: this query did not return within 30s. It is probably too
> expensive or matches too many rows. Make it MORE SPECIFIC: add a tighter
> type constraint or FILTER, restrict the entity set, or add a LIMIT for
> inspection, then retry. Do not re-run the same broad query.`

**Rejected:** Letting an expensive query hang until some outer timeout, or
returning a bare `ERROR: query failed` with no guidance.

**Rationale:** A generic error invites the model to retry the identical
query (it doesn't know *why* it failed). The explicit "do not re-run the
same broad query" instruction directly targets the observed over-exploration
pattern of re-issuing near-identical broad queries.

---

## Decision 3 — Surface the live budget to the model instead of letting it discover the cap only via failure

**Chosen:** A `_budgeted` decorator on the exploration tools appends a live
`[tool call N/20]` tag to each tool's string result (module-global
`_tool_call_count`, reset per question because the MCP server process is
spawned fresh per question), escalating to an explicit "near budget, commit"
warning from call 16 onward. The system prompt also states the ~20-call
budget up front (`SYSTEM_PROMPT` `BUDGET:` line in
`ama_kbqa/agents/wikidata_agent/prompts.py`).

**Rejected:** Relying solely on the framework's force-synthesis cutoff
without telling the model the cap exists. This would still recover from
failure mode (a) — the framework would emit a query regardless — but would
waste calls on a model that doesn't know it's being timed and continues
exploring past the point where it should have committed, producing a worse
(less-polished-but-not-actually-needed) version of the same problem.

**Implementation note:** the decorator skips JSON-looking results (`out.lstrip()[:1]
in "{["`) so it never corrupts `GetJournalStateJSON`'s machine-readable
output, and uses `functools.wraps` so FastMCP's tool-schema introspection
(which reads the wrapped function's signature) is unaffected.

---

## Decision 4 — Retry transient LLM provider errors instead of aborting the question

**Chosen:** `BaseKBQAAgent._llm_call` wraps `chat.completions.create` in up to
3 attempts with linear backoff, retrying only on a transient-error pattern
match (`"server connection error"`, `"open webui"`, `"connection error"`,
`"timeout"`/`"timed out"`, `"502"`/`"503"`/`"504"`, `"overloaded"`,
`"temporarily unavailable"`, `"429"`/`"rate limit"`). Non-matching exceptions
(deterministic errors — bad request, auth) re-raise immediately without
retry.

**Rejected:** A blanket retry-on-any-exception (would mask real bugs and
waste time retrying deterministic failures that will never succeed) and
leaving this only in the benchmark-runner-level watchdog
(`Decisions/kit-outage-watchdog.md`) without an LLM-call-layer retry (the
watchdog catches *consecutive* infra failures across questions and aborts the
whole run; it does not recover a single flaky call within one question).

**Rationale:** This is a **shared framework change** — it lives in
`base_agent.py`, so it benefits KQAPro and SciQA as well as WikidataAgent, not
just this challenge. It complements the existing watchdog: the watchdog is a
last-resort circuit-breaker across questions, this retry recovers individual
transient blips without ever reaching the watchdog's failure count.

**Companion fix (same diff, not separately decided but worth recording):** a
response with no `choices` (observed on OpenRouter/Gemma after a rejected
malformed tool call) previously crashed on `response.choices[0]`. The loop
now tracks `empty_response_count`, nudges the model to retry with valid
arguments or a final answer, and breaks to synthesis after 3 consecutive
empty responses.

---

## Validation

**Unit/integration tests (2026-06-30):** 199 tests passing: `uv run pytest
tests/framework tests/server tests/wikikgqa -q`.

**Live re-run, commitment fix confirmed (2026-07-01):** Re-ran the Wikidata
agent with model `kit.qwen3.5-397b-A17b` on the 5 questions that previously
emitted **no query at all** under the old wall-clock-primary regime — q3,
q49, q183, q279, q302 — this time with the new tool-call budget
(`max_tool_calls=20`), the per-tool `RunSPARQL` timeout, the live
`[tool call N/20]` counter, and the `_llm_call` transient-error retry all
active.

Result: **all 5 now emit a valid, executing query — zero empties.** This is
the direct confirmation of Decision 1 (tool-call budget binds before the
wall-clock, so the agent always exits through `_run_synthesis()`).

Per-question outcome:

| Question | Result | Notes |
|---|---|---|
| q3 | F1 = 1.0 | Clean commit |
| q49 | F1 = 1.0 | Used the R10 `wdt:P171*` taxonomy path correctly |
| q183 | F1 = 1.0 | Previously the canonical over-exploration case — a 16-iteration wall-clock timeout that cancelled with nothing; now commits cleanly |
| q279 | F1 = 1.0 | Clean commit |
| q302 ("largest moon by mass") | Wrong answer, but **now commits a valid query** | Failure mode has shifted from *commitment* (this ADR's bug) to *content*: needs the R8 normalized-quantity path plus `P397+` → Sun scoping. Tracked as an answer-conventions issue, not a regression of this fix — see `System/wikikgqa_agent.md` R1-R10 conventions |

Net: **+4.0 F1** across these 5 previously-empty questions. q302's remaining
gap is a separate, pre-existing convention coverage issue (out of scope for
this ADR).

**Conclusion:** the commitment fix works as designed. Qwen's empties were
**process failures** — over-exploration hitting the wall-clock cancel path,
compounded by transient KIT proxy errors — not knowledge failures. The model
had the right answer reachable in 4/5 cases and was simply never given the
chance to emit it before this fix.

## Open Questions

| Question | Notes |
|---|---|
| Does `max_tool_calls=20` ever truncate a genuinely-needed exploration (false negative)? | Not observed in the 5-question re-run (0/5 truncated short of a correct answer on commitment grounds); still not measured on a larger held-out set |
| Should the transient-error retry pattern list be config-driven rather than hardcoded? | Currently a fixed string-match list in `base_agent.py`; fine for now, revisit if KIT's error vocabulary changes |
| q302-class content gaps (normalized-quantity comparisons + implicit scoping like "moon" → "of the Sun's system") | Separate from this ADR's commitment fix; candidate for a follow-up ADR or `ANSWER_CONVENTIONS.md` (R8) hardening pass |

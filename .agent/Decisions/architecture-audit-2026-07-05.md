# Architecture & Implementation Audit — 2026-07-05

**Date:** 2026-07-05
**Branch:** `dev-routing-evidence`
**Auditor:** Claude (with docs-agent + wiki cross-check)
**Status:** Findings triaged and decided with user. FIX items not yet implemented as of this writing — see CHANGELOG for landing dates.
**Prior art:** Some items independently flagged in a 2026-07-03 wiki audit (`wikikgqa-implementation-audit-2026-07-03.md`) and re-verified live during this pass.

## Related Docs
- [../System/project_architecture.md](../System/project_architecture.md) — MCP servers, tool catalogs, config system
- [../System/agent_system.md](../System/agent_system.md) — agent loop, loop detection, journal model, orchestrator routing
- [orchestrator-evidence-based-routing.md](orchestrator-evidence-based-routing.md) — the routing rework this audit's B4/C2 items build on
- [../Tasks/active/deferred-tool-loading.md](../Tasks/active/deferred-tool-loading.md) — scheduled follow-up task from C3

---

## Purpose

This is the authoritative record of an 18-point audit across correctness/robustness, architecture, performance, and housekeeping. It exists so that:

1. FIX items have a documented rationale and fix sketch before implementation.
2. LEAVE / WONTFIX items are **not re-discovered from scratch** in a future audit — the reasoning for deferring or rejecting them is recorded here, with an explicit trigger condition for reopening where one exists.

## Summary

12 FIX · 6 LEAVE · 1 WONTFIX · 1 SCHEDULED TASK

| ID | Area | Decision | One-line summary |
|----|------|----------|-------------------|
| B1 | Correctness | FIX | Max-iter/zero-tool hard-stops bypass synthesis, return raw error |
| B2 | Correctness | FIX | Unguarded synthesis I/O throws instead of degrading |
| B3 | Correctness | FIX | Malformed tool-call args silently become `{}` and execute |
| B4 | Correctness | FIX (observability only) | Routing silently defaults to KQAPro on any failure |
| B5 | Correctness | LEAVE | Loop-detection false positives on intended refinement |
| A1 | Architecture | FIX | Three near-duplicate MCP clients, diverged |
| A2 | Architecture | LEAVE | Duplicated server infra across kqapro/sciqa servers |
| C7 | Architecture | FIX | SciQA embedding cache missing (user-requested) |
| C1 | Performance | FIX | Parallelize the benchmark runner |
| C2 | Performance | FIX (a+c only) | Collapse routing from 3 LLM round-trips to 1 |
| C3 | Performance | SPLIT: FIX (cheap filtering) + SCHEDULED (two-tier loading) | Tool-schema token overhead |
| C4 | Performance | LEAVE | Async DB clients + parallel independent tool calls |
| C5 | Performance | LEAVE | Embedding path network hop removal |
| C6a | Performance | FIX | MCP close() 700ms sleep + probe server per-question churn |
| C6b | Performance | LEAVE | `_manage_context_window` full char-sum recompute per iteration |
| C6c | Performance | FIX | `GetSchemaForAttribute` re-embeds candidates every call |
| H1 | Housekeeping | WONTFIX | Leaked OpenRouter key, ~€2, not rotated |
| H2 | Housekeeping | FIX | `.mcp.json` untracked — add to `.gitignore` |

---

## Correctness / Robustness

### B1 — Max-iterations & zero-tool hard-stop return raw error string, bypassing synthesis

- **Location:** `framework/base_agent.py:1238` (max-iter), `framework/base_agent.py:1405` (zero-tool hard stop)
- **Problem:** Both exit paths return a raw error string directly instead of routing through synthesis. The full journal (found_values, verified_facts) is discarded even when the answer is already present in it — guaranteeing a 0 score on the benchmark judge for a question the agent had actually solved.
- **Decision:** FIX (2026-07-05).
- **Rationale:** Funnel both exits through a guarded `_run_synthesis(query, qtype)` call so the journal is used before giving up. The zero-tool path may still fall back to an error/soft-IDK response if synthesis itself has nothing to work with — that's a legitimate "no evidence" case, not a bug. Land together with B2 (same code path, same guard needed).

### B2 — Unguarded synthesis I/O

- **Location:** `framework/base_agent.py:1891` (`GetJournalSummary` call has no try/except), `framework/base_agent.py:~2283` (`response.choices[0]` accessed unguarded)
- **Problem:** A hiccup on the final synthesis step (empty choices, provider error) throws an exception for the whole question instead of degrading gracefully.
- **Decision:** FIX (2026-07-05).
- **Rationale:** Fall back to the last `journal_snapshot` / `final_agent_content` instead of raising. Bundled with B1 since both land in the same synthesis-hardening pass.

### B3 — Malformed tool-call args silently become `{}` and execute

- **Location:** `framework/base_agent.py:1546-1547`
- **Problem:** A JSON parse failure on tool-call arguments is swallowed and the tool executes with `{}`, producing a phantom empty-arg call that burns an iteration and forces the model to recover from a confusing result instead of being told parsing failed.
- **Decision:** FIX (2026-07-05).
- **Rationale:** On `JSONDecodeError`, skip execution, return the parse error as the tool result, and `continue` the loop. Cheap, contained, no behavior-changing side effects beyond making the failure visible to the model.

### B4 — Routing silently defaults to KQAPro on any failure, no signal

- **Location:** `agents/orchestrator_agent/agent.py:423-424`; `_route_autonomously` returns `None` on any exception, which triggers `_fallback_kqapro`
- **Problem:** A transient router-LLM error misroutes all SciQA questions to KQAPro invisibly — nothing in logs or traces distinguishes "routed by evidence" from "fell back after a failure."
- **Decision:** FIX, **observability half only** (2026-07-05).
- **Rationale:** Add a structured `route_outcome` span attribute plus a benchmark-summary counter distinguishing routed vs. fell-back. The KQAPro-as-default fallback itself is **retained intentionally** — the question domain skews general-knowledge/KQAPro, so defaulting there is the statistically safer choice on failure. Making routing *retry* instead of falling back immediately is deferred into C2 (routing round-trip work), not duplicated here.

### B5 — Loop-detection false positives on intended refinement workflow

- **Location:** `framework/base_agent.py:746-773` — tool-call caps (`FindResource` > 8, `RunORKGSPARQL` > 10) and the "same tool 5x within 6 iterations" detector
- **Problem:** Detections 1-2 (identical-3x repeat, oscillation) are precise and considered good. Detection 3 ("5x/6 iters") and the hard per-tool caps are blunt count-only checks that can trip a legitimate SciQA iterative-refinement workflow (distinct args, journal progressing) purely on call count.
- **Fix sketch (deferred, not applied):** Make the blunt detectors progress-aware — only trip when the journal (`found_values`/`verified_facts`) is unchanged across the repeated calls. The signal is already available via journal snapshots; this is a real, buildable fix, just not applied now.
- **Decision:** LEAVE FOR NOW (2026-07-05).
- **Rationale:** Speculative without a benchmark A/B — the caps are load-bearing per the wiki's prior thrashing-tail fix, and hastily loosening them risks reintroducing the runaway-loop failure mode they were added to close. **Trigger to reopen:** SciQA accuracy plateaus *and* trace evidence shows the cap forcing synthesis on productive runs (distinct args, journal advancing across the repeated calls).

---

## Architecture

### A1 — Three near-duplicate MCP clients, diverged

- **Location:** `framework/mcp_client.py` vs. the inline `MCPClient` class in `orchestrator_agent/agent.py:56-123`
- **Problem:** Maintenance trap — the two implementations have already diverged (`close()` sleep 0.7s vs 0.05s), and a prior env-passthrough fix had to be manually double-applied across both (commit `90915e4`). Silent divergence risk on every future client-level fix.
- **Decision:** FIX (2026-07-05).
- **Rationale:** Delete the inline class, import the framework `MCPClient` instead. Requires a live routing smoke-test before landing since the orchestrator's routing path is critical/user-facing. May be bundled with C6a (MCP close-sleep + probe-server lifecycle cleanup) since both touch the same client lifecycle code.

### A2 — Duplicated server infrastructure across kqapro/sciqa servers

- **Location:** `server/kqapro_server.py`, `server/sciqa_server.py` — duplicated embedding+cache plumbing, `log_tool_duration`, URI formatting, SPARQL compaction
- **Fix sketch (not applied):** Extract `server/_server_base.py` with KG-agnostic plumbing only — explicitly do **not** change any tool return shape or text, since that would move benchmark numbers (envelope unification was deliberately deferred per the abstract-operations-contract stance).
- **Decision:** LEAVE FOR NOW (2026-07-05).
- **Rationale:** Modest standalone payoff against a high blast radius — both server files are ~5k lines on the answer-generating critical path. **Revisit** as a prerequisite when picking up C5 (embedding path work), since C5 would otherwise touch the same code twice. If/when done, requires a before/after byte-equality check on tool outputs to guarantee no benchmark drift.

### C7 — SciQA embedding cache missing (RAG step uncached) [user-requested]

- **Location:** `server/sciqa_server.py:956-964` — `get_embedding` has no cache; callers at `:1073` and `:1231` (entity and relation vector search) pay a fresh network embedding call every time. Contrast with `server/kqapro_server.py:449-474`, which has a 256-entry LRU `_embedding_cache` with eviction.
- **Problem:** SciQA repeats a network round-trip for terms it has already embedded in the same run, a direct and avoidable latency cost on the RAG lookup path.
- **Decision:** FIX (2026-07-05, user-requested).
- **Rationale:** Port the KQAPro cache into SciQA's `get_embedding` inline for now. Explicitly noted: this re-duplicates infrastructure that A2 wants to eventually unify — when A2/C5 land, the cache moves into the shared base. Consider extending coverage to other expensive/repeated ops (e.g. schema-attribute embeds, ties to C6c) under the same cache mechanism.

---

## Performance (ranked by impact/effort)

### C1 — Parallelize the benchmark runner

- **Location:** `benchmark_agents.py` — serial for-loop, single shared agent + `soft_reset` between questions
- **Problem:** Serial execution on an I/O-bound workload (LLM + tool calls) leaves throughput on the table; this is the pacing bottleneck for the whole research loop.
- **Fix sketch:** Agent pool + `asyncio.Semaphore(K)`, respecting provider rate limits.
- **Decision:** FIX (2026-07-05).
- **Rationale:** Concurrency `K` exposed as **both** a CLI flag and a config key, with CLI overriding config. **Default = 1** (serial) — preserves current behavior exactly; concurrency is opt-in only. Each agent must be fully isolated (own MCP subprocess, no shared mutable state). The existing watchdog (5-consecutive-failure abort) needs adapting since consecutive-failure counting assumes serial execution and needs to work under concurrent, non-serial failures.

### C2 — Collapse routing from 3 LLM round-trips to 1

- **Location:** `orchestrator_agent/agent.py:_route_autonomously`, `server/orchestrator_server.py:extract_semantics`
- **Sub-levers:**
  - **(a)** Invoke `analyze_query_recommend_db` deterministically, skipping the forced step-1 LLM call, and pass evidence straight into the single `select_agent` call. Saves 1 LLM round-trip.
  - **(b)** Replace `extract_semantics`'s LLM-based NER with local noun-chunking or a whole-question embedding. Saves 1 round-trip, but can shift which agent gets picked.
  - **(c)** `asyncio.gather` the Qdrant probes (currently a serial collections×terms loop).
- **Decision:** FIX **(a) + (c) only** (2026-07-05). **(b) deferred** as a separate, validated experiment.
- **Rationale:** (a) and (c) are pure latency wins with no effect on the routing decision itself — safe to land directly, but must re-verify routing still works afterward since this is the live critical path (same evidence-based-routing ADR territory). (b) changes *which agent gets picked*, so it needs a labeled-set routing A/B before landing, and touches the newest [orchestrator-evidence-based-routing.md](orchestrator-evidence-based-routing.md) ADR's guarantees — not something to bundle into a latency-only pass.

### C3 — Attack 39.7% tool-schema token overhead (deferred tool loading)

- **Location:** `framework/mcp_client.py:convert_tools_to_openai_format` — the full tool catalog is resent on every `_llm_call`
- **Problem:** Per a prior (2026-04-05) feasibility study, tool schemas make up ~39.7% of token overhead per call; ~33% token reduction is achievable, with a direct cost + latency win.
- **Decision:** SPLIT (2026-07-05):
  - **Cheap filtering → FIX NOW.** Audit `_get_allowed_tools_for_qtype` per question type and shrink each qtype's allowed tool set to what it actually needs. Zero new machinery. Must be benchmarked before/after since tool visibility changes agent behavior, not just token count.
  - **Deferred/two-tier tool loading → SCHEDULED AS TASK, not this pass.** See [../Tasks/active/deferred-tool-loading.md](../Tasks/active/deferred-tool-loading.md) for the full design.
- **Rationale:** The cheap filtering captures much of the win with no new failure surface. The two-tier design is a bigger, benchmarked, multi-day change and belongs in its own tracked task rather than folded into this audit's immediate FIX batch.

### C4 — Async DB clients + parallel independent tool calls

- **Location:** `_execute_tool_calls` serial for-loop (`base_agent.py:1524`); sync `QdrantClient` (has a `# TODO` marker), `SPARQLWrapper`
- **Sub-items:** (i) parallelize multi-tool turns via `asyncio.gather`; (ii) migrate to `AsyncQdrantClient` + async SPARQL/HTTP.
- **Decision:** LEAVE FOR NOW (2026-07-05).
- **Rationale:** Most turns emit a single tool call, so there's nothing to parallelize in the common case; in-tool chains (embed → search → resolve) are data-dependent and can't be parallelized anyway. Question-level parallelism (C1) captures the real throughput win at far lower risk. Risk if attempted: (i) collides with order-sensitive loop detection (`tool_call_history`/`tool_sequence`) and per-mutating-tool journal snapshots — would need read-only-only parallelism or an order-independence guarantee; (ii) (ii) is a broad async migration across two ~5k-line servers. **Revisit** only if traces show frequent multi-independent-tool turns; if reopened, restrict parallelism to read-only tools first.

### C5 — Embedding path: remove network hop per entity search

- **Location:** `server/kqapro_server.py:get_embedding` (256-entry per-process dict cache)
- **Options considered:** (a) persist the API-vector cache to disk (no re-index needed); (b) switch to a local sentence-transformers model (requires a full Qdrant re-index).
- **Decision:** LEAVE (2026-07-05).
- **Rationale:** User does not want a local/persistent copy of embeddings. The network embed call is not the dominant cost in the pipeline, so the re-index risk of option (b) isn't justified by the payoff. Note this is distinct from C7: the SciQA in-memory cache fix under C7 is still a FIX — that's runtime parity with KQAPro's existing behavior, not a persistent store, and doesn't touch this decision.

### C6 — Cheap cleanups (split into three sub-decisions)

- **Sources:** MCP `close()` hardcoded 700ms sleep (`mcp_client.py:199,221`); orchestrator opens/closes a probe MCP server per question (`agent.py:427-430`); `_manage_context_window` recomputes the full char sum every iteration (`base_agent.py:1800`); `GetSchemaForAttribute` re-embeds all candidate attributes on every call, no cache (`kqapro_server.py:786-806`)

#### C6a — MCP close() 700ms sleep + orchestrator probe server opened/closed per question

- **Location:** `mcp_client.py:199,221`; `agent.py:427-430`
- **Decision:** FIX (2026-07-05).
- **Rationale:** Keep the orchestrator's probe MCP connection open across the session the same way sub-agents do, and trim the hardcoded sleeps. Pairs naturally with A1 (client consolidation) since both touch client lifecycle code.

#### C6b — `_manage_context_window` full char-sum recompute per iteration

- **Location:** `base_agent.py:1800`
- **Decision:** LEAVE (2026-07-05).
- **Rationale:** Micro-optimization; not worth the churn/risk relative to its actual cost.

#### C6c — `GetSchemaForAttribute` re-embeds candidate attributes every call

- **Location:** `kqapro_server.py:786-806`
- **Decision:** FIX (2026-07-05).
- **Rationale:** Memoize / embed once at startup instead of re-embedding candidates on every call. Rides on the C7 cache work (same embedding-cache pattern, same PR is a reasonable bundling).

---

## Housekeeping (out of code scope)

### H1 — Leaked OpenRouter key still compromised, needs rotation (ROADMAP)

- **Decision:** WONTFIX (2026-07-05).
- **Rationale:** The leaked key holds only ~€2 of balance; rotation isn't worth the hassle relative to the exposure per the user's explicit call. Documented here as a **deliberate accepted risk**, not an oversight — do not re-flag this as a fresh security finding in a future audit without new information (e.g., the key gains higher balance/scope, or is found to be reused elsewhere).

### H2 — `.mcp.json` untracked in working tree — commit or gitignore?

- **Decision:** FIX (2026-07-05).
- **Rationale:** `.mcp.json` should not be committed (likely contains local/environment-specific MCP config) — add it to `.gitignore` instead of leaving it as a perpetually-untracked file that shows up in every `git status`.

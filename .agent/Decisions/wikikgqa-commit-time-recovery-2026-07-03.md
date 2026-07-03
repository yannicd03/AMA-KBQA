# ADR: WikiKGQA Commit-Time Non-Empty-Prior Recovery, ASK Self-Consistency, Run Manifests

**Date:** 2026-07-03
**Commit:** `9c2a431` "WikiKGQA: exploit non-empty prior at commit time, ASK self-consistency, run manifests"
**Status:** Settled — implemented, 270 tests passing; strategic recommendations from a full implementation audit, several items still open (see below)

## Related Docs
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference for `AgentSparqlGenerator`; update this ADR alongside it if the recovery/voting mechanics change again
- [Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md](./wikikgqa-synthesis-context-ab-2026-07-02.md) — prior-day decision on the same generator (synthesis context default)
- [Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md](./wikikgqa-tool-budget-and-resilience-2026-06-30.md) — the journal-recovery mechanism this decision builds on (evidence-based verify-on-empty repair)
- **External:** `0_Claude/AMA-KBQA/wikikgqa-implementation-audit-2026-07-03.md` in the user's Obsidian wiki — the full implementation audit this commit implements strategic recommendations from. Read that note for the complete audit findings; this ADR only records what was *implemented* and what remains *open*.
- [Decisions/wikikgqa-conventions-default-2026-07-03.md](./wikikgqa-conventions-default-2026-07-03.md) — final results of the held-out seed-99 conventions A/B tracked as in-flight below; this later ADR has the full analysis

---

## Background

A full implementation audit of the WikiKGQA generator/agent pipeline (recorded
in the wiki, not duplicated here) surfaced that the journal-recovery fallback
introduced in the 2026-06-30 tool-budget ADR was not actually firing in one
important case: when the agent's final synthesis output was an **error
string** (e.g. `"Error: Agent reached maximum iteration limit."`) rather than
empty text. `strip_sparql` had no SPARQL-keyword/fence match to work with, so
it fell through to returning the raw text — which was then **executed as a
SPARQL query**, failed, and the question was scored as a hard miss even
though a validated query existed in the journal from earlier in the
exploration.

The audit also identified two independent opportunities: (1) a domain fact —
**every gold answer in this benchmark is non-empty** — was not being
exploited at commit time; a 0-row result from the committed query is
therefore *known-wrong*, not merely suspicious. (2) ASK (boolean) questions
are a small (~7%) but measurably noisier slice, a natural candidate for
self-consistency voting.

## Decisions

### 1. `strip_sparql` returns `""` on no-match, instead of passing prose through

**Chosen:** `ama_kbqa/wikikgqa/generator.py::strip_sparql` now returns `""`
when the input text has neither a fenced code block nor a SPARQL keyword
match. Previously it returned the raw text unchanged.

**Rationale:** An empty return is the generator's existing signal for "no
query produced," which downstream code already handles via journal recovery.
Returning prose that *looks* like it might parse (but isn't SPARQL) silently
defeated that recovery path — the exact bug motivating this change. Empty
output is not new behavior for callers; it's making an already-handled case
fire correctly.

### 2. Non-empty-prior fallback at commit time

**Chosen:** In `AgentSparqlGenerator._generate_once`, after executing the
committed query: if it errors or returns 0 rows, **and** the journal holds a
different validated query from exploration (`recovered`), execute that
alternate query too. If the alternate query has rows, keep it as the final
answer instead of the committed 0-row/error result.

**Rationale:** 0/462 gold answers in this benchmark are empty, so a
committed query that returns 0 rows is not "possibly correct, possibly an
edge case" — it is known-wrong. Falling back to a validated-but-not-committed
query that *did* return rows is strictly better than keeping a result that
provably cannot match the gold answer.

**Rejected:** Always preferring the last successful query in the journal
regardless of what was committed (would discard the model's own judgment
about which query best answers the *specific* question, not just which
query happened to return rows — a query with rows can still answer the wrong
sub-question).

### 3. ASK self-consistency voting

**Chosen:** `AgentSparqlGenerator(ask_votes=N)` / `benchmark.py --ask-votes N`
(default `1` = off). When the committed query executes to a boolean (`ASK`
result), and `ask_votes > 1`, the full generation is re-run up to
`ask_votes` times total; the boolean results are majority-voted. Ties (e.g. a
re-run that doesn't produce a boolean at all) keep the first run's answer.
Non-boolean (SELECT) results are never re-run — voting only applies to the
ASK slice.

**Rationale:** targets the ~7% ASK slice specifically identified in the audit
as unusually nondeterministic, without adding cost to the much larger SELECT
majority.

### 4. Run manifests

**Chosen:** `benchmark.py` writes `run_manifest.json` into the run's output
directory: git commit hash (`git rev-parse HEAD`), ISO timestamp, resolved
SPARQL endpoint, and the full parsed CLI args. Mirrors the existing
`benchmark_agents._write_run_manifest` pattern used elsewhere in the repo.

**Rationale:** closes a runs-to-commits mapping gap — without this, a
`summary.json` could not be reliably attributed to a code version, model,
conventions setting, or seed after the fact, which had already caused
unreliable failure-mode analysis on at least one prior occasion.

## Validation

Tests: +34 new/extended (`tests/wikikgqa/test_nomentions.py`,
`test_endpoint.py`, `test_benchmark.py` new; `test_generator.py` extended).
`uv run pytest tests/wikikgqa tests/framework tests/server -q` = **270
passed** (up from 199 as of the 2026-06-30 ADR).

**Held-out seed-99 `conventions` A/B — completed.** The held-out seed-99
`conventions` A/B (full R1-R10 vs minimal R1-R6, model
`kit.qwen3.5-397b-A17b`, `rand-50`, challenge endpoint) launched 2026-07-03
through this commit's recovery/voting code path has finished: minimal 0.7311
Macro F1 vs full 0.6833, both 50/50 exec_ok. R7-R10 show no held-out benefit;
the default flipped to minimal. Full results, the noise-dominated
sign-test/trajectory-divergence analysis, and the implementation diff are in
[Decisions/wikikgqa-conventions-default-2026-07-03.md](./wikikgqa-conventions-default-2026-07-03.md)
— this note is retained only to mark that the run referenced above completed.
Output directories: `benchmark_results/wikikgqa/run-rand50-seed99-conv-full/`
and `run-rand50-seed99-conv-minimal/`.

## Open Items (not yet fixed — from the audit, tracked here for follow-up)

The audit identified further paths that can still discard a validated query
before it reaches the recovery logic above. None of these were fixed in this
commit:

| Location | Issue |
|---|---|
| `ama_kbqa/framework/base_agent.py:1289-1296` | Max-iterations exit path |
| `ama_kbqa/framework/base_agent.py:1330-1332` | Unhandled mid-loop LLM exception |
| `ama_kbqa/framework/base_agent.py:1975`, `:2283` | Unguarded synthesis call sites |
| `ama_kbqa/agents/wikidata_agent` loop-detection (approx. `base_agent.py:798-806`) | False-positive loop detection on repeated `RunSPARQL` calls that are legitimately similar (e.g. incremental narrowing), not actual stalls |
| Budget accounting | Double-bookkeeping between the tool-call budget counter and the framework's own `total_tool_calls_made` — not reconciled |
| Local scorer vs Codabench | The local macro-F1 scorer is stricter than the Codabench submission scorer on at least one class of answers (exact class not re-derived here — see the wiki audit for detail) |

These four `base_agent.py` paths are **framework-level**, so a fix benefits
KQAPro/SciQA too, not just WikiKGQA — same shared-framework pattern as the
2026-06-30 ADR's `_llm_call` retry. Candidate for a follow-up ADR once
prioritized.

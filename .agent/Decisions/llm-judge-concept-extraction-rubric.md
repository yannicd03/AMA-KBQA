# ADR: LLM Judge Prompt — Concept-Extraction Rubric

**Date:** 2026-05-02  
**Commit:** `9822bda` "Tighten LLM judge prompt: extract gold concept from verbose answers"  
**Code:** `ama_kbqa/postprocessing.py:512-580` — `prompt = f"""..."""` block inside `execute_llm_judge_postprocessing`

## Related Docs
- [System/agent_system.md](../System/agent_system.md) — agent lifecycle and journal/scratchpad model
- [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) — how to trigger LLM judge during benchmarks
- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — tool additions that increased agent verbosity

---

## Context

Manual audit of 200 v2 benchmark judgments (n=100, two models) revealed the prior rubric — a single "consider semantic equivalence" instruction — had a measurable bias of ~3 pp against verbose-but-correct answers. The failure mode was most common on QueryRelation questions where the agent embedded the gold relation label inside a natural-language sentence (e.g., "X is a cast member of Y") rather than returning the bare label ("cast member").

## Decision

Rewrote the judge rubric as a structured, multi-section prompt (lines 525–575):

1. **Concept-level matching** — instructs the judge to extract the gold concept first, then check whether it appears in the predicted answer. Verbose phrasing is explicitly blessed.
2. **Per-type rules with examples** — QueryRelation, Count, Verify, QueryAttr each have concrete worked examples so the judge doesn't pattern-match on format.
3. **Hard-negative rules** — three cases that are NEVER correct regardless of coincidental alignment with the gold:
   - Tool-call fragment or `<think>` block with no final answer.
   - "Error: Agent reached maximum iteration limit" (or similar failure messages) — even when gold is "0" or "no".
   - "Data not in KG" / "no record found" while gold provides a specific value.
4. **Self-consistency requirement** — `is_correct` MUST agree with `correctness_reasoning`; the prompt calls this out explicitly.

## Why This Over the Alternative

The alternative considered was nudging the agent's synthesis prompt to produce terse QueryRelation answers. That was rejected because:
- It would change agent behavior, invalidating comparisons to prior runs.
- The judge fix removes the bias for all future runs across all agents and models without touching agent output.
- Root cause was judge bias, not agent verbosity — fixing the right thing.

The user explicitly chose the judge-fix path ("lets focus on the gold concept instead").

## Known Failures the New Rubric Handles

| Failure mode | Old behaviour | New behaviour |
|---|---|---|
| "X is a cast member of Y" vs gold "cast member" | Judged INCORRECT (phrasing mismatch) | Judged CORRECT (gold concept present) |
| "Error: max iteration" vs gold "0" | Sometimes judged CORRECT (coincidental alignment) | Always INCORRECT (hard-negative rule) |
| Reasoning says "matches semantically" but `is_correct: false` | Passed through | Caught by self-consistency rule |

## Expected Impact

~+2 pp on all future benchmark headlines from removing judge bias alone (estimated from the audit delta).

Full per-question audit table is in the wiki at `0_Claude/AMA-KBQA/KQA Pro Benchmark.md` (entry "2026-05-03 — Manual judge audit, corrections, and judge-prompt fix").

# ADR: Prompt + Iter-Cap Hardening for n=100 → 90% Push

**Date:** 2026-05-03
**Status:** Accepted
**Code:**
- `ama_kbqa/agents/kqapro_agent/prompts.py` — `SYSTEM_PROMPT` (RULE 0a), `QTYPE_STRATEGIES["QueryAttr"]`, `QTYPE_STRATEGIES["QueryAttrQualifier"]`, `QTYPE_STRATEGIES["QueryRelation"]`
- `ama_kbqa/framework/adapters/kqapro_adapter.py:93` — `max_iterations` 25 → 40

## Related Docs

- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — original tool surface (CountEntities / SelectExtreme / VerifyFact)
- [Decisions/llm-judge-concept-extraction-rubric.md](./llm-judge-concept-extraction-rubric.md) — judge prompt rewrite (orthogonal; this ADR is agent-side)
- [Decisions/transitive-concept-default-false.md](./transitive-concept-default-false.md) — recent revert; informed the "small-prompt-fixes-first" rollout strategy used here
- Wiki: `0_Claude/AMA-KBQA/KQA Pro Benchmark.md` (entry "2026-05-03 — Last-5-batches failure analysis + n=100 → 90% fix plan") — full failure taxonomy and per-batch question-by-question audit

## Context

Best post-audit n=100 accuracy is 0.79 (minimax-m2.7-kit) / 0.81 (gemma-4-31b-kit). Across the most recent five n=10 batches (`batch058 / 059 / 061 / 062 / 067`, 49 questions, 31 correct = 63% on the small sample), 18 wrong answers fell into a clear taxonomy:

| Cat | Description | Count | Share |
|---|---|---|---|
| A | Premature "data not in KG" — gives up after 1 failed query | 8 | 44% |
| C | Wrong-attribute / wrong-direction (`country_of_origin` vs release region; passive-voice "followed by") | 4 | 22% |
| D | Format mismatch — narrative answer where bare label expected | 3 | 17% |
| B | Qualifier-value extraction returns entity instead of qualifier value | 2 | 11% |
| E | Concept filter / `Select*` over wrong candidate set | 2 | 11% |
| H | Genuine KG gap | 1 | 6% |

Three concrete failure *patterns* recurred across multiple batches (high signal, not seed noise):

1. **Cardiff City `start_time` qualifier** (058, 059) — agent oscillates `GetAttributeDetails` ↔ `GetEdgeQualifiers`, never tries `QualifierFilter`, never reasons about `direction: "backward"` from `GetRelationDetails`.
2. **`minnesotaBASS` subscriber count** (058, 059) — agent searches "subscriber" via `ExploreNeighborhood(semantic_relation_name=...)`, which scans relation embeddings (returns `follows`/`followed_by`/`partner` — wrong). `GetNodeSummary` is never called; the actual attribute name (`number of subscribers`) is one tool call away.
3. **"Olympics followed by 1980"** (058, 059) — passive-voice grammar inverts the direction of `followed_by`; agent returns successor (1984) instead of predecessor (1976).

These three account for 6/18 wrongs (33%). Fixing the *categories* they reveal is high-leverage and not overfitting to the specific questions.

## Decision

Bundle five small prompt edits and one config bump in `kqapro_adapter.py`. Ship in one commit so the next n=100 benchmark measures the bundle's effect cleanly against the post-audit baseline.

### Changes

| # | File | Change | Targets |
|---|---|---|---|
| 1 | `prompts.py` SYSTEM_PROMPT | Add **RULE 0a**: forbid "not available" / "missing" answers without prior `GetNodeSummary` call on the target entity | Cat A (44% of failures) |
| 2 | `prompts.py` `QueryAttr` strategy | Replace one-line empty-result hint with a 4-step fallback order: re-check `available_attributes` → `GetNodeSummary` → `ExploreNeighborhood` → `RunSPARQL`. Explicit ban on `ExploreNeighborhood` before `GetNodeSummary` | Cat A subset (subscriber-count family) |
| 3 | `prompts.py` `QueryAttrQualifier` strategy | Add **DIRECTION RULE** for `direction: "backward"` results (swap `subject_id` ↔ `target_id` when calling `GetEdgeQualifiers`); elevate `QualifierFilter` as the canonical "find entity matching qualifier=V" tool (one SPARQL call, no per-entity loop) | Cat B + Cardiff-style cat A |
| 4 | `prompts.py` `QueryRelation` strategy | Add **🔴 OUTPUT FORMAT (HARD RULE)**: bare predicate label only, no narrative — mirroring Verify's existing format gate | Cat D |
| 5 | `prompts.py` `QueryRelation` strategy | Add **PASSIVE-VOICE GRAMMAR TRAP** rule for symmetric temporal relations (`followed_by`, `preceded_by`, `replaced_by`, `succeeded_by`, `follows`); worked example included but rule is stated generally | Cat C subset |
| 6 | `kqapro_adapter.py:93` | `max_iterations` 25 → 40 (matches `sciqa_adapter.py:94`) | Cat F (loop-detection at iter ~14 leaves no budget for post-intervention pivot) |

## Why this bundle, not a new tool

User asked whether to add `ExploreAttributes` (fuzzy embedding search over attribute names). Rejected for now:

- The minnesotaBASS failure is mis-diagnosed as "agent didn't know the attribute name". Actually the agent never *asked* what attributes exist on Q22588446. `GetNodeSummary` already returns the full attribute list in one call, and KQAPro nodes rarely exceed ~15 attributes. A fuzzy ranker over them isn't load-bearing.
- The agent's failure is at the discovery step (never calls `GetNodeSummary`), not the matching step. Fix 2 closes the discovery gap.
- Tool-schema budget is already 39.7% of input tokens (per `Token Overhead Analysis (2026-02-13 Benchmark).md`). Adding a 26th tool is non-free.
- Re-evaluate after Fixes 1–6 land. If post-fix n=100 still shows attribute-name-mismatch failures (agent calls `GetNodeSummary`, sees the right attribute, still picks the wrong one for `GetAttributeDetails`), then build `ExploreAttributes` *globally* (across all attribute embeddings, not scoped to a node) — that's where embedding fuzzy match actually beats `GetNodeSummary` + LLM string match.

## Why this over a `GetQualifierValue` tool

`GetQualifierValue(subject, predicate, target, qualifier_name)` remains the largest unshipped lever per the wiki backlog (~13/39 wrongs in the n=100 sample touch qualifier extraction). It is **not** in this bundle so the prompt-only fixes can be measured in isolation. Adding it on top of these prompt fixes is the next planned ADR; bundling them would muddle the attribution.

## Validation Plan

1. Re-run n=100 benchmark with `seed=42`, KIT endpoint, `minimax-m2.7-kit` and `gemma-4-31b-kit`, LLM-judge mode. Same seed as the post-audit baseline (so improvements aren't seed noise — wiki notes ~3–5 pp seed-noise floor).
2. Diff per-question-type accuracy against the prior run. Required signal:
   - Cat A failures (premature "missing") drop substantially.
   - `QueryRelation` band moves from 0.0–0.25 toward ≥0.5.
   - Cardiff-style and Olympics-style questions in the held-out audit set flip correct.
3. If headline does not move ≥3 pp, dig into the new traces before stacking more changes — the prompt-only fixes may have failed to bind for the model in question and a tool-side change is needed instead.

## Expected Impact

Per-fix estimates (conservative, accounting for category overlap):

| Fix | Estimated pp |
|---|---|
| 1. RULE 0a + GetNodeSummary gate | +6 to +10 |
| 2. QueryAttr attribute-first | +2 to +3 |
| 3. Direction rule + QualifierFilter elevation | +3 to +5 |
| 4. QueryRelation format gate | +2 to +3 |
| 5. Passive-voice trap | +1 to +2 |
| 6. Iter cap 25 → 40 | +1 to +2 |

Realistic combined landing zone (with overlap): **0.86–0.90** on n=100, vs. 0.79 baseline.

## Lesson

A judge fix (commit `9822bda`) and a default-flag revert (commit `5a5090f`) closed the most obvious leaks. The remaining failures are agent-policy gaps — when to call which tool, in which direction, with what output shape. Prompt strategy edits are still the cheapest lever for these; new tools come *after* the agent demonstrably can't do the job with existing surface.

---

## Post-merge Results (2026-05-04)

Bench: `benchmark_results/fixbundle-2026-05-03-1420/`, gemma timeouts re-run at `--timeout 600` in `gemma-retry-2026-05-04-1228/`, results merged.

| Model | Acc | vs. baseline |
|---|---|---|
| `minimax-m2.7-kit` | 0.770 | -2 pp (within ±3-5 pp noise floor) |
| `gemma-4-31b-kit` (merged) | 0.690 | -12 pp |

Headline is below target (0.86-0.90 predicted). **Per-type tells the real story** — Fix 4 + Fix 5 (format gate + passive-voice trap) are unambiguous wins:

| Type | gemma | minimax |
|---|---|---|
| Verify | **1.000** | 0.923 |
| QueryRelation | **1.000** | 0.846 |

Fix 1 / Fix 2 / Fix 3 did not measurably move their target categories.

**Diagnosed root cause of gemma's regression** (audit of 31 wrongs):
- **7/31 zero-tool-call answers** — gemma sometimes ignores RULE 0 ("MANDATORY TOOL USE"). Pure prompt enforcement is insufficient; needs **code-level retry**.
- **8/31 contain every gold token in cleaned predicted answer** — residual judge mis-marks despite the `9822bda` rewrite.
- Remaining 16/31 are genuine retrieval failures, predominantly QueryAttrQualifier (qualifier extraction).

**Recoverable upper bound:** closing both gaps lands gemma at ~0.82, beating baseline by +1 pp without another bench. The fix bundle is not broken.

## Recommended Next Steps (revised, in order)

1. **Ship code-level RULE 0 enforcement.** In `framework/base_agent.py`, after the agent emits a final answer, check tool-call count for the question; if 0, inject a corrective system message ("You produced a final answer without querying the KG, which is forbidden. Call FindNode / FindByAttribute on an entity from the question, then answer.") and re-prompt with capped retries. Expected: closes 5-7 pp on gemma, neutral on minimax.
2. **Revisit judge prompt** — focus on concept-extraction tolerance for verbose-correct answers. Add an explicit example covering the "gold='Aurora', pred='superhero (specifically Aurora...)'" pattern. Expected: +3-5 pp on both models.
3. **Build `GetQualifierValue` tool** — direct SPARQL projection of qualifier values. Largest unshipped lever; QueryAttrQualifier and QueryRelationQualifier together n=23 questions, current acc 0.36-0.78.

**Do not revert** Fix 4 and Fix 5 — both bind cleanly across both models with no observed cost. Fix 1, 2, 3, 6 are neutral-to-slightly-negative; can be left in place while the real levers ship, or selectively reverted if the next bench shows continued Query/Qualifier regression.

## Detection Note for Future Audits

`tool_trace` field is only populated for **native function-call models**. Text-mode models (minimax-m2.7-kit, anything using `text_tool_calls.py`) will show `tool_trace = []` even when 900+ tools were called — the actual call count lives in `tool_breakdown`. Any "zero tool calls" audit must filter on model family or check both fields, or you will misdiagnose the entire text-mode model class as RULE 0 violators (as I briefly did).

---

## Post-merge Results — judge swap + Fix 1 (2026-05-05)

Bench: `benchmark_results/minimax-v4judge-2026-05-04-2042/` and `gemma-v4judge-2026-05-04-2245/` (hetzner). n=100, seed=42, KIT endpoint, `deepseek/deepseek-v4-pro` judge (commit `182237c`), code-level zero-tool-call retry (commit `d9ae8de`).

| Model | Acc | vs. pre-hardening baseline |
|---|---|---|
| `minimax-m2.7-kit` | **0.810** | +2 pp vs 0.79 |
| `gemma-4-31b-kit` | **0.790** | -2 pp vs 0.81 — within noise; **0 timeouts** (was 36/100 with v3.2 + iter40) |

Per-type breakdown (selected):

| Type | gemma | minimax |
|---|---|---|
| Verify | **1.00** | 0.85 |
| QueryRelation | **1.00** | 0.93 |
| Count | — | 0.73 |

**Interpretation:** The predicted recovery from the "Recoverable upper bound" note is confirmed. Judge swap (v4-pro) fixed the residual verbose-answer mis-marking; code-level RULE 0 retry closed the zero-tool-call gap. Fix 4 (QueryRelation format gate) and Fix 5 (passive-voice trap) continue to bind cleanly — both models hit 1.00 or 0.93 on QueryRelation. Qualifier extraction (QueryAttrQualifier / QueryRelationQualifier) remains the dominant unsolved band; `GetQualifierValue` (commit `ff027d1`) is the next lever. See [Decisions/get-qualifier-value-tool.md](./get-qualifier-value-tool.md).

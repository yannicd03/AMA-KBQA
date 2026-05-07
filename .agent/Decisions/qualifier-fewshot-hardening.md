# ADR: Qualifier Fewshot Hardening — Encoding GetQualifierValue into Trace Examples

**Date:** 2026-05-05
**Status:** Accepted
**Code:**
- `db/datasets/kqapro/fewshot-examples/QueryRelationQualifier.json` — 3 existing examples rewritten; 1 new example added (Canberra/capital_of/Australia start_time=1927)
- `db/datasets/kqapro/fewshot-examples/QueryAttrQualifier.json` — 1 existing example rewritten; 2 new examples added (Russian-genitive criterion_used pattern; Manitoba demonym applies_to_part=masculine pattern)
- `db/datasets/kqapro/fewshot-examples/_tool_tips.json` — `GetQualifierValue` tip added with wh-word→qualifier-name map; `GetEdgeQualifiers` demoted to "discovery only" with explicit warning

## Related Docs

- [Decisions/get-qualifier-value-tool.md](./get-qualifier-value-tool.md) — the tool itself: signature, auto-detection logic, journal write, QTYPE_TOOL_MAP integration
- [Decisions/qualifier-and-format-prompt-hardening.md](./qualifier-and-format-prompt-hardening.md) — prior prompt-level qualifier work; post-merge audit that identified `GetQualifierValue` invocation rate as near-zero despite prompt-level recommendation
- [System/agent_system.md](../System/agent_system.md) — QTYPE_STRATEGIES and agent loop context

---

## Context

After `GetQualifierValue` shipped (commit `ff027d1`) and the n=100 benchmark was re-run (unstratified, seed=42, `minimax-m2.7-kit`, `benchmark_results/minimax-getqualval-2026-05-05-1100`), accuracy was **0.80/100** — up from 0.81 at the v4-pro baseline but not a clear improvement in the qualifier bands.

Trace audit of the 20 failures revealed the root cause: **`GetQualifierValue` was invoked on only 3 of 100 questions — all 3 correct** — despite:
1. `QTYPE_STRATEGIES["QueryAttrQualifier"]` and `["QueryRelationQualifier"]` explicitly naming it as preferred.
2. `QTYPE_TOOL_MAP` listing it for both types.

Of the 4 failures classified as "qualifier missed" (Q17 Russian-genitive criterion_used, Q62 Canberra-since-when, Q74 UWO-postal-code, Q87 Manitoba-demonym gender), the model used `GetEdgeQualifiers` / `GetQualifiersByPredicate` / `GetAttributeWithQualifiers` instead — the exact tools demonstrated in the existing fewshot traces.

This is a **"do as I show, not as I say"** failure. LLMs copy demonstrated trace patterns over instructed policies. The existing fewshot files showed the old tools; the model followed the examples, ignoring the text-level recommendation.

**Failure-mode breakdown for the n=100 run (for context):**

| Mode | Count | Share |
|---|---|---|
| A. Truncated tool-call as final answer | 6 | 30% |
| F. Wrong entity / wrong attribute | 5 | 25% |
| D. Qualifier missed (GetQualifierValue not used) | 4 | 20% |
| C. Count compositional error | 2 | 10% |
| B. Early give-up (≤3 calls) | 2 | 10% |
| E. Format mismatch (year vs full date) | 1 | 5% |

Mode D (4 failures, 20% of wrongs) is the target of this fewshot hardening. Mode A (6 failures, 30%) is addressed separately — see [Decisions/truncated-tool-call-retry.md](./truncated-tool-call-retry.md).

---

## Decision

Rewrite existing qualifier fewshot traces and add new examples that **structurally demonstrate** the wh-word → qualifier-name → `GetQualifierValue` call chain. Do not change the tool or the prompt strategy — only the examples the model imitates.

### QueryRelationQualifier.json changes

| Example | Old pattern | New pattern |
|---|---|---|
| Einstein/Nobel prize point_in_time | `GetEdgeQualifiers` → visual scan | `GetQualifierValue(subject, predicate, target, "point_in_time")` |
| Curie/role object_has_role | `GetEdgeQualifiers` → visual scan | `GetQualifierValue(subject, predicate, target, "object_has_role")` |
| Spielberg/Schindler ceremony | `GetEdgeQualifiers` → visual scan | `GetQualifierValue(subject, predicate, target, "ceremony")` |
| **Canberra/capital_of/Australia since when (NEW)** | — | `GetQualifierValue(Q3114, "capital_of", Q408, "start_time")` → 1927 |

The Canberra example directly mirrors Q62 from the audit (one of the 4 Mode D failures), so when "Since when has X been Y of Z?" patterns appear, the model has a near-identical exemplar.

### QueryAttrQualifier.json changes

| Example | Old pattern | New pattern |
|---|---|---|
| Pasco County language qualifier | `GetEdgeQualifiers` → visual scan | `GetQualifierValue(subject, "language", target_literal, "language")` |
| **Russian month genitive criterion_used (NEW)** | — | `GetQualifierValue(Q_month, "native_label", Russian_genitive_literal, "criterion_used")` |
| **Manitoba demonym applies_to_part=masculine (NEW)** | — | `GetQualifierValue(Q35617, "demonym", demonym_literal, "applies_to_part")` → masculine |

The Russian-genitive and Manitoba examples directly mirror Q17 and Q87 from the audit.

### _tool_tips.json changes

Added `GetQualifierValue` tip with a full wh-word → qualifier-name mapping:

| Wh-word | Qualifier name |
|---|---|
| When | point_in_time |
| Since when | start_time |
| Where | location |
| What role | object_has_role |
| What ceremony | ceremony |
| For what work | for_work |
| What language | language |
| What part / Which form | applies_to_part |

`GetEdgeQualifiers` tip downgraded to "discovery only — use when the qualifier name is not yet known; **do not use for extraction** (visual scan of the full qualifier dict is the most common qualifier-extraction failure mode)".

---

## Why fewshot, not prompt

The prompt strategy already named `GetQualifierValue` as preferred. That wasn't enough. The choice of trace example format is the dominant signal for LLMs on structured-reasoning tasks. This is a **representation fix**, not a **policy fix** — the policy was already correct.

## Why not extend GetQualifierValue's prompt entry with FORBIDDEN language

`FORBIDDEN` language on `GetEdgeQualifiers` would be too broad: `GetEdgeQualifiers` is still legitimately needed for qualifier *discovery* (when the question doesn't name the qualifier). A ban would hurt discovery-first patterns. The right lever is showing the model *when* each tool applies via examples, not forbidding the older tools globally.

## Empirical context

- **Run:** `benchmark_results/minimax-getqualval-2026-05-05-1100`, n=100, unstratified, seed=42, minimax-m2.7-kit
- **GetQualifierValue invocation rate:** 3/100 questions (vs. expected ~20, given qualifier question prevalence)
- **Mode D upper bound:** 4 pp improvement if all 4 qualifier-missed failures recover

Stratified A/B comparison (output dirs `minimax-kqapro-2026-05-05-fewshot` etc.) is in-flight at time of writing. Results will be appended to CHANGELOG once available.

---

## Invariants for Future Fewshot Edits

- When adding a new tool, **always** update the fewshot traces for every question type the tool targets — prompt-level recommendation alone is insufficient.
- Match at least one fewshot example to a specific observed failure case from an audit; the structural similarity is what triggers imitation.
- Demote the old tool's tip whenever a better tool for the same sub-task exists — side-by-side tips compete for the model's attention.

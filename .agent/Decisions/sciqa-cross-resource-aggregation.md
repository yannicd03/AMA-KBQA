---
# ADR: SciQA Cross-Resource Aggregation — FindFrequentValues + comparison_ids extension

**Date:** 2026-05-09
**Status:** Shipped
**Files:** `ama_kbqa/server/sciqa_server.py`, `ama_kbqa/agents/sciqa_agent/prompts.py`

## Related Docs
- [Agent System](../System/agent_system.md) — SciQA tool table, question-type strategy table, tool count
- [Decisions/sciqa-aggregation-and-coauthor-tools.md](sciqa-aggregation-and-coauthor-tools.md) — prior ADR that introduced `AggregateComparisonValues` (tool 19); this ADR extends it
- [Decisions/sciqa-node-type-filter.md](sciqa-node-type-filter.md) — `node_type_filter` for `FindResource`; surface the right Comparison IDs before aggregating

---

## Context

A minimax SciQA paper run (seed=42, stratified) produced 53 failures. 12 of these were **global-scope superlative/aggregation questions** — questions of the form "most popular X", "largest Y across the papers", "frequency of Z in the studies". The shared failure pattern was:

1. Agent found a specific Comparison via `FindResource`.
2. Called `AggregateComparisonValues(comparison_id=<single Comparison>)`.
3. Reported that Comparison's data as the answer.

But the gold answer required aggregation across **all** Contributions linked to any Comparison, or across several named Comparisons, because the data is distributed across the ORKG graph.

Diagnosis from failure traces:
- `GetRelationTargets` had a 70:9 fail-vs-correct call ratio — used as a hammer instead of aggregation tools.
- `RunORKGSPARQL` appeared 2.3× more often in failures than in correct traces; manual SPARQL for aggregation is unreliable without tool-enforced HAS_VALUE indirection.
- Average tool-call count per failure: 21–37 vs 14–20 for correct answers, indicating thrashing without a "synthesize what you have" trigger.

---

## Decision

### 1. New tool: `FindFrequentValues` (tool 20 in sciqa_server.py)

`FindFrequentValues` is a cross-resource aggregation tool that does not require a Comparison anchor. It answers questions where the relevant data is distributed across many Comparisons or an entire research field.

**Signature:**
```python
FindFrequentValues(
    value_predicate: str,                        # Predicate ID to aggregate (e.g. "P15585")
    agg: str = "mode_top",                       # mode_top|count|count_distinct|sum|avg|min|max|all_values
    research_field_id: str = "",                 # Scope to P30 → field (e.g. "R132")
    comparison_ids: str = "",                    # Scope to VALUES-clause union of listed Comparisons
    group_by_predicate: str = "",                # Optional grouping predicate
    filter_predicate: str = "",                  # Optional pre-filter predicate
    filter_value: str = "",
    filter_match: Literal["exact","contains","regex"] = "contains",
    top_n: int = 5,
    limit_subjects: int = 5000,                  # Hard cap on contributions scanned
)
```

**Scope tiers (checked in order):**

| Priority | Trigger | SPARQL scope |
|----------|---------|-------------|
| 1 | `research_field_id` non-empty | `?paper orkgp:P30 orkgr:<field> . ?paper orkgp:P31 ?contrib` |
| 2 | `comparison_ids` non-empty | `VALUES ?cmp { R1 R2 ... } . ?cmp orkgp:compareContribution ?contrib` |
| 3 | (default) | `?cmp orkgp:compareContribution ?contrib` — every Contribution of any Comparison |

**Aggregation modes** are identical to `AggregateComparisonValues`: `mode_top` (frequency table, most useful for "most popular X"), `count`, `count_distinct`, `sum`, `avg`, `min`, `max`, `all_values`. HAS_VALUE indirection and rdfs:label fallbacks are applied automatically.

**Hard cap:** `limit_subjects=5000`. For very large research fields, the agent should narrow scope via `comparison_ids` rather than raising this cap.

### 2. Extension to `AggregateComparisonValues`: `comparison_ids` parameter

`AggregateComparisonValues` gained a new `comparison_ids: str = ""` parameter (comma-separated Comparison IDs). When non-empty, it **overrides** `comparison_id` and unions rows across all listed Comparisons via a `VALUES` clause:

```sparql
VALUES ?cmp { orkgr:R1 orkgr:R2 orkgr:R3 }
?cmp orkgp:compareContribution ?contrib .
```

**Response shape:** gains a `comparison_ids` key (instead of `comparison_id`) when in multi-source mode. Journal bucketing keys on the joined CSV string. Backwards-compatible: every existing single-comparison call with `comparison_id="R<id>"` and `comparison_ids=""` continues to work unchanged.

### 3. Prompt updates: Superlative and Aggregation strategies

Both `QTYPE_STRATEGIES["Superlative"]` and `QTYPE_STRATEGIES["Aggregation"]` were rewritten to lead with a **SCOPE decision**:

| Scope | Signal phrases | Tool route |
|-------|---------------|-----------|
| Single-comparison | "the studies", "the comparison", "the analysis" | `AggregateComparisonValues(comparison_id=...)` |
| Multi-comparison | "several related comparisons", data split across 2–5 known Comparisons | `AggregateComparisonValues(comparison_ids="R1,R2,R3")` |
| Cross-graph / global | "across the papers", "most popular X overall", "in the studies" (global) | `FindFrequentValues(value_predicate, agg=...)` |

A **GLOBAL-SCOPE WARNING** block was added to `Superlative` strategy explicitly naming this as "the highest-leverage failure mode in v1" and requiring the agent to use `FindFrequentValues` or multi-`comparison_ids` rather than locking onto a single Comparison.

### 4. NO_PROGRESS_TEMPLATE: decision tree with 25-call hard stop

The no-progress intervention was rewritten as a 5-branch decision tree. The critical branch:

> **5. If you have already used 25+ tool calls without journal change: STOP exploring. Synthesize the best answer you can from what is already in the journal. Further tool calls are likely just thrashing.**

Branch 1 of the tree specifically targets the global-scope failure class:

> **1. If the question is global-scope ("most popular X", "largest Y", "frequency of Z across the studies/papers") and you have been narrowing into one Comparison: → STOP. Call FindFrequentValues(...).**

### 5. New fewshot examples (3 added)

| Example | Pattern demonstrated |
|---------|---------------------|
| Largest sample size cross-graph max | `FindFrequentValues(value_predicate, agg="max")` — no Comparison anchor |
| Most frequent drug cross-graph mode_top | `FindFrequentValues(value_predicate, agg="mode_top", top_n=5)` |
| Energy sector cross-comparison frequency | `AggregateComparisonValues(comparison_ids="R1,R2,R3", agg="mode_top")` |

---

## Why Not Just Use RunORKGSPARQL?

The failure class was **model-independent**: minimax, gemma, and earlier qwen all reproduced it. That rules out a model-capability gap. The structural problem is that correctly composing a cross-resource aggregation SPARQL requires knowing:

1. That the answer spans multiple Comparisons (not obvious from the question surface).
2. The HAS_VALUE + rdfs:label indirection pattern (depth-3 or depth-4).
3. The `compareContribution` pivot predicate.

`FindFrequentValues` encapsulates all three. The model only decides `value_predicate` and `agg` — the narrowest possible interface for the decision it needs to make.

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Directly closes the 12-failure global-scope class | Increases tool surface from 20 to 21 |
| `comparison_ids` on `AggregateComparisonValues` is backwards-compatible | `FindFrequentValues` has 10 parameters — the most complex SciQA tool |
| `limit_subjects=5000` keeps SPARQL bounded | Very large research fields may need chunking via `comparison_ids` |
| SCOPE decision prompt prevents silent single-Comparison lock-in | Model must learn to recognize "global scope" from question wording |
| 25-call hard stop prevents thrashing | May cut off genuinely difficult questions at call 26+ |

---

## Deployment Note

Shipped locally as of 2026-05-09. Hetzner deployment needed before benchmark impact can be measured.

---
# ADR: Multi-Label Classifier-Output Tolerance in SciQA _classify_question

**Date:** 2026-05-09
**Status:** Shipped
**Files:** `ama_kbqa/agents/sciqa_agent/agent.py`

## Related Docs
- [Decisions/classifier-think-prefix-fix.md](classifier-think-prefix-fix.md) — companion fix: JSON extraction from `<think>` prefixes; bumped classifier `max_tokens` 300→1500
- [Agent System](../System/agent_system.md) — SciQA question-type classification table; `_classify_question` override

---

## Context

`SciQAAgent._classify_question()` calls the LLM with `response_format=json_object` and expects a single `question_type` string. It then does `FEWSHOT_EXAMPLES.get(qtype, "")` to load the relevant few-shot trace.

The SciQA gold dataset (`db/sciqa_questionnaire.json`) contains questions annotated with **compound question types** in the `Q Content` column, e.g.:

- `"Factoid\nSuperlative"` — a factoid question that requires a superlative operation
- `"Factoid, Count"` — a factoid-style count question
- `"Boolean/Aggregation"` — boolean question involving aggregation

When the LLM classifier reproduces these compound labels verbatim (common when fewshots or prompts reference the gold annotation vocabulary), the exact string `"Factoid\nSuperlative"` does not match any key in `FEWSHOT_EXAMPLES`. The result is a **silent empty fewshot** — the agent uses the correct strategy name in logs but gets zero few-shot examples, which specifically hurts Aggregation and Superlative questions that rely on `FindFrequentValues` / `AggregateComparisonValues` traces.

This was identified as failure root cause 5 in the minimax seed=42 paper run (contributing to the Aggregation/Superlative failure class alongside the cross-resource scope problem captured in `sciqa-cross-resource-aggregation.md`).

---

## Decision

### Multi-label split and merge in `_classify_question`

When `FEWSHOT_EXAMPLES.get(qtype, "")` returns empty (exact key miss), the code now:

1. Splits `qtype` on `r"[\n,/+|;]+"` — all common compound separators.
2. Case-normalizes each candidate: `cand[:1].upper() + cand[1:]`.
3. Looks up each normalized candidate in `FEWSHOT_EXAMPLES`.
4. Concatenates all matches (deduplicated, order preserved from the split).
5. Sets `chosen_label` to the first matching candidate label (used for logging and strategy selection).

```python
candidates = [
    c.strip() for c in re.split(r"[\n,/+|;]+", qtype) if c.strip()
]
seen = set()
parts: List[str] = []
for cand in candidates:
    cand_norm = cand[:1].upper() + cand[1:] if cand else cand
    if cand_norm in FEWSHOT_EXAMPLES and cand_norm not in seen:
        seen.add(cand_norm)
        parts.append(FEWSHOT_EXAMPLES[cand_norm])
        if chosen_label == qtype:
            chosen_label = cand_norm
fewshot = "\n".join(parts)
```

**Result for `"Factoid\nSuperlative"`:** both `FEWSHOT_EXAMPLES["Factoid"]` and `FEWSHOT_EXAMPLES["Superlative"]` are concatenated (~4 KB total). The agent receives Superlative fewshots with `FindFrequentValues` examples that were previously invisible.

### What is NOT changed

- The SPARQL strategy loaded from `QTYPE_STRATEGIES` is keyed by `chosen_label` (first match), not the full compound string. This is intentional: the Superlative strategy already covers both factoid-superlative and pure superlative.
- The base `KQAProAgent._classify_question()` is not affected; it has its own separate implementation. This fix is SciQA-only.
- `FEWSHOT_EXAMPLES` keys are unchanged — no new keys added.

---

## Why Not Normalize the Classifier Prompt?

An alternative would be to force the classifier to output only a single canonical label (e.g., adding "output ONLY one of: Factoid, Count, ..."). This was rejected because:

1. The multi-label outputs reflect genuine question duality — a "Factoid\nSuperlative" question genuinely needs both Factoid and Superlative guidance. Forcing a single label would silently discard the Superlative strategy.
2. The compound-label vocabulary leaks in from the gold dataset via context. Suppressing it in the classifier would require sanitizing the gold data or the classification prompt in ways that could break other things.
3. The merge-mode is additive and safe: it can only provide more guidance, never less.

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Fixes silent empty-fewshot for compound classifier outputs | Slightly more fewshot tokens injected per question for compound types |
| Additive — cannot break existing single-label paths | If classifier emits a long compound like "Factoid\nComparison\nAggregation\nSuperlative", all four fewshot blocks concatenate (~8 KB) |
| Does not require changing the classifier prompt | Deduplication prevents exact-duplicate blocks but not near-duplicate |

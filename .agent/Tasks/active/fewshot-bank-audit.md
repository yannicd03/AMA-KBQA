# Few-Shot Bank Audit Against Gold + Decouple Retrieval From the Single Classifier Label

**Status:** 📋 Planned
**Priority:** Medium (larger design work — prompt architecture, not a bug fix)
**Identified:** 2026-07-25 (reasoning-error and capability-gap analysis, recommendation #9)

## Related Docs
- [../../Decisions/reasoning-error-analysis-2026-07-25.md](../../Decisions/reasoning-error-analysis-2026-07-25.md) — full analysis; §4.1 "The classifier is a single point of failure" and §4.2 "The prompt library contradicts the tool contracts" are the source findings
- [../../Decisions/multi-label-fewshot-tolerance.md](../../Decisions/multi-label-fewshot-tolerance.md) — prior, narrower fix in the same area: split-and-merge fewshot lookup for compound classifier outputs (`"Factoid\nSuperlative"`). This task generalizes that fix from a parsing bug to a retrieval-architecture problem
- [../../Decisions/qualifier-fewshot-hardening.md](../../Decisions/qualifier-fewshot-hardening.md) — an example of the exact failure pattern this task targets: a fewshot rewrite that worked, but only for traces that got the right classifier label
- [../../Decisions/fewshot-generator-enrichment.md](../../Decisions/fewshot-generator-enrichment.md) — the generator pipeline that produces the fewshot bank this audit needs to check against gold
- [../../System/kqapro_agent.md](../../System/kqapro_agent.md) — KQAPro 10-type classification
- [../../System/sciqa_agent.md](../../System/sciqa_agent.md) — SciQA 8-type classification, multi-label tolerance mechanism

## Problem

A single question-type classifier label gates **both** the strategy text shown to the agent **and**
which few-shot examples get retrieved. Two independent failure patterns compound this:

**(a) Misclassification silently strips guardrails (§4.1).**
- idx 68 tagged `QueryAttr` instead of `QueryAttrQualifier`; idx 87 tagged `QueryName`. Both
  misclassifications stripped the "the answer is a qualifier value, do not collapse to the entity"
  guardrail that appears verbatim in *correctly*-classified traces — the guardrail text exists and
  works, it's just invisible to misrouted questions.
- SciQA "most commonly modeled energy sector" tagged `Superlative`. The few-shot containing the
  **exact resource, exact predicate path, and exact gold answer** ("Heat sector, 8") lives in the
  `Count` bank and was never shown to the agent because it was never looking there.

**(b) The prompt library contains contradicting few-shots that were never retracted (§4.2).**
- `AggregateComparisonValues`'s docstring (`sciqa_server.py:3756-3760`) states that
  `group_by_path="^P31,P29"` is "the correct time axis for 'per year' / 'in N-year intervals'
  questions." `prompts.py:1093` ships a *different* worked example for that exact question class
  teaching `group_by_path="P37581,P43138,P43139"` (the scenario's internal goal year). Every sampled
  run followed the (wrong) few-shot instead of the (correct) docstring. The capability added in commit
  `6e30da5` works; the competing wrong example living alongside it in the prompt library was never
  retracted.
- Separately, the energy-sectors few-shot names `R153801` **by ID as a known trap** ("if you only
  `AggregateComparisonValues` over R153801 you get the WRONG answer"), and the agent used R153801
  anyway, in the same context window as the warning, in 2 of 3 runs — suggesting the trap-warning
  format itself isn't landing, independent of retrieval.

**Pattern:** fixes are being written as per-question worked examples, which are then withheld by
classifier routing, contradicted by stale siblings, or ignored outright even when shown. This is a
prompt-*architecture* problem, not a one-off content bug — hardening individual few-shots (as done
repeatedly in [qualifier-fewshot-hardening.md](../../Decisions/qualifier-fewshot-hardening.md) and
similar past ADRs) treats the symptom each time without addressing why the fixes don't generalize.

## Goal

1. **Audit the full few-shot bank against the gold corpus** (2,565-query SciQA Zenodo release +
   KQAPro `val.json`, the same gold sources used in the parent analysis) to find every case of (a)
   contradicting examples for the same tool/parameter, and (b) known-correct examples that live under
   a classifier label other than the one the matching questions actually get tagged with.
2. **Decouple few-shot retrieval from the single classifier label.** Retrieval should not be a strict
   lookup keyed on one label; it should tolerate — at minimum — the multi-label case already handled
   by [multi-label-fewshot-tolerance.md](../../Decisions/multi-label-fewshot-tolerance.md), and ideally
   extend toward similarity-based retrieval (e.g. embed the question, retrieve the top-K nearest
   few-shots across *all* labels, not just the classified one) so a single misclassification doesn't
   fully hide the one example that would have fixed the answer.

## Design

Not yet fully speced — this is exactly the "larger design work" the parent analysis flags as needing
a dedicated session. Starting points:

1. **Static audit pass first, no code changes.** Grep/diff the current fewshot JSON files against
   each other for contradicting parameter guidance on the same tool (the `group_by_path` case is the
   known instance; there are likely others). Cheap, no benchmark risk, produces a punch list.
2. **Cross-check classifier-label assignment against gold qtype** for every fewshot example — flag
   any example whose content clearly targets a question shape that could plausibly be classified
   under a *different* label than the one it's filed under.
3. **Retrieval architecture change** (the harder part): evaluate embedding-based few-shot retrieval
   (independent of, or supplementing, the classifier label) vs. simply loosening the label-match to
   "top-N labels by classifier confidence" if the classifier already exposes a confidence signal.
   Needs its own design sub-doc once the audit pass identifies how large the mislabeled-example
   problem actually is — don't over-build the retrieval change before knowing the audit's scope.

## Risks

- **Audit-then-retrieval-change is two separate risk profiles.** The audit itself (step 1-2) is safe
  and read-only. The retrieval architecture change (step 3) touches the classification/fewshot path
  for *every* question, not just the failure cases identified — a regression here could be worse than
  the problem it's fixing. Needs a full benchmark A/B, not a targeted re-run.
- **Contradicting examples might both be "correct" in different contexts.** Before deleting or
  demoting a "wrong" few-shot, confirm it isn't actually correct for a different sub-case that just
  looks similar on the surface (the `group_by_path` case is confirmed genuinely contradictory per the
  docstring cross-check, but this needs to be verified case-by-case, not assumed for every finding the
  audit turns up).
- **This compounds with recommendation #1** ([locate-term-tool.md](locate-term-tool.md)): a new tool
  is only as good as its fewshot/strategy-text coverage, so this audit should also cover whatever
  fewshot examples get written for `LocateTerm` and the SciQA grouped-aggregation tools
  ([sciqa-grouped-aggregation-tools.md](sciqa-grouped-aggregation-tools.md)) once those ship, not just
  the existing bank.

## Validation plan

- Static audit output: a list of contradicting-example pairs and mislabeled-example candidates, each
  with the specific gold question(s) it affects.
- Before/after re-run of the specific idx cases named in §4.1/§4.2 of the parent analysis (idx 68,
  87, the energy-sectors SciQA question) once fixes land.
- Full n=100 KQAPro + SciQA benchmark A/B for any retrieval-architecture change (step 3), since it
  touches every question's fewshot exposure, not just the identified failures.

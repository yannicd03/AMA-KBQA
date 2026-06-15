# SciQA Raw SPARQL Tool: Keep Decision (A/B Experiment 2026-06-15)

## Related Docs
- [../System/agent_system.md](../System/agent_system.md)
- [root-cause-tool-generalization-2026-05-14.md](root-cause-tool-generalization-2026-05-14.md)
- [sciqa-cross-resource-aggregation.md](sciqa-cross-resource-aggregation.md)
- [sciqa-aggregation-and-coauthor-tools.md](sciqa-aggregation-and-coauthor-tools.md)

## Context

Three SciQA 100-question runs (seed 42 baseline, seed 42 with agent improvements, seed 43
validation) accumulated 509 `RunORKGSPARQL` calls. Trace analysis showed the tool is heavily
used but largely unproductive:

- ~70% of questions invoke `RunORKGSPARQL` at least once.
- Raw SPARQL is ~20% of all tool calls per run.
- Only ~32% of those calls return rows; ~55% return empty (0 results); ~10% are malformed
  (Virtuoso `QueryBadFormed`); ~3% are blocked by the loop detector.

**Measurement note:** `summary.json` `tool_breakdown` reports near-zero failures because the
tool wraps errors into a normal return value (the error text becomes the return string). Success
rates must therefore be measured from raw traces, not the summary file.

Load-bearing analysis of the 77 correct questions from the seed-42 baseline (raw SPARQL ON):

| Bucket | Count | Interpretation |
|--------|-------|----------------|
| Used no raw SPARQL at all | 29 | Not load-bearing |
| Used raw SPARQL, got only empty/error | 26 | Wasted; correct answer came from dedicated tools |
| Got rows; dedicated tool also succeeded | 16 | Recoverable without raw SPARQL |
| Got rows; NO dedicated tool also succeeded | 6 | At-risk if raw SPARQL removed |

Predicted removal cost before the experiment: ~3-6 questions, with ~3 genuinely unrecoverable.

## The A/B Experiment

**Gate implementation (commit 5f7fc76):** A new off-by-default env flag
`AMA_SCIQA_DISABLE_RAW_SPARQL` drops `RunORKGSPARQL` from the advertised OpenAI tool set when
truthy. Mechanism: `BaseKBQAAgent._get_denied_tool_names()` is a new hook (returns empty set by
default) applied to `openai_tools` AFTER the qtype filter, so it works even when all tools are
permitted. `SciQAAgent` overrides it to return `{"RunORKGSPARQL"}` when the env flag is set.
15 new tests were added; suite reached 375.

**Runs compared:**

| Run | Seed | Raw SPARQL | Score | Tokens |
|-----|------|-----------|-------|--------|
| Baseline | 42 | ON | 77/100 (summary) | 11.87M |
| A/B gate ON | 42 | OFF | 73/100 (summary) | 12.10M |

Question-level judge diff (gate ON vs baseline): 7 broke, 2 gained, net -5.
Rounding summary counts: net -4.

Token budget did NOT decrease with raw SPARQL removed. The agent spent additional budget on
dedicated-tool exploration attempts instead. Runtime was 4674s (ON) vs 3719s (OFF); this gap
reflects KIT API latency variance across runs and is not meaningful.

**Outcome per at-risk question:**

| Question | Prediction | Actual with gate ON |
|----------|-----------|---------------------|
| "Date of the first paper related to X-Rays" (global recency, no comparison anchor, no dedicated tool) | Unrecoverable | BROKE |
| "2050 time frame" (comparison-scoped numeric filter, recoverable via `QueryComparisonRows` nested path) | Recoverable | BROKE (agent did not choose the path) |
| "Definition of Raman" | Recoverable via `GetResourceSummary` | Survived |
| "Studies published after 2019" | Recoverable | Survived |
| Boolean ASK: "children examined" | Recoverable by inference | Survived |
| Boolean ASK: "integrity constraints in OWLMAP" | Recoverable by inference | Survived |

Three additional breakages appeared outside the at-risk bucket (Chloride-anion count; "6th Open
Challenge" research problem; most-commonly-modeled sector). Two of the seven total breakages
(EXPO ontology full name; least-frequently-used metric) are considered noise-band, as these
questions flip across seeds without code changes.

## Decision

**Keep `RunORKGSPARQL`. Do not remove it.**

Removing the tool costs ~4-5 accuracy points, saves no tokens, and yields no offsetting
accuracy gain (only 2 gained vs 7 broke). The hypothesis that forcing dedicated-tool use
improves accuracy is not supported by the data.

Instead, the forward path is:

1. **Keep the gate as the permanent A/B harness and last-resort lever.** `AMA_SCIQA_DISABLE_RAW_SPARQL`
   remains available for future experiments or as a budget override.

2. **Backfill the two genuine tool gaps exposed by the experiment:**
   - A global recency/extreme tool for "first/latest paper about TOPIC" with no comparison
     anchor and no year-range filter (the "X-Rays" question has no existing tool path).
   - A comparison-scoped count-with-numeric-filter that cleanly handles inverse `^P31,P29`
     paper-year hops (the "2050 time frame" class; `QueryComparisonRows` can express this but
     agent adoption is unreliable without a more explicit tool surface).

3. **Boolean ASK questions are lower priority** for a dedicated tool. The gate experiment
   confirmed both boolean cases are recoverable via inference from retrieved summaries; the
   main failure mode is scope confusion, not a missing ASK primitive.

## Consequences

- `RunORKGSPARQL` stays in the advertised SciQA tool set unconditionally.
- `AMA_SCIQA_DISABLE_RAW_SPARQL=1` can suppress it for experiments or ablations.
- The ~65% unproductive-call rate is accepted as the cost of covering the ~6 genuinely
  raw-SPARQL-dependent questions per 100-question run.
- Future tool additions (global recency/extreme, comparison-scoped year-filter) should be
  validated against the gate-OFF baseline (77/100) to verify they cover the remaining
  raw-SPARQL-only question class before re-running the removal experiment.

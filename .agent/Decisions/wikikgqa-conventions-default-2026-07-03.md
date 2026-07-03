# ADR: WikiKGQA Answer Conventions Default Reverted to Minimal (R1-R6)

**Date:** 2026-07-03
**Commit:** `23e5538` "WikiKGQA: default conventions to minimal (R1-R6) after held-out A/B"
**Status:** Settled

## Related Docs
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference for `WikidataAgent`'s `conventions` parameter and `ANSWER_CONVENTIONS.md`
- [Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md](./wikikgqa-synthesis-context-ab-2026-07-02.md) — the prior day's wash (full vs minimal synthesis context); same default-to-cheaper reasoning applied here
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md) — same-day ADR whose "In-flight validation run" section tracked this A/B before it completed; updated to point here now that it has final numbers

---

## Background

`WikidataAgent(conventions="full"|"minimal")` was a held-out A/B toggle:
`"full"` appends `EXTENDED_CONVENTIONS` (R7-R10 — normalized-quantity
statement path, "currently/still" exclusion filters, instance-of
completeness, etc.) to `SYSTEM_PROMPT`; `"minimal"` uses only R1-R6. R7-R10
were mined from the seed-42 tuning set and had been the default (`"full"`)
pending a held-out check of whether they generalize to unseen questions.

The [2026-07-03 commit-time-recovery ADR](./wikikgqa-commit-time-recovery-2026-07-03.md)
recorded this A/B as in-flight (both `run-rand50-seed99-conv-{full,minimal}`
directories incomplete at that capture). It has since finished.

## Evidence

Held-out seed-99, `rand-50`, model `kit.qwen3.5-397b-A17b`, challenge
endpoint, both arms run through the same 2026-07-03 commit's recovery/voting
code path:

| Conventions | Macro F1 | Precision | Recall | exec_ok |
|---|---|---|---|---|
| Minimal (R1-R6) | **0.7311** | 0.7771 | 0.7432 | 50/50 |
| Full (R1-R10) | 0.6833 | 0.7476 | 0.7159 | 50/50 |

Both arms executed a query for all 50 questions (zero empty submissions) —
the gap is a scoring-quality difference, not a resilience difference.

11/50 questions score differently between arms (7 favor minimal, 4 favor
full; a sign test on this split is not significant at n=11). The four
largest full-arm losses (q44, q236, q275, q435) were traced to **divergent
agent trajectories** between the two runs (different exploration paths led
to different committed queries), not to the R7-R10 rules themselves actively
misleading the model on those questions. Net: R7-R10 show no held-out
benefit on this sample, and the delta is noise-dominated rather than a clear
regression attributable to the extended rules.

## Decision

Flip the default to **minimal (R1-R6)**. Full conventions become opt-in via
`WikidataAgent(conventions="full")` / `benchmark.py --conventions full`.

**Rationale:** identical reasoning to the synthesis-context wash the day
before — when an A/B is noise-dominated with no measurable benefit, default
to the cheaper, lower-overfitting-risk variant (minimal conventions is a
shorter prompt, and R7-R10 were mined from the tuning set, so keeping them
default risks encoding tuning-set-specific patterns as if they were general
rules). Full stays available for future targeted experiments.

**Rejected:** Keeping `"full"` as the default on the theory that more
explicit modeling guidance is "probably not worse." This A/B existed
specifically to test that assumption on held-out data and found no benefit
to justify it — same posture as the synthesis-context decision.

## Implementation

- `ama_kbqa/agents/wikidata_agent/agent.py` — `WikidataAgent.__init__`
  `conventions: str = "minimal"` (was `"full"`).
- `ama_kbqa/wikikgqa/benchmark.py` — `--conventions` CLI flag default
  `"minimal"` (was `"full"`); help text updated with the A/B numbers.
- `ama_kbqa/wikikgqa/generator.py` — `AgentSparqlGenerator.__init__`
  `conventions: str = "minimal"` (was `"full"`), matching the agent default.

## Secondary finding: held-out F1 calibration

This run also recalibrates expectations for the full pipeline: held-out
seed-99 full-system F1 is **~0.73**, well below the seed-42 tuning-set 0.807
cited in the synthesis-context ADR. Leaderboard/submission expectations
should anchor to the held-out number (~0.73), not the tuning-set number —
the tuning-set F1 reflects prompts and settings partially selected against
that same seed-42 sample, so it is optimistic relative to genuinely unseen
questions.

## Consequence

Any run invoked without an explicit `--conventions` flag from this commit
onward uses minimal (R1-R6). `run_manifest.json` records the resolved
`conventions` value per run going forward (see the commit-time-recovery
ADR); runs predating `23e5538` (2026-07-03) that relied on the old default
were running `"full"` and should be re-checked via their manifest or
console log before comparison.

## Open Questions

| Question | Notes |
|---|---|
| Do any individual R7-R10 rules help on a narrower question subclass even though the aggregate is a wash? | Not decomposed here; would require a per-rule ablation, not just full-vs-minimal |
| Why did q44/q236/q275/q435 diverge in agent trajectory between arms rather than just in final-answer correctness? | Consistent with the framework's general sensitivity to small context/prompt-length changes early in exploration (same class of effect as trajectory divergence noted elsewhere); not investigated further here |

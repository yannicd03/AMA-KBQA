# `SelectExtreme` Relation Filter (Avoid Pasting Hundreds of Entity IDs)

**Status:** 📋 Planned
**Priority:** Medium (ergonomics fix — capability exists, composing the call correctly does not)
**Identified:** 2026-07-25 (reasoning-error and capability-gap analysis, recommendation #6)

## Related Docs
- [../../Decisions/reasoning-error-analysis-2026-07-25.md](../../Decisions/reasoning-error-analysis-2026-07-25.md) — full analysis; §2 "Secondary reasoning error: constraint dropped under pressure" (idx 98) is the source finding
- [self-diagnosing-empty-results.md](self-diagnosing-empty-results.md) — companion task attacking the same idx 55/98 failure class from the empty-result-diagnosis side; land together where practical
- [../../Decisions/kqapro-tool-surface-expansion.md](../../Decisions/kqapro-tool-surface-expansion.md) — original ADR that added `SelectExtreme`; this task extends its filter surface rather than replacing it
- [../../System/kqapro_agent.md](../../System/kqapro_agent.md) — KQAPro tool catalog `SelectExtreme` entry

## Problem

idx 98: "cheapest/shortest Belgium-release feature film" class of question. `GetRelationDetails(Q31,
"film release region")` returned all **311** Belgium-release films into context, untruncated. A loop
guard then fired on a subsequent raw-SPARQL attempt, and the agent fell back to
`SelectExtreme(attribute_name="duration", concept="feature film", mode="min")` — **dropping the
Belgium constraint entirely** — and answered *Gulliver's Travels* (which has **zero**
`film_release_region` triples). Gold is *Alvin and the Chipmunks: Chipwrecked*.

`SelectExtreme` already accepts `entity_ids` and `concept` together, so the correct call — extreme
duration *within* the Belgium-release candidate set — was technically expressible. The failure is
that composing it correctly required the model to emit all 311 entity IDs through its context window
as the `entity_ids` argument. That's a tool-ergonomics defect: the capability exists, but using it
correctly is impractically expensive for any candidate set of meaningful size, so the model doesn't
even attempt it and drops the constraint instead.

## Goal

Let `SelectExtreme` take a **relation-based filter directly** (subject/predicate/target, the same
shape as `GetRelationDetails`'s query) instead of requiring the caller to first materialize the full
candidate list as `entity_ids`. The model should be able to express "extreme duration among films
that have relation film_release_region → Belgium" as one call with a relation constraint, not two
calls plus a 311-ID paste.

## Design

Not yet fully speced. Starting point:

1. **Add a relation-filter parameter to `SelectExtreme`**, structurally similar to
   `CountEntities`'s existing relation-constraint handling (see
   [count-entities-not-conditions.md](../../Decisions/count-entities-not-conditions.md) for the
   `_attr_condition_sparql`-style helper pattern already used for attribute constraints) — e.g.
   `relation_name`, `relation_target_id(s)`, `relation_direction`, mirroring the shape
   `CountUnion` already added for branch-local relation filters per
   [root-cause-tool-generalization-2026-05-14.md](../../Decisions/root-cause-tool-generalization-2026-05-14.md).
2. **Compose the extreme-value query and the relation filter in one SPARQL query server-side**,
   rather than requiring the client to first fetch candidates and then re-supply them — this is the
   core ergonomics win, since it removes the need for the model to ever see (or repeat) the
   intermediate 311-row candidate list.
3. **Keep the existing `entity_ids` + `concept` path** for cases where the candidate set genuinely
   comes from a prior tool call the model already has in hand and is small enough to compose (per the
   original ADR's intent) — this is an additive parameter, not a replacement of the existing contract.

## Risks

- **Overlap with recommendation #2** (self-diagnosing empty results): idx 98's failure is really two
  compounding problems — an oversized, uncomposable result (`GetRelationDetails` returning 311
  untruncated rows) and a downstream tool without a relation-filter escape hatch. Fixing only this
  half (the escape hatch) without also addressing why `GetRelationDetails` dumps 311 rows into context
  in the first place leaves the root oversized-result problem for other tool combinations. Consider
  whether `GetRelationDetails` also needs a truncation/pagination guard as part of this task or a
  sibling one.
- **New parameter surface adds to an already-large tool.** `SelectExtreme` already takes
  `entity_ids`, `concept`, `attribute_name`, `mode`. Adding relation-filter parameters risks the same
  "parameterized-to-death mega-tool" concern flagged in
  [sciqa-grouped-aggregation-tools.md](sciqa-grouped-aggregation-tools.md) — keep the added surface
  minimal and mirror an existing pattern (`CountEntities`/`CountUnion`'s relation-constraint shape)
  rather than inventing a new one.
- **Adoption risk.** Per §4 of the parent analysis, a new parameter on an existing tool is only as
  good as the strategy-text/fewshot coverage teaching the model to reach for it instead of the
  entity_ids-paste-or-drop-constraint pattern seen in idx 98. Needs explicit fewshot coverage,
  coordinated with [fewshot-bank-audit.md](fewshot-bank-audit.md).

## Validation plan

- Smoke-test directly against idx 98's gold answer (*Alvin and the Chipmunks: Chipwrecked*) with the
  new relation-filter parameter before any benchmark run.
- Confirm the existing `entity_ids` + `concept` path is unaffected (regression check against
  `SelectExtreme`'s existing test coverage).
- Full n=100 KQAPro benchmark A/B, since this changes what a widely-used tool (`SelectExtreme`) can
  express and therefore can shift agent behavior beyond just the target failure class.

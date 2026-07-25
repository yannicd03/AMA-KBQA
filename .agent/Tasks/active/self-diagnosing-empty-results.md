# Self-Diagnosing Empty Results

**Status:** 📋 Planned
**Priority:** High (second of the deferred set — the revision-failure enabler behind most adoption gaps)
**Identified:** 2026-07-25 (reasoning-error and capability-gap analysis, recommendation #2)

## Related Docs
- [../../Decisions/reasoning-error-analysis-2026-07-25.md](../../Decisions/reasoning-error-analysis-2026-07-25.md) — full analysis; §1 and §2 are the source findings, "Root enabler" callouts in both
- [locate-term-tool.md](locate-term-tool.md) — new tool sharing the same "make emptiness informative" goal; design together where they overlap
- [../../Decisions/sciqa-raw-sparql-keep-decision.md](../../Decisions/sciqa-raw-sparql-keep-decision.md) — documents the raw-SPARQL escalation pattern this task targets (agents fall back to `RunORKGSPARQL`/raw SPARQL when a wrapped tool goes empty, often making things worse)
- [../../System/agent_framework.md](../../System/agent_framework.md) — `BaseKBQAAgent` tool-result handling and journal-write path, the layer this contract change touches

## Problem

Across both KQAPro and SciQA, the recurring failure shape is: the agent forms a plausible model of
the question on its first tool call, that call returns an empty result for reasons the agent cannot
distinguish, and the agent never revises its model — it either concludes the fact is absent (§1,
schema-role confusion, 5/14 KQAPro failures) or abandons a correct constraint and widens the search
instead of fixing it (§2, "constraint dropped under pressure": idx 55, 98).

The idx 98 case is the clearest illustration: `GetRelationDetails(Q31, "film release region")`
returned all 311 Belgium-release films untruncated into context. A loop guard then fired on a raw
SPARQL attempt, and the agent fell back to `SelectExtreme(attribute_name="duration",
concept="feature film", mode="min")` — **dropping the Belgium constraint entirely** — and answered
*Gulliver's Travels* (zero `film_release_region` triples), when gold is *Alvin and the Chipmunks:
Chipwrecked*. The tool call needed to express the correct answer was available
(`SelectExtreme` accepts `entity_ids` + `concept` together); the failure was that composing it
correctly required pasting 311 entity IDs through context, which the agent didn't do and instead
silently dropped the constraint.

**Root enabler common to both patterns:** an empty (or oversized-and-unfilterable) result carries no
information about *why* it's empty or *what would fix it*. A wrong-namespace query and a
genuinely-absent fact look identical to the agent. Nothing forces model revision.

## Goal

On the **zero-result path only** (this is a targeted contract addition, not a rewrite of every tool's
return shape), have tools report which conjunct/filter of the query failed to match, so the agent can
repair the specific broken piece instead of either giving up or abandoning the whole constraint.

## Design

1. **Scope the change to the zero-result branch.** Every KQAPro/SciQA tool that composes a
   multi-conjunct SPARQL query (attribute filter + relation filter + qualifier filter, e.g.
   `CountEntities`, `FilterEntities`, `GetRelationDetails`) already knows its own conjuncts. When the
   final result set is empty, instead of returning bare `{}`/`[]`, re-run (or track during the main
   query) which individual conjunct(s) independently matched zero rows vs. which matched something
   but the *intersection* was empty.
2. **Report shape:** something like `{"result": [], "diagnostic": {"empty_conjuncts": ["attr:reviewer"], "matched_conjuncts": ["prop:film_release_region"], "note": "no entity has attr:reviewer — this term may be a relation or qualifier, not an attribute"}}`. Exact shape needs design work against the existing per-server envelope contract (see [abstract-operation-contract.md](../../Decisions/abstract-operation-contract.md)) — this is additive, not a shape change to the non-empty path.
3. **Do not touch the non-empty path.** Per the abstract-operation-contract stance, tool return
   shapes on the success path are deliberately stable to avoid moving benchmark numbers outside a
   controlled experiment. This task's contract change applies strictly to the zero-result case.
4. **Coordinate with `LocateTerm`** ([locate-term-tool.md](locate-term-tool.md)) where they overlap:
   `LocateTerm` is a dedicated tool for *proactively* checking a term's role before querying;
   this task is *reactive* diagnosis after a query already came back empty. Both should point the
   agent toward the same underlying "check the other two namespaces" recovery action, ideally with
   consistent wording so a fewshot teaching one reinforces the other.

## Risks

- **False precision.** A wrong or overly specific diagnostic message could send the agent down a
  *new* wrong path with high confidence, which is arguably worse than an honest empty result. Needs
  conservative diagnostic language ("this may be...") not asserted fact, especially for the
  fuzzy-match-driven suggestions.
- **Cost of computing the diagnostic.** Determining which conjunct(s) independently matched requires
  extra sub-queries (or restructuring the main query to expose per-conjunct match counts) on every
  zero-result call. Needs to stay cheap — these are exactly the empty/wasted calls the
  raw-SPARQL-keep-decision doc already found are common (idx 98's `GetRelationDetails` alone returned
  311 untruncated rows; adding sub-query cost to every empty call needs to not become the next
  latency problem).
- **Doesn't address the oversized-result half of §2 on its own.** idx 98's failure was partly "results
  too large to compose with" rather than purely "results empty." This task is scoped to the
  zero-result diagnostic; the oversized-result ergonomics problem is `SelectExtreme`'s relation filter
  (recommendation #6, [selectextreme-relation-filter.md](selectextreme-relation-filter.md)) — the two
  should land together since they attack the same idx 55/98 failure class from opposite ends.

## Validation plan

- Unit-level: verify the diagnostic fires correctly on synthetic zero-result queries with known
  conjunct failure (e.g. force an `attr:` miss when the predicate is actually a `prop:` relation).
- Targeted re-run of idx 6, 68, 74, 86, 90 (schema-role confusion, §1) and idx 55, 98 (constraint
  dropped under pressure, §2) — confirm the diagnostic changes agent behavior on these specific cases
  before running a full benchmark.
- Full n=100 KQAPro + SciQA benchmark A/B, since this changes tool *contracts* the agent reasons
  over, not just internal plumbing.

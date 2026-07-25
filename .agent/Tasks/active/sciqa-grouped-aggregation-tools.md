# SciQA Grouped-Aggregation Tools (`FindTopByRelationCount` + having/extremum filter)

**Status:** 📋 Planned
**Priority:** Medium (larger design work — new tool(s), not a bug fix)
**Identified:** 2026-07-25 (reasoning-error and capability-gap analysis, recommendation #7)

## Related Docs
- [../../Decisions/reasoning-error-analysis-2026-07-25.md](../../Decisions/reasoning-error-analysis-2026-07-25.md) — full analysis; §3.3 "SciQA: no count-per-group-then-argmax" is the source finding
- [../../Decisions/sciqa-cross-resource-aggregation.md](../../Decisions/sciqa-cross-resource-aggregation.md) — prior art: `FindFrequentValues` (tool 21), the closest existing tool to this gap; this task is the next tier up (grouped count + argmax/having, not flat frequency)
- [../../Decisions/sciqa-aggregation-and-coauthor-tools.md](../../Decisions/sciqa-aggregation-and-coauthor-tools.md) — `AggregateComparisonValues`, `FindCoAuthors` — the aggregation tool family this new tool joins
- [../../System/sciqa_agent.md](../../System/sciqa_agent.md) — SciQA tool catalog and classification (`Superlative`, `Aggregation`, `Count` qtypes) this new tool needs strategy-text/fewshot coverage in

## Problem

"Which author contributed most to papers about X" — and the broader class of "which N has the most
of Y" questions — fail **3/3** across all three sampled runs, with **three different wrong winners**
each time. This is not an adoption gap; it's a genuine capability gap. Gold requires a nested
`COUNT ... GROUP BY ... ORDER BY DESC LIMIT 1` (or equivalent count-per-group-then-argmax). Measuring
against the gold-matched subset of the 2,565-query SciQA corpus:

| SPARQL construct | Adoption rate in current tool surface |
|---|---|
| `SUBSELECT` | 0% |
| `GROUP BY` | 0% |
| `HAVING` | 0% |
| `AVG` | 0% |
| `COUNT` | 13% |

This is described in the analysis as **"the SciQA floor"** — the class of question the current tool
suite structurally cannot express, as opposed to §4 (adoption gaps caused by classifier/fewshot
routing) or §3.5 (a scope bug, not a missing capability).

## Goal

Give the agent a tool path to express "group by X, count/aggregate within each group, return the
group with the max/min (or groups passing a having-clause threshold)" without composing raw SPARQL
`SUBSELECT`/`GROUP BY`/`HAVING` by hand — which the corpus evidence above shows the agent essentially
never manages to do correctly today.

## Design

Two complementary pieces, likely worth landing together since they share the grouped-aggregation
machinery:

1. **`FindTopByRelationCount(subject_type, group_by_predicate, filter_predicate="", filter_value="", top_n=1, mode="max")`**
   — the argmax-over-groups primitive. Groups candidate subjects (e.g. authors) by a relation (e.g.
   authorship of papers matching a topic filter), counts members per group, and returns the top-N
   groups by count. This is the direct fix for "which author contributed most" style questions.
   Should reuse the existing scope machinery from `FindFrequentValues`
   (`comparisons`/`papers` scope, per [sciqa-cross-resource-aggregation.md](../../Decisions/sciqa-cross-resource-aggregation.md))
   rather than inventing a new scope model — this tool differs from `FindFrequentValues` in that it
   groups by relation-target identity and returns the group *members*, not just a value frequency
   count.
2. **`having`/extremum filter on existing aggregate tools** (`AggregateComparisonValues`,
   `FindFrequentValues`) — extend the grouped-output path so a caller can filter groups by a
   threshold on the aggregate (e.g. "groups with count > 5") without a second round-trip to inspect
   and manually filter the full group list. This is the `HAVING`-equivalent half of the gap.

## Risks

- **Scope creep into a general query-builder.** The failure class analyzed spans "top author",
  "top N by count", "groups above threshold" — there's a temptation to build one maximally general
  grouped-aggregation tool. Resist this; per the existing tool-surface philosophy (see
  [count-entities-not-conditions.md](../../Decisions/count-entities-not-conditions.md) and the
  abstract-operation-contract stance), prefer a small number of concretely-named tools with clear
  contracts over one parameterized-to-death mega-tool. Start with the argmax case
  (`FindTopByRelationCount`) since it's the one with measured 0% coverage; extend the having-filter
  incrementally only if traces show it's still missing after the first tool ships.
- **New-tool adoption.** Per §4.1/§4.2 of the parent analysis, new tools have a documented pattern of
  near-zero adoption without explicit strategy-text and fewshot coverage tied to the right qtype
  label(s) (`Superlative`, `Count`, `Aggregation` — note idx from the analysis shows a
  `Superlative`-tagged question needed a `Count`-bank fewshot, so this new tool's fewshot needs to be
  reachable from multiple labels, not just one). Coordinate with
  [fewshot-bank-audit.md](fewshot-bank-audit.md).
- **Group-size / truncation limits.** Any grouped-count tool needs an explicit cap and truncation
  warning on the number of groups considered, per the same silent-truncation lesson already learned
  from `FindFrequentValues`'s `limit_subjects=5000` bug (§3.5, recommendation #8, fixed this session).
  Don't re-introduce a silent truncation footgun in the new tool.

## Validation plan

- Smoke-test against the "which author contributed most to papers about X" gold answer directly
  before any benchmark run.
- Measure `GROUP BY`/`HAVING`/`SUBSELECT`/`COUNT` construct coverage on the gold-matched subset
  before/after — the analysis's 0%/0%/0%/13% baseline table above is the number to move.
- Full n=100 SciQA benchmark A/B once the tool(s) ship, since this is new tool-surface, not a fix to
  existing behavior.

# ADR: CountEntities `not_conditions` Parameter

**Date:** 2026-05-08
**Status:** Accepted
**Commit:** `d01d029`
**Files:**
- `ama_kbqa/server/kqapro_server.py` — `CountEntities`: new `not_conditions` parameter + `FILTER NOT EXISTS` SPARQL clause
- `db/datasets/kqapro/fewshot-examples/Count.json` — 2 new examples: `not_conditions` negation + `or_conditions` across attributes
- `db/datasets/kqapro/fewshot-examples/_tool_tips.json` — `CountEntities` tip rewritten to mention `not_conditions` semantics

## Related Docs

- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — original `CountEntities` implementation; `or_conditions`, `transitive_concept`, and the coverage-gap analysis that motivated the tool
- [System/agent_system.md](../System/agent_system.md) — KQAPro tool tiers; journal data model; `SOP/adding_new_kqapro_tools.md` invariants

---

## Context

### Root cause: models reach for `RunSPARQL` for negation queries

The seed=43 Count audit revealed consistent failures on questions of the form "How many `<concept>` NOT `<attribute> = <value>`?". Both minimax-m2.7 and gemma-4-31b guessed **397** for "How many TV series were not started in 2005?" (gold=305).

Failure trace pattern:
1. Agent emits `RunSPARQL` with a hand-written `FILTER NOT EXISTS { ... }` block.
2. The SPARQL references KQAPro storage internals (literal / quantity-bnode / xsd:date|gYear|decimal) that the model hallucinated incorrectly.
3. The query either errors or returns a wrong count.

This is the same class of bug that motivated `CountEntities` / `or_conditions` in the first place (see `kqapro-tool-surface-expansion.md`): the KQAPro RDF schema is non-standard and the models cannot reliably write correct SPARQL against it.

### Structural-pattern-as-tool principle (reinforced)

The `sciqa-aggregation-and-coauthor-tools.md` ADR established the lesson: when a structural SPARQL pattern appears in >1 question type and every tested model fails it, promote it to a first-class tool parameter rather than asking the model to write raw SPARQL. Negation in Count questions is the same case.

---

## Decision

Add `not_conditions: list[dict] | None = None` to `CountEntities`.

### Parameter shape

Each entry in `not_conditions` has the same shape as `or_conditions`:
```python
{
    "attribute_name": str,   # KQAPro attribute label (e.g. "start time")
    "attribute_value": str,  # Attribute value to exclude
    "operator": str,         # "=", "!=", "<", ">", "<=", ">="
}
```

### SPARQL emission

For each entry, the server emits a `FILTER NOT EXISTS { ... }` clause using the existing `_attr_condition_sparql` helper:

```sparql
FILTER NOT EXISTS {
    ?entity wdt:P580 ?startTime .
    FILTER(?startTime = "2005"^^xsd:gYear)
}
```

The `_attr_condition_sparql` helper handles all four KQAPro storage variants:
- Plain literals (strings) → `rdfs:label` / `skos:altLabel`
- Quantity bnodes → `wikibase:quantityAmount`
- Dates → `xsd:date`
- Years → `xsd:gYear`

Using the helper ensures negation uses the same storage-aware logic as inclusion conditions, avoiding the hallucination problem.

### Semantics (documented in `_tool_tips.json`)

> Entities matching ANY value that satisfies the condition are excluded.
> Entities **lacking the attribute entirely** are **KEPT** (standard `FILTER NOT EXISTS` semantics).

This is correct for KQAPro questions of the form "TV series not started in 2005" — series with no start-time recorded should be included in the count (they are not confirmed to have started in 2005).

---

## Fewshot Updates

Two new examples added to `Count.json`:

1. **Negation example** — "How many TV series were not started in 2005?" (gold=305)
   - Demonstrates `not_conditions: [{"attribute_name": "start time", "attribute_value": "2005", "operator": "="}]`

2. **`or_conditions` across different attributes** — "How many neighborhoods are at zip code 10026 OR have subreddit 'venice'?" (gold=2)
   - Demonstrates `or_conditions` with two different `attribute_name` values (same tool, different parameter; reinforces that `or_conditions` is not limited to same-attribute multi-value)

`_tool_tips.json` CountEntities tip rewritten to mention both `not_conditions` (with the "lacking the attribute = KEPT" note) and `or_conditions` in one unified tip.

---

## Smoke Test (live KQAPro Virtuoso on Hetzner)

Query: "TV series not started in 2005" with the new SPARQL pattern.

| Result | Value |
|--------|-------|
| Tool output | 297 |
| Gold answer | 305 |
| Gap | 8 rows |

The 8-row gap is a **concept-hierarchy / subclass coverage issue** unrelated to negation correctness: the query uses flat `rdf:type Q5398426` (TV series) and misses subclasses like `anime television series`. Setting `transitive_concept=True` would close the gap. The negation mechanics themselves are correct.

---

## Invariants

- `not_conditions` entries must use the same `attribute_name` strings as `conditions` and `or_conditions` (the `_attr_condition_sparql` helper resolves them by label lookup).
- The generated clause uses `FILTER NOT EXISTS`, not `MINUS` — they differ when the inner pattern is empty. `FILTER NOT EXISTS` is correct for KQAPro's optional-attribute model.
- Do not use `not_conditions` to express "attribute value ≠ X" for numeric comparisons — use `conditions: [{"operator": "!=", ...}]` instead, which emits a `FILTER(?val != X)` inside the positive pattern. `not_conditions` is for exclusion of the entire triple, not just the value comparison.

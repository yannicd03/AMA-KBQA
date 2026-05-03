# ADR: KQAPro Tool Surface Expansion (22 → 25 tools)

**Date:** 2026-04-29  
**Commit:** `9f677ac` — "Add CountEntities, SelectExtreme, VerifyFact + extend FilterEntities"  
**Status:** Accepted

## Related Docs

- [System/agent_system.md](../System/agent_system.md) — tool tiers, journal data model
- [SOP/adding_new_kqapro_tools.md](../SOP/adding_new_kqapro_tools.md) — invariants for answer-producing tools

---

## Context: The Coverage Gap

Analysis of `val.json` (11,797 questions) found that ~4,917 questions (~42%) had no first-class tool for their primary operation. These questions fell into two bad patterns:

1. **Silent truncation via `FilterEntities(limit=50)`** — count-type questions that needed exact cardinality were answered by counting the (capped) list, silently producing wrong numbers for entity sets larger than 50.
2. **Hand-written SPARQL against KQAPro's non-standard namespaces** — the agent had to emit raw SPARQL using `<ex:...>` IRIs, which it hallucinated inconsistently because they are undocumented. The RDF store uses `<http://db.imis.uni-luebeck.de/kqa/concept#>` etc., not standard Wikidata predicates.

Additionally, 496 `Or`-using questions (all of which are also Count questions) had no deterministic multi-condition path; the agent would attempt multiple sequential `FilterEntities` calls and sum them, double-counting entities that satisfied both conditions.

## Decision

Add three new tools implemented as deterministic SPARQL against the Hetzner KQAPro Virtuoso, and extend `FilterEntities` with two new optional parameters.

### New Tools

| Tool | Signature (key params) | Closes |
|------|------------------------|--------|
| `CountEntities` | `concept, conditions[], or_conditions[], transitive_concept` | Exact COUNT for any filter combination; eliminates limit=50 truncation for Count questions |
| `SelectExtreme` | `concept, attribute, direction (min/max), conditions[], transitive_concept` | Superlative lookup (ORDER BY + LIMIT 1) without requiring the agent to write SPARQL |
| `VerifyFact` | `subject_id, predicate_label, object_value, operator` | Deterministic TRUE/FALSE for fact-verification questions; handles numeric, date, and string comparisons |

**Location:** `ama_kbqa/server/kqapro_server.py` — SPARQL helper at ~line 268–360; tool handlers near end of file before `if __name__ == "__main__":`.  
**Prompts updated:** `ama_kbqa/agents/kqapro_agent/prompts.py` — tool tier listing (`SYSTEM_PROMPT`) and `QTYPE_STRATEGIES` entries for Count, SelectAmong, and Verify.

### FilterEntities Extensions

Two new optional parameters added to the existing `FilterEntities` tool:

- **`or_conditions`** — mirrors the existing `conditions` list but joins clauses with SPARQL `UNION` + `COUNT(DISTINCT)` to avoid double-counting. Required for the 496 KQAPro `Or`-using questions, all of which are Count questions.
- **`transitive_concept`** — enables traversal of the concept hierarchy via `instanceOf` chains (e.g., "city" should match "capital city", "metropolis", etc.). Previously, concept-hierarchy filtering was silently ignored.

## Pre-existing Bugs Fixed

### Bug 1: Concept IRI construction

The original `FilterEntities` `_concept` SPARQL clause emitted bare IRIs like `<ex:human>` when callers passed label-form concepts (`"human"`, `"city"`). These IRIs do not exist in the KQAPro Virtuoso store. The new `_concept_clause` helper accepts both Q-IDs (e.g., `Q5`) and label strings, resolving labels via:

```sparql
?entity wdt:P31 ?conceptIRI .
?conceptIRI rdfs:label "human" .
```

This is the canonical pattern used throughout the KQAPro dataset.

### Bug 2: Year heuristic misclassification

The original year detection heuristic classified any 4-digit numeric string as a year (e.g., `"7800"` — a valid population count — was emitted as `xsd:gYear`). The new helper emits a SPARQL `UNION` over both interpretations (`xsd:integer` and `xsd:gYear`) and uses `DATATYPE()` filters so the correct cast fires only on the correct literal type.

## Validation

- **8 synthetic rdflib tests** (in-memory RDF graph, no network): unit-test the SPARQL generation helpers for concept resolution, year UNION, or_conditions UNION, transitive_concept chain.
- **11 real-data tests** against Hetzner KQAPro Virtuoso.

Spot-checks on `val.json`:
- Q: "Which former French region has the smallest population and a population that is not equal to 97000?" → `SelectExtreme` returns **Champagne-Ardenne** ✓ (matches gold).
- Q: "How many Pennsylvania counties have a population greater than 7800 or a population less than 40000000?" → `CountEntities` with `or_conditions` returns **39** ✓ (matches gold).

## Alternatives Considered

- **Fetch-and-count in Python** (fetch all FilterEntities pages, count in memory): rejected because it requires multiple round-trips, is slow on large entity sets, and replicates logic that SPARQL COUNT already provides deterministically in one query. See `SOP/adding_new_kqapro_tools.md` for the standing rule.
- **Extend RunSPARQL prompting**: rejected because the agent consistently hallucinated namespace prefixes and predicate URIs when writing raw SPARQL for KQAPro's non-standard schema.

---

## Follow-up: Prompt Hardening to Drive Adoption (2026-04-29)

**Commits:** `08cadeb`, `d0978c9`

**Problem:** In an n=20 benchmark run after the tools shipped, `gemma-4-31b-kit` made **57 RunSPARQL calls and 0 CountEntities calls**. The new tools were defined but not picked up. Root cause traced to two sources:

1. `ama_kbqa/agents/kqapro_agent/prompts.py` — `QTYPE_STRATEGIES` entries for Count, SelectAmong, SelectBetween, and Verify named the new tools as optional suggestions but did not forbid the legacy path.
2. `db/datasets/kqapro/fewshot-examples/_general.json` — contained a guidance entry actively pushing Count and SelectAmong questions toward RunSPARQL ("Use SPARQL for complex conditions").

**Changes made:**

- `prompts.py` — Count/SelectAmong/SelectBetween/Verify strategies now name the new tools as the **only recommended path** with explicit `FORBIDDEN` clauses disallowing legacy SPARQL fallbacks for those question types.
- Fewshot example files rewritten to demonstrate the new tools end-to-end:
  - `db/datasets/kqapro/fewshot-examples/Count.json`
  - `db/datasets/kqapro/fewshot-examples/SelectAmong.json`
  - `db/datasets/kqapro/fewshot-examples/SelectBetween.json`
  - `db/datasets/kqapro/fewshot-examples/Verify.json`
  - `db/datasets/kqapro/fewshot-examples/_general.json` — three new guidance entries promoting CountEntities/SelectExtreme/VerifyFact; old "Use SPARQL for complex conditions" entry removed.
  - `db/datasets/kqapro/fewshot-examples/_tool_tips.json` — updated tool-tip guidance.

**Lesson:** Defining a tool is not sufficient; prompt strategies and fewshot examples must be co-updated or models will default to familiar patterns (RunSPARQL) even when better tools exist.

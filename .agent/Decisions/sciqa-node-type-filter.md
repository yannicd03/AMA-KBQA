# ADR: SciQA `FindResource` `node_type_filter` Parameter

**Date:** 2026-05-08
**Status:** Accepted
**Commit:** `d01d029`
**Files:**
- `ama_kbqa/server/sciqa_server.py` — `FindResource`: new `node_type_filter: str = ""` parameter + SPARQL ASK-batch pruning
- `ama_kbqa/agents/sciqa_agent/prompts.py` — `Aggregation` fewshot updated: patient-count example leads with `FindResource(..., node_type_filter="Comparison")`

## Related Docs

- [Decisions/sciqa-aggregation-and-coauthor-tools.md](./sciqa-aggregation-and-coauthor-tools.md) — `AggregateComparisonValues` ADR; context for why aggregation failures motivated this fix
- [System/agent_system.md](../System/agent_system.md) — SciQA tool table and question-type strategies; `FindResource` is Tier 1

---

## Context

### The failure mode: wrong resource type silently fed to `AggregateComparisonValues`

After fewshot hardening (qualifier-and-format prompt-hardening bundle), minimax-sciqa-v2 stayed flat at 0.60 accuracy across two seeds. Trace analysis of aggregation failures showed a consistent pattern:

1. Agent calls `FindResource("patient demographics studies")` (or similar topic query).
2. Vector search returns high-scoring Papers and Contributions alongside Comparisons.
3. Agent picks the top result (often a Paper or Contribution with a closely matching label).
4. That ID is passed to `AggregateComparisonValues`.
5. `AggregateComparisonValues` runs SPARQL over the wrong resource and returns nonsense numbers (e.g., 217918 patients vs gold 6452).
6. The agent has no signal that the number is wrong — it accepts and records it.

The core problem: `FindResource` is a vector similarity search over ORKG resources. All ORKG resource classes (Paper, Contribution, Comparison, Author, Dataset, Model, Metric) share the same embedding space and often have overlapping topic labels. A query for "patient demographics" returns Papers about patient demographics as readily as Comparisons about patient demographics.

### Why this wasn't caught by the fewshot examples

The fewshot examples for Aggregation showed `FindResource` calls without type constraints. The agent learned the pattern correctly but had no mechanism to filter results by type — so it would pick whatever had the highest cosine similarity, regardless of class.

---

## Decision

Add `node_type_filter: str = ""` to `FindResource`.

### Mechanism

1. **Over-fetch:** When `node_type_filter` is set, the candidate pool is expanded to `4 × top_n` results from the vector index.
2. **SPARQL ASK-batch pruning:** A single SPARQL query runs a `VALUES + rdf:type` check against `orkgc:<filter>` for all candidate IDs:

```sparql
SELECT ?resource WHERE {
    VALUES ?resource { <orkgr:R155266> <orkgr:R75638> <orkgr:R33008> ... }
    ?resource rdf:type <https://orkg.org/class/<node_type_filter>> .
}
```

3. **Return top_n from matches:** From the filtered set, the top `top_n` results (by original cosine score) are returned.

### Supported filter values

The parameter accepts any ORKG class name. Documented values in the tool tip and fewshot:
- `"Comparison"` — most important; used for aggregation questions
- `"Paper"` — filter to paper resources
- `"Contribution"` — filter to contribution resources
- `"Author"` — filter to author resources
- `"Dataset"` — filter to dataset resources
- `"Model"` — filter to model resources
- `"Metric"` — filter to metric resources

Empty string (default) disables filtering — existing behavior is preserved.

### Fewshot update (`Aggregation` strategy in `prompts.py`)

The patient-count example was updated to lead with:
```python
FindResource("patient demographics studies", node_type_filter="Comparison")
```

A sanity-check hint was added: "If the result looks implausible (e.g., 217918 patients vs an expected ~6000), the comparison_id is wrong — search with a different query or try other results."

---

## Smoke Test (live Hetzner Virtuoso)

Test query: `FindResource("patient demographics", node_type_filter="Comparison")` vs `FindResource("patient demographics", node_type_filter="Paper")` against the same 6 candidate IDs (R155266, R75638, R33008, + 3 papers).

| Filter | Matches |
|--------|---------|
| `orkgc:Comparison` | 3 (R155266, R75638, R33008 — the 3 known Comparisons) |
| `orkgc:Paper` | 0 |

The filter correctly identifies all three Comparisons and rejects all Papers. A `Paper`-filter returning 0 confirms that the pruning is actually discriminating, not returning everything.

---

## Why Not a Separate `FindComparison` Tool?

A dedicated `FindComparison(topic)` was considered but rejected:

| Option | Assessment |
|--------|-----------|
| `FindComparison(topic)` | Clean API; but proliferates the tool surface (one more tool name to learn per resource class) |
| `FindResource(..., node_type_filter=...)` | One parameter extension; same tool name the model already calls; works for all 7 ORKG classes without 7 new tools |

The parameter approach is consistent with the existing `CountEntities` / `or_conditions` / `not_conditions` pattern of extending tools rather than multiplying them.

---

## Invariants

- The `node_type_filter` value is injected directly into a SPARQL IRI: `<https://orkg.org/class/<node_type_filter>>`. Do not allow user-supplied arbitrary strings without validation in a production context.
- When `node_type_filter` is set and the over-fetched pool contains fewer matches than `top_n`, return all matches (do not error or pad with unfiltered results).
- The 4× over-fetch ratio is a heuristic. If aggregation questions consistently return 0 matches (all top-4×n results are the wrong type), increase the multiplier — do not disable the filter.
- Always update the Aggregation fewshot examples when the canonical Comparison IDs for a question change (e.g., after a data refresh of the ORKG endpoint).

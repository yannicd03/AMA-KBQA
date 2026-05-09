---
# ADR: SciQA AggregateComparisonValues and FindCoAuthors Tools

**Date:** 2026-05-05  
**Status:** Shipped  
**Files:** `ama_kbqa/server/sciqa_server.py` (tools 19/20), `ama_kbqa/agents/sciqa_agent/prompts.py`

## Related Docs
- [Project Architecture](../System/project_architecture.md) — MCP server tier listings and tool counts
- [Agent System](../System/agent_system.md) — SciQA tool table and question-type strategy table
- [Decisions/kqapro-tool-surface-expansion.md](kqapro-tool-surface-expansion.md) — analogous coverage-gap analysis for KQAPro

---

## Context

An audit of `db/sciqa_questionnaire.json` (the standard n=20 SciQA benchmark set) combined with historical traces from:
- `used_results/sciqa-minimax-m2.1-50/`
- `benchmark_results/2026-02-09_sciqa-qwen3-32b-n20-45pct/`

revealed two systematic failure categories that affected every tested model (minimax-m2.1, qwen3-32b, gpt-oss-120b):

| Category | Questions | Failure mode |
|----------|-----------|--------------|
| Aggregation over Comparison contributions | 9/20 (Q3, Q11, Q12, Q13, Q16, Q18, Q19, …) | Agent skipped aggregation steps or guessed from partial HAS_VALUE values |
| Co-author enumeration | 1/20 (Q2) | Agent confused paper-level author listing with co-author shared-count pattern |

The agent *could* compose `RunORKGSPARQL` for both patterns, but in practice it either:
1. Ran per-value SPARQL loops without aggregating, or
2. Stopped after retrieving one author's papers without cross-referencing co-authors.

---

## Decision

Add two tools to `sciqa_server.py` (registered with `@mcp.tool()`, inserted just before `ManageJournal`), numbered 19/20 in the file's section comments. `ManageJournal` and `GetJournalSummary` header comments were bumped accordingly.

Advertise both tools under TIER 3 in `SYSTEM_PROMPT`'s `KNOWLEDGE GRAPH ACCESS TOOLS` section in `ama_kbqa/agents/sciqa_agent/prompts.py`.

### Tool 19: `AggregateComparisonValues`

**Signature:**
```python
AggregateComparisonValues(
    comparison_id: str,         # ORKG Comparison resource ID (e.g. "R155266")
    value_predicate: str,       # Predicate for the numeric/text value (e.g. "P43156")
    agg: str,                   # Aggregation: avg|sum|min|max|count|count_distinct|mode_top|all_values
    group_by_predicate: str | None = None,   # Predicate to group results by
    filter_predicate: str | None = None,     # Pre-filter contributions by this predicate
    filter_value: str | None = None,         # Expected filter predicate value
    filter_match: str | None = None,         # "exact"|"contains"|"regex" (default "contains")
    top_n: int | None = None,               # Truncate results to top N groups
    value_via_group: bool = False,           # 2-hop: ?contrib group_pred ?group; ?group value_pred ?valueObj
)
```

**What it does:**

Runs a single SPARQL that:
1. Joins `compareContribution` to enumerate the comparison's contributions.
2. Lifts `HAS_VALUE` and `rdfs:label` indirection (the common pattern where a contribution links to an intermediate resource whose `HAS_VALUE` holds the scalar, and whose `rdfs:label` holds the string label).
3. Optionally pre-filters contributions by `filter_predicate`/`filter_value` before aggregating.
4. Optionally groups results by another predicate (`group_by_predicate`) and returns one aggregate per group.
5. Computes the aggregate in Python over parsed float values (safe against SPARQL type heterogeneity).

The `value_via_group=True` switch handles the 2-hop case where the measurement hangs off the grouping node rather than the contribution directly:
```sparql
?contrib group_pred ?group .
?group value_pred ?valueObj
```
This covers Q18 (energy sources × min/max installed capacity, predicate P43135 → P43133) and similar grouped energy-domain questions.

**Supported aggregations:**
- `avg`, `sum`, `min`, `max` — numeric; values parsed with `float()`
- `count` — total values (including duplicates)
- `count_distinct` — unique values only
- `mode_top` — most frequent string value (with count)
- `all_values` — return all values unsummarized (for debugging or list questions)

### Tool 20: `FindCoAuthors`

**Signature:**
```python
FindCoAuthors(
    author_name: str,   # Seed author name (case-insensitive partial match)
    top_n: int = 20,    # Max co-authors to return
)
```

**What it does:**

Single SPARQL that:
1. Finds papers by anyone whose `rdfs:label` contains the seed name (using `FILTER(CONTAINS(LCASE(?authorLabel), LCASE("name")))`); handles both resource-typed authors (`orkgp:P27/P6 ?author . ?author rdfs:label ?label`) and literal string authors (`orkgp:P27/P6 ?authorLabel . FILTER(isLiteral(?authorLabel))`).
2. Enumerates all other authors on those same papers (co-authors).
3. Returns co-authors sorted by shared paper count descending, with the shared papers listed.

---

## Why Not Just Use RunORKGSPARQL?

The agent *can* write SPARQL. The problem is **execution discipline**, not expressiveness:

- Aggregation queries for comparison data require knowing the 3-node pattern (`Comparison → compareContribution → Contribution → value_pred → ValueNode → HAS_VALUE → scalar`). Models consistently missed the `HAS_VALUE` hop or aggregated at the wrong level.
- Co-author queries require a self-join on papers: models retrieved one author's papers but did not cross-join to find co-authors with counts.
- Both failures were model-independent — every tested model reproduced them. This indicates the failure is structural (the query pattern is too deep for reliable in-context SPARQL composition) rather than model-specific.

A purpose-built tool encapsulates the correct SPARQL pattern and delegates only the aggregation function selection to the model — a much narrower decision.

---

## What Is Intentionally Out of Scope

| Question | Pattern | Why excluded |
|----------|---------|--------------|
| Q12 (energy sector via P37586→P37675→P37668→label) | 3-hop beyond `value_via_group` | Falls back to `RunORKGSPARQL`; adding another hop parameter was judged not worth the API complexity |
| Q4 (% comparisons without class link) | Global graph-wide COUNT | Requires a full-graph scan; better expressed as a raw SPARQL query |

---

## Validation

End-to-end smoke tests against live Hetzner Virtuoso confirmed gold-answer reproduction:

| Question | Gold | Tool result |
|----------|------|-------------|
| Q3 mean efficiency R155266/P43156 | 93.3125 | 93.3125 ✓ |
| Q11 Chloride count R110597/P37458 | 2 | 2 ✓ |
| Q13 mode lead compound R75638/P35194 | "Aurein 1.2" | "Aurein 1.2" ×6 ✓ |
| Q16 sum patients R33008/P15585 | 6452 | 6452 ✓ |
| Q18 min/max grouped R153801 P43133 via P43135 | per-source pairs | matches all ✓ |
| Q2 co-authors of "Kurt Thomas" | 4 papers, 7 co-authors | {Chris Grier, Vern Paxson, Alek Kolcz, Damon McCoy, Dawn Song, Frank Li, Michael Zhang} ✓ |

---

## Deployment Note

As of 2026-05-05, the tools exist locally but have **not yet been deployed** to `hetzner:~/AMAKBQA-main/`. The sciqa benchmark runs queued for 2026-05-05 (`benchmark_results/sciqa-{minimax,gemma}-2026-05-05`, waiting on PID 3195923) will run against the existing 18-tool surface as a clean baseline. A separate post-deployment run is needed to measure impact.

**2026-05-09 update:** `AggregateComparisonValues` gained a new `comparison_ids` parameter (multi-Comparison union mode) and `FindFrequentValues` was added as a new tool for cross-resource aggregation. See `Decisions/sciqa-cross-resource-aggregation.md` for the full ADR on these changes.

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Closes 9/20 aggregation questions and 1/20 co-author question for all models | Increases tool surface from 18 to 20; more for the model to learn |
| Encapsulates the `HAS_VALUE` + label indirection pattern reliably | `AggregateComparisonValues` has 9 parameters — the most complex SciQA tool so far |
| `value_via_group=True` handles grouped energy-domain patterns without a new SPARQL | 3-hop patterns (Q12, Q4) still require `RunORKGSPARQL` |
| Smoke-tested against live Virtuoso with gold answers | No automated unit tests; repo has no SciQA tool test suite |

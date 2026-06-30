# ADR: WikiKGQA 2026 Challenge Adaptation

**Date:** 2026-06-25  
**Branch:** `dev-routing-evidence` (worktree `wikikgqa-2026`)  
**Status:** Settled — implementation in progress  
**Deadlines:** Codabench solution 2026-07-03 · Paper 2026-07-24 · Venue ISWC 2026, Bari

## Related Docs
- [System/project_architecture.md](../System/project_architecture.md) — overall system and backend pattern (KQAPro/SciQA reference)
- [System/agent_system.md](../System/agent_system.md) — agent lifecycle, question classification, tool loop
- [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) — benchmark runner (WikiKGQA scorer differs — see below)

---

## Challenge Overview

WikiKGQA: given a natural-language question (English or Spanish), produce answers over a fixed snapshot of Wikidata. Four submission tracks: {English, Spanish} × {with-mentions, without-mentions}. Metric: **Macro QALD F1** over executed answer sets.

**Submission format** (verified against train/test JSON, Zenodo v2026.2, doi `10.5281/zenodo.19474009`): extended QALD-JSON. Each question carries bilingual `question` strings, optional `mentions` (entity QIDs + property PIDs, with an `inverse` flag), an **optional** `query.sparql` field ("preferably add if your approach generates one"), and a **mandatory** `answers` field holding W3C SPARQL-JSON results (`head.vars` + `results.bindings`, or `boolean` for ASK queries).

**Training data:** 497 fully-labeled questions downloaded to `data/wikikgqa/wikikgqa.json`. Test sets: 75 questions each (with/without mentions). Answer distribution: 462 single-column SELECT + 35 boolean ASK; answer values are Wikidata URIs/literals; result-set sizes range from 1 to ~847k rows.

---

## Decision 1 — Replace NL synthesis head with SPARQL-generate → execute → return-bindings

**Chosen:** Replace the existing LLM-judge synthesis head (NL answer + rubric scoring) with a SPARQL generation → endpoint execution → canonical bindings passthrough.

**Rejected:** Generating an NL answer and scoring it with an LLM judge.

**Rationale:** QALD F1 compares exact entity sets (`wd:Q…` URIs, typed literals). A semantically correct NL answer cannot match `wd:Q…` identifiers and will score 0. Executing generated SPARQL against the challenge endpoint mints the canonical answer set in the required format.

**Implications:**
- `answers` field = raw SPARQL-JSON from endpoint (pass-through, no reformatting).
- `query.sparql` field = generated SPARQL query (optional but submitted when available).
- The generate→execute→validate loop enables self-correction: if execution fails or returns an empty set, the agent can revise.
- The existing LLM-judge scorer in `ama_kbqa/benchmark/` is **not** reused for WikiKGQA; a dedicated Macro QALD F1 scorer lives in `ama_kbqa/wikikgqa/`.

---

## Decision 2 — No entity/relation embedding; use live SPARQL as the index

**Chosen:** Zero local Wikidata embedding. Graph exploration via live SPARQL against the challenge endpoint.

**Rejected:** Embedding Wikidata entities/relations into Qdrant (the KQAPro/SciQA pattern).

**Rationale:** Wikidata contains ~115M entities. At 4096 dimensions that is ~1.6 TB of vectors — infeasible to index locally and unnecessary given the endpoint is available. The KQAPro KB (1.6M triples) is small enough to embed wholesale; Wikidata is not.

**Substitutions by component:**

| KQAPro/SciQA component | WikiKGQA equivalent |
|---|---|
| Qdrant dense search | Live SPARQL exploration against the challenge endpoint |
| Local entity linking | With-mentions track: QIDs/PIDs provided directly (no linking needed) |
| Local entity linking | Without-mentions: `wbsearchentities` / `mwapi SERVICE` Wikidata entity search API |
| Relation embedding | ~12k Wikidata properties are small enough to optionally embed for without-mentions relation matching only |

---

## Decision 3 — Backend = challenge SPARQL endpoint; dev against public WDQS

**Chosen:** Challenge SPARQL endpoint (URL on Codabench, registration-gated) as the production backend. Dev and smoke-testing against public WDQS (`query.wikidata.org`).

**Rejected:** Self-hosting the 121 GB Wikidata dump as a Virtuoso instance.

**Rationale:** The dump is large and slow to ingest. The challenge provides a dedicated endpoint that is the authoritative answer oracle. Self-hosting is a fallback only if rate-limiting makes the endpoint impractical.

**Critical invariant:** The exploratory `RunSPARQL` tool truncates results for display. The **answer-generation query must not truncate** — the full result set must be captured and placed verbatim in the `answers` field.

---

## Decision 4 — Initial target: English with-mentions track

**Chosen:** Start with the English with-mentions track.

**Rationale:** The with-mentions track provides QIDs and PIDs directly, eliminating entity/relation linking — the hardest unsolved component for Wikidata scale. This reduces the initial port to: (a) re-point exploration tools to live SPARQL, and (b) swap the synthesis head to SPARQL-generate → execute → bindings. The without-mentions and Spanish tracks are planned follow-ons once the core pipeline is validated.

**Open:** Whether to attempt without-mentions and/or Spanish tracks before the deadline.

---

## Decision 5 — New `wikidata` backend; WikiKGQA concerns in `ama_kbqa/wikikgqa/`

**Chosen:** Implement a new `wikidata` backend mirroring the `kqapro` / `sciqa` pattern, plus a dedicated `ama_kbqa/wikikgqa/` module for challenge-specific concerns.

**Structure:**

```
ama_kbqa/
  wikidata/
    adapter.py          # WikidataAdapter: wd/wdt/p/ps/pq prefixes, Q/P id patterns
    server.py           # wikidata_server.py MCP: live-SPARQL exploration tools
    agent.py            # WikidataAgent
  wikikgqa/
    loader.py           # QALD-JSON loader for training/test data
    scorer.py           # Macro QALD F1 (must match Codabench/GERBIL-QA definition)
    submission.py       # Submission-file writer (QALD-JSON + answers field)
    runner.py           # Dedicated benchmark runner (separate from LLM-judge harness)
```

**Config:** `[wikidata]` section in `config.toml`, following existing `[kqapro]` / `[sciqa]` pattern.

**Why separate `wikikgqa/` module:** The scoring semantics differ fundamentally from KQAPro/SciQA (exact URI/literal set matching vs LLM rubric). Mixing them into the existing benchmark runner risks silent scorer substitution errors.

---

---

## Decision 6 — Entity/relation linking strategy for without-mentions track

> **Scope:** Challenge work only. Porting any of this to the KQAPro/SciQA main branch is out of scope until the challenge results justify evaluating it.

**Three sub-cases, handled differently:**

### With-mentions (initial target)
No linking component needed — QIDs and PIDs are provided in the input. Build nothing here.

### Entities (without-mentions)
**Chosen:** Out-of-band HTTP calls to the Wikidata `wbsearchentities` MediaWiki API, followed by LLM disambiguation over the small candidate set.

**Rejected:** Embedding ~115M entities locally; using the in-SPARQL `mwapi SERVICE` clause.

**Rationale:** Embedding is infeasible (see Decision 2). Lexical label/alias search is precise on entity names — empirically verified: "Stranger Things" → `Q19798734` as top hit; Spanish "Esquisto de Burgess" → `Q852085`. The `mwapi SERVICE` clause cannot be assumed available on the challenge's plain SPARQL endpoint. Because QIDs are stable Wikidata identifiers, it is safe to link against live Wikidata and then run SPARQL on the pinned challenge endpoint — the QIDs transfer.

**Flow:** `wbsearchentities(query, language)` → small candidate list (label + description) → LLM picks best match given question context → QID used in generated SPARQL.

### Properties (without-mentions)
**Chosen:** Small semantic index over the ~13.6k Wikidata properties, queried by embedding similarity.

**Rejected:** Lexical search alone.

**Rationale:** Lexical search is too weak on indirect phrasing — empirically verified: searching "played" returns junk like `P54` (member of sports team) and misses `P161` (cast member) and `P175` (performer). Property aliases are the key signal: `P175` has aliases "played by"/"portrayed by"; `P161` has "actor"/"starring". Embedding the full `"label. description. aliases"` string captures this.

**Implementation:** See Decision 7 for the property index build recipe.

### Pluggable linker backend
The linker is parameterized with a backend selector: `given-mentions | wbsearchentities | property-semantic`. Default: `given-mentions` (with-mentions track). The without-mentions path activates `wbsearchentities` for entities and `property-semantic` for relations.

---

## Decision 7 — Property semantic index build recipe

> **Scope:** Challenge work only. Porting to the KQAPro/SciQA main branch is out of scope until challenge results justify it.

**Chosen:** Enumerate all Wikidata properties via a single SPARQL query; embed `"label. description. aliases"` text; store in a small Qdrant `wikidata-properties` collection (or in-memory, brute-force cosine — both viable at this size).

**Rejected:** Extracting properties from the 121 GB Wikidata dump; relying on lexical property search.

**Key facts (verified against WDQS):**
- Wikidata has **13,586 properties** total.
- One SPARQL query over `wikibase:Property` returns label + description + `skos:altLabel` aliases for both `en` and `es`. No dump needed.
- Aliases are the critical signal (e.g. `P175 performer` aliases include "played by"/"portrayed by"; `P161 cast member` aliases include "actor"/"starring").

**Size:** ~13.6k vectors at 4096 dimensions ≈ ~200 MB. ~13.6k embedding calls, a few minutes, one-time. This is small enough that a Qdrant server is optional — an in-memory array with brute-force cosine is equally valid.

**Build target:** WDQS (`query.wikidata.org`) for dev; challenge endpoint when available. Properties are stable so the source barely matters.

**Scope boundary:**
- Properties only — not entities (entities use `wbsearchentities` live API, no embedding).
- Without-mentions track only — with-mentions needs none of this.

---

## Open Questions

| Question | Blocking? | Notes |
|---|---|---|
| Official Macro QALD F1 scorer definition | Yes (for valid evaluation) | Must match Codabench's scorer exactly; verify against GERBIL-QA / qald-eval reference impl |
| Challenge endpoint URL | Yes (for final submission) | Gated behind Codabench registration |
| Without-mentions track | No (not initial target) | Decisions 6+7 cover the approach; activate after with-mentions is validated |
| Spanish without-mentions | No | `wbsearchentities` supports `language=es`; property index includes `es` altLabels — no structural blocker |
| Result-set truncation in self-correction loop | Design | Exploratory queries can truncate; final answer query must not |

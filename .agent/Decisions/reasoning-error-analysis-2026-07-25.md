# Reasoning-Error and Capability-Gap Analysis — 2026-07-25

**Date:** 2026-07-25
**Scope:** Why the KBQA agents get questions wrong, and which of those causes are missing
capabilities versus unused ones.
**Status:** Analysis complete and triaged. Small fixes (items 3, 4, 5, 8, 10 below) **landed and
were verified** in commit `8d519ca`; a second batch landed in `e1ce3bf` (see the addendum at the
end of this document). Items 1, 2, 6, 7, 9 remain deferred to a dedicated session (see the five
`Tasks/active/` docs and the existing `deferred-tool-loading.md`).

> This status line originally read "being implemented concurrently, do not treat as verified".
> Updated 2026-07-25 after verification. Retained here rather than deleted, because the
> verification turned up something worth keeping: see below.

**Verification note — mock tests passed a broken fix.** Item 3 (the qualifier tools' bnode
assumption) was implemented with 15 passing unit tests and was still wrong. The tests use a
`FakeSPARQL` that returns canned bindings and therefore never exercises RDF datatypes. Checked
against live Virtuoso, `attr:ranking` values are stored as `xsd:decimal`, so the tool's
`FILTER(?value = "64")` — a plain-string comparison — matched nothing and returned zero
qualifiers for data that is present. That single missing `STR()` was the actual cause of idx 68
("who was the reviewer ... ranking of 64" → `FIFA`) being unanswerable; the bnode/direct-literal
UNION alone would not have fixed it. `GetQualifierValue` already normalised this way, which is
why the inconsistency survived review. **Takeaway for future tool fixes in this repo: a
SPARQL-shape change is not verified until it has been run against a real endpoint.** The
regression test now asserts on the generated query text (`FILTER(STR(?value) = ...)`) rather than
on mock bindings.

## Related Docs
- [root-cause-tool-generalization-2026-05-14.md](root-cause-tool-generalization-2026-05-14.md) — this analysis continues that thread: several 2026-05-14/05-15 findings (qualifier discoverability via `GetAttributeDetails`, `QualifierFilter` adoption) recur here, now with root cause and a fix design instead of a symptom description
- [architecture-audit-2026-07-05.md](architecture-audit-2026-07-05.md) — the C3 tool-schema-token finding this analysis re-prioritises (see "Re-prioritisation" below)
- [sciqa-raw-sparql-keep-decision.md](sciqa-raw-sparql-keep-decision.md) — background on why `RunORKGSPARQL` stays in the tool set; §3.4/§4 findings here explain *why* agents still reach for it (comparison-anchoring bias, missing grouped-aggregation tools)
- [text-mode-tool-calls.md](text-mode-tool-calls.md) — turn-count/adoption dynamics documented here (§4.1-4.2 classifier/fewshot gating) compound with the text-tool-call parsing overhead for non-native-function-call models
- [../System/agent_system.md](../System/agent_system.md) — index into the 4 agent-system docs; §4.1 classifier gating and §4.2 fewshot retrieval both live in the KQAPro/SciQA agent docs below
- [../System/kqapro_agent.md](../System/kqapro_agent.md) — 10-type classification and journal model referenced by §1 (schema-role confusion) and §4.1 (classifier single point of failure)
- [../System/sciqa_agent.md](../System/sciqa_agent.md) — 8-type classification, ORKG predicate reference; referenced by §3.3-3.5 SciQA capability gaps
- [../Tasks/active/deferred-tool-loading.md](../Tasks/active/deferred-tool-loading.md) — re-prioritised (moved down the queue) by this analysis's evidence; see the dated note added there
- [../Tasks/active/locate-term-tool.md](../Tasks/active/locate-term-tool.md) — recommendation #1, highest priority
- [../Tasks/active/self-diagnosing-empty-results.md](../Tasks/active/self-diagnosing-empty-results.md) — recommendation #2
- [../Tasks/active/selectextreme-relation-filter.md](../Tasks/active/selectextreme-relation-filter.md) — recommendation #6
- [../Tasks/active/sciqa-grouped-aggregation-tools.md](../Tasks/active/sciqa-grouped-aggregation-tools.md) — recommendation #7
- [../Tasks/active/fewshot-bank-audit.md](../Tasks/active/fewshot-bank-audit.md) — recommendation #9

---

## Evidence base

- **KQAPro:** 600 traces (6 run variants over the same 100 questions) — `dev-caches-100q-2026-06-12`,
  `agent-improvements-100q-2026-06-12` (Hetzner), `rag-rerank-100q-2026-06-07/{dense_rerank,hybrid_rerank}`,
  `rag-compare-100q-2026-06-07/{dense,hybrid}` (local). Pooled accuracy 81.2%.
- **SciQA:** 300 traces — `norawsparql-sciqa-100q-2026-06-15`, `dev-rowfix-sciqa-100q-seed43-2026-06-13`,
  `dev-caches-100q-2026-06-12`. 73/100 in all three.
- **Gold coverage:** KQAPro gold programs joined to `db/datasets/kqapro/val.json` by exact question
  text (100/100 matched). SciQA gold recovered from the official Zenodo release (2,565 queries) plus
  the live CSV on Hetzner, since `db/datasets/SciQA/` is an empty gitignored placeholder in this checkout.
- **Root causes verified directly** against the raw KG (`db/datasets/kqapro/kb.nt`, live Virtuoso on
  Hetzner) and tool source, not inferred from agent or judge claims.
- **Validity note:** every commit between 2026-06-15 and 2026-07-25 is infrastructure only (chatkit
  extraction, TransientRetry, CI, KG adapter config resolution — see
  [transient-retry-and-chatkit-extraction.md](transient-retry-and-chatkit-extraction.md) and
  [kg-adapter-config-resolution.md](kg-adapter-config-resolution.md)). The June traces still describe
  current agent behaviour.

## Headline

**The dominant cause of wrong answers is not a missing tool.** Across the 14 KQAPro questions that
fail in 5 or 6 of 6 run variants: roughly **10 are adoption gaps**, **2 are genuine capability
gaps**, **3 are benchmark artifacts**, and **essentially none are misread tool results**. The
recurring shape is that the agent forms a wrong model of the question on the first step and never
revises it, because empty tool results carry no information that would force a revision.

---

## Summary of recommendations

| # | Change | Type | Addresses | Triage |
|---|---|---|---|---|
| 1 | `LocateTerm(entity_id, term)` → role (attribute / relation / qualifier), namespace, owning statement | new tool | 5 of 14 KQAPro failures | **DEFERRED** — see [locate-term-tool.md](../Tasks/active/locate-term-tool.md), highest priority |
| 2 | Self-diagnosing empty results: on the zero-result path only, report which conjunct failed | tool contract | the revision-failure enabler; raw-SPARQL escalation | **DEFERRED** — see [self-diagnosing-empty-results.md](../Tasks/active/self-diagnosing-empty-results.md) |
| 3 | Fix bnode assumption in `GetEdgeQualifiers` / `GetQualifierValue` | bug | idx 20 class | LANDED + VERIFIED (`8d519ca`) |
| 4 | Fix `FindNode` `limit=5`; return same-label candidates with discriminating attributes | bug | common-name collisions | LANDED + VERIFIED (`8d519ca`) |
| 5 | `CountEntities.or_conditions` accept relation-shaped branches; reject rather than drop unmatched | bug | `Or` at 0/3 | LANDED + VERIFIED (`8d519ca`) |
| 6 | `SelectExtreme` relation filter (avoid pasting 311 IDs) | ergonomics | idx 55/98 class | **DEFERRED** — see [selectextreme-relation-filter.md](../Tasks/active/selectextreme-relation-filter.md) |
| 7 | SciQA `FindTopByRelationCount` + `having`/extremum filter on aggregates | new tool | the 0% bucket | **DEFERRED** — see [sciqa-grouped-aggregation-tools.md](../Tasks/active/sciqa-grouped-aggregation-tools.md) |
| 8 | `FindFrequentValues` scope default + truncation warning | bug | graph-wide questions | LANDED + VERIFIED (`8d519ca`) |
| 9 | Audit few-shot bank against gold; decouple few-shot retrieval from the single classifier label | prompt architecture | §4.1 and §4.2 | **DEFERRED** — see [fewshot-bank-audit.md](../Tasks/active/fewshot-bank-audit.md) |
| 10 | Fix `GetJournalSummary` placeholder rendering | bug | §4.3 | LANDED + VERIFIED (`8d519ca`) |

Items 3, 4, 5, 8, 10 are small, code-verified, independently testable, and were judged safe to land
without a dedicated design doc. Items 1 and 2 are the ones expected to move accuracy the most.
Items 6, 7, 9 are larger design work.

---

## 1. Primary reasoning error: schema-role confusion

Five of fourteen KQAPro failures (idx 6, 68, 74, 86, 90) are the same error. The agent does not
know whether a term in the question denotes an **attribute**, a **relation**, or a **qualifier on a
statement**. It guesses, queries the wrong namespace, receives `{}`, and concludes the fact is
absent.

| idx | Question term | Agent's model | Reality (verified in `kb.nt`) |
|---|---|---|---|
| 68 | "reviewer" | attribute → `GetAttributeDetails("reviewer")` | `qual:review_score_by` on the `ranking=64` statement → "FIFA" |
| 74 | "postal code" | attribute of the target city | qualifier on the `(UWO, headquarters_location, London)` edge |
| 86 | "located in" | attribute branch in `or_conditions` | `prop:` relation only |
| 90 | "main subject" | attribute branch in `or_conditions` | `prop:` relation only (1,135 `property/main_subject` vs **0** `attribute/main_subject`) |
| 6 | "Barbara McLean" | entity node | literal value of `qual:winner`; no entity with that label exists |

For idx 86 and 90 the consequence is silent: the attribute branch of the OR matches nothing,
`CountEntities` drops it without complaint, and returns `"trusted": true`. Gold 53 → answered 1.
Gold 3 → answered 2.

For idx 6 the agent had the full 75-film candidate list in context at step 23 and never called
`QualifierFilter` over it. **The capability existed.** An earlier draft of this analysis wrongly
claimed reverse lookup by qualifier value was inexpressible; it is expressible once a candidate set
is in hand, which it was.

**Root enabler:** a wrong-namespace query returns an empty result rather than "that predicate
exists in `prop:`, not `attr:`". Nothing forces model revision.

This is the same gap recorded on 2026-05-15 in
[root-cause-tool-generalization-2026-05-14.md](root-cause-tool-generalization-2026-05-14.md)
(`GetAttributeDetails` does not expose that a qualifier exists) and never closed.

## 2. Secondary reasoning error: constraint dropped under pressure

idx 55 and 98. The agent builds a correct filter, hits an empty result from an unrelated cause, and
widens the candidate pool instead of repairing the filter.

idx 98 is the clearest instance. `GetRelationDetails(Q31, "film release region")` returned all
**311** Belgium-release films into context, untruncated. After a loop guard fired on raw SPARQL, the
agent fell back to `SelectExtreme(attribute_name="duration", concept="feature film", mode="min")`,
dropping the Belgium constraint entirely, and answered *Gulliver's Travels* — which has **zero**
`film_release_region` triples. Gold is *Alvin and the Chipmunks: Chipwrecked*.

`SelectExtreme` accepts `entity_ids` and `concept` together, so the call was expressible. But
composing it correctly would require the model to emit 311 entity IDs through its context. That is a
tool-ergonomics defect, not only a discipline problem.

## 3. Genuine capability gaps

### 3.1 Qualifier tools broken for direct-literal attributes (code-verified)

`GetEdgeQualifiers` (`kqapro_server.py:933-956`) and `GetQualifierValue` (`:1362-1373`) both
hardcode:

```sparql
ex:{id} attr:{p} ?bnode .
?bnode rdf:value ?value .
```

This requires the object to be a blank node bearing `rdf:value`. When the KG stores the value as a
plain literal, the join cannot match, because literals cannot be RDF subjects. idx 20 stores
`exploitation_visa_number "116454"` as a direct literal, with `qual:start_time "2006-11-20"` on the
reification bnode. Both tools report "no qualifiers found" for data that is present.

**Fix:** union in a `?stmt rdf:object ?value` pattern alongside the existing bnode-unwrap path.

### 3.2 `FindNode` caps exact-label matches at 5 (code-verified)

`_find_node_impl` (`kqapro_server.py:1709`) hardcodes `limit=5` (`:1755`) on the exact-match Qdrant
scroll. There are **13 entities labelled "Roger Moore"**; the one born 1943 with
`rogermoorephotography.com` was structurally unreachable. Vector similarity cannot discriminate among
byte-identical labels. Systemic for any common name.

### 3.3 SciQA: no count-per-group-then-argmax

"Which author contributed most to papers about X" fails 3/3 with three different wrong winners.
Gold is a nested `COUNT ... GROUP BY ... ORDER BY DESC LIMIT 1`. On the gold-matched subset:
`SUBSELECT` 0%, `GROUP BY` 0%, `HAVING` 0%, `AVG` 0%, `COUNT` 13%. This is the SciQA floor.

### 3.4 SciQA: Comparison-anchoring assumed universally

Gold metadata says `"no comparison used"` for **38 of 100** handcrafted questions; those are
graph-wide or title-regex queries over Papers. The tool suite and the strategy prompt ("~55% of
questions") both bias toward locating one named Comparison. "Average capacity for carbon fuel" (gold
2857.36) is a title-regex average over all papers matching `/fuel|CO2/`; the agent picked a
Comparison at relevance 0.58 and reported 5722.78 with full confidence. See
[sciqa-raw-sparql-keep-decision.md](sciqa-raw-sparql-keep-decision.md) for the prior A/B on why
`RunORKGSPARQL` still stays in the tool set despite this bias.

### 3.5 `FindFrequentValues` scope bug (not adoption)

`scope` defaults to `"comparisons"` (`sciqa_server.py:4849`), so graph-wide questions silently
exclude every paper not in a Featured Comparison. Silent `limit_subjects=5000` truncation on top. All
three runs called the tool correctly by name and all three produced the same wrong ranking. (This is
recommendation #8, FIX NOW this session.)

## 4. Architectural causes behind the adoption gaps

### 4.1 The classifier is a single point of failure

A single question-type label gates **both** the strategy text and few-shot retrieval (see
[../System/kqapro_agent.md](../System/kqapro_agent.md) and
[../System/sciqa_agent.md](../System/sciqa_agent.md) for the classification schemes).

- idx 68 tagged `QueryAttr` instead of `QueryAttrQualifier`; idx 87 tagged `QueryName`. Both
  misclassifications stripped the "the answer is a qualifier value, do not collapse to the entity"
  guardrail that appears verbatim in correctly-classified traces.
- SciQA "most commonly modeled energy sector" tagged `Superlative`. The few-shot containing the
  **exact resource, exact predicate path, and exact gold answer** ("Heat sector, 8") lives in the
  `Count` bank and was never shown.

### 4.2 The prompt library contradicts the tool contracts

`AggregateComparisonValues`'s docstring (`sciqa_server.py:3756-3760`) states that
`group_by_path="^P31,P29"` is "the correct time axis for 'per year' / 'in N-year intervals'
questions". `prompts.py:1093` ships a worked example for that exact question class teaching
`group_by_path="P37581,P43138,P43139"` (the scenario's internal goal year). Every run followed the
few-shot. The capability added in commit `6e30da5` (per
[root-cause-tool-generalization-2026-05-14.md](root-cause-tool-generalization-2026-05-14.md))
works; the competing wrong example was never retracted.

Separately, the energy-sectors few-shot names `R153801` **by ID as a known trap** ("if you only
`AggregateComparisonValues` over R153801 you get the WRONG answer"), and the agent used R153801
anyway, in the same context window as the warning, in 2 of 3 runs.

Pattern: fixes are being written as per-question worked examples, which are then withheld by
classifier routing, contradicted by stale siblings, or ignored outright. This is recommendation #9,
deferred — see [fewshot-bank-audit.md](../Tasks/active/fewshot-bank-audit.md).

### 4.3 Journal rendering corrupts RunSPARQL-sourced facts

`GetJournalSummary`'s formatter (`kqapro_server.py:1630-1674`) renders SPARQL-derived entries as
literal `"✓ results: ?"` and `"• ? (?) ->[?]-> ?"` placeholders, because `.get(key, "?")` defaults do
not match the key shape RunSPARQL's journal-write path produces. It flipped no verdicts in this set,
but the system prompt repeatedly directs the agent to trust the journal when composing its final
answer. (Recommendation #10, FIX NOW this session.)

## 5. Benchmark artifacts — do not chase

Roughly 3 of 14 KQAPro failures are not agent errors:

- **idx 71:** question asks date of **birth**; the gold program's final step is
  `QueryAttr("date of death")`. The agent retrieved both correctly and answered as worded.
- **idx 80:** question says "16th October"; the gold parameter is `2007-10-15`. The agent found
  "Rome" and then suppressed it over the one-day mismatch.
- **idx 69:** correct retrieval of `1982-01-01`, verbalised as "in 1982", judged wrong on format.

SciQA:
- Several golds are tinyurls whose underlying SPARQL binding is empty **at the source**.
- The 5-year-interval gold is pinned to `R153801` while the question names `R153799`'s title (ORKG
  resource drift; the two have byte-identical schemas).
- One question's text is literally `"all comparisons"`, a metadata sentinel leaked into the question
  field. Unanswerable as posed; 0% across all three runs.

## 6. Corrections to prior beliefs

- **Inverse property paths are not a gold requirement.** The 2,565-query SciQA gold corpus contains
  **zero** true inverse paths, zero transitive paths, zero `MINUS`, one `NOT EXISTS`. ORKG gold
  reifies multi-hop traversal through explicit intermediate variables. The traversal need is real
  (contribution → owning paper), but it is a reverse-direction join, not path syntax. Commit
  `6e30da5`'s inverse-hop support is the right mechanism; the problem is that the few-shot teaches a
  different axis.
- **Year/interval binning is already covered** by `group_bucket_size` / `group_bucket_start`.
- **Negation is not a real gap** in this benchmark.
- **`QualifierFilter` is not structurally broken.** Its query shape was reconstructed against live
  Virtuoso and returns correctly. Its 29/29 empty rate is an argument-supply failure with no
  differential feedback.

---

## Re-prioritisation: `deferred-tool-loading` moves down the queue

**`Tasks/active/deferred-tool-loading.md` (audit finding C3, see
[architecture-audit-2026-07-05.md](architecture-audit-2026-07-05.md)) should move down the queue.**
Schema overhead is 39.7% of per-call tokens, but the dominant driver is turns × transcript at a
measured **153.6:1 prompt-to-completion ratio**, so eliminating wasted turns pays more than shrinking
schemas. The risk is now measurable rather than hypothetical: the tools built for these exact failure
patterns already have near-zero adoption, and `FindFrequentValues` appears in **0% of correct
traces**. Hiding their schemas behind a `load_tools` call worsens an adoption problem in exchange for
a token win that items 1, 2, and 6 deliver anyway by cutting turn counts. A dated note recording this
has been added directly to `deferred-tool-loading.md`.

## Decision

Land items 3, 4, 5, 8, 10 now (concurrent work this session, verify independently before treating as
shipped). Track items 1, 2, 6, 7, 9 as separate `Tasks/active/` design docs for a dedicated session —
each requires either a benchmark A/B or enough new-tool design work that it doesn't belong in an
opportunistic bug-fix pass. De-prioritise `deferred-tool-loading.md` given the adoption evidence
above; do not resume it until items 1/2/6 have shipped and been measured.

---

## Addendum 2026-07-25: second fix batch (commit `e1ce3bf`) and a correction

Items 3, 4, 5, 8, 10 above landed in commit `8d519ca` ("Reasoning-error analysis + five verified
tool fixes"), concurrently with this ADR being written, as flagged throughout. A second,
independent batch of harness and generator defects — found while producing the trace evidence this
ADR is built on, not predicted by it — landed afterward in commit `e1ce3bf` ("Harness and generator
fixes: trace joins, resume, escaping, fewshot routing"). Two of these are worth recording because
they affect how any future trace analysis on this codebase should be trusted or conducted.

### The `tool_traces/question_NNN.json` ↔ `results.json` join did not hold under concurrency

`_save_tool_traces` (`ama_kbqa/benchmark_agents.py`, ~line 1215) named each trace file by the
result's *position* in the `results` list via `enumerate(results)`. `results` is populated by
`run_benchmark_for_model_agent_parallel` in **completion order**, not launch order — with variable
per-question latency the append order can be e.g. `[3, 0, 4, 2, 1, 5, 8, 6, ...]`. So
`tool_traces/question_007.json` could silently hold the trace for a different question than row 7
of `results.json`. Fixed by keying trace filenames on `r.question_id` directly (now documented
in-line at `benchmark_agents.py:1221-1225`).

**Implication:** any trace-level analysis of a *concurrent* benchmark run performed before this
commit — including earlier passes feeding this same investigation — could not assume
`tool_traces/question_NNN.json` and `results.json` row `NNN` describe the same question, and had to
be cross-checked or worked around. Analysis of *serial* runs is unaffected (completion order equals
launch order). Trust the join only for runs captured at or after `e1ce3bf`.

### Empty `benchmark_results/2026-*` dirs and `logs/*.log` noise were test pollution, not failed runs

23 empty dated directories (`2026-07-18-1` through `-18`, `2026-07-24-1` through `-5`) were initially
read as evidence of dead/crashed benchmark runs during this investigation. They were not: root cause
was `tests/test_seed_and_progress.py::test_main_exports_seed_env` calling `main()` with no
`--output-dir`, while `ama_kbqa/benchmark_agents.py:2139` unconditionally ran `output_dir.mkdir(...)`
— every test invocation left an empty dated directory in the real `benchmark_results/`. Separately,
the three server modules (`ama_kbqa/server/{kqapro,sciqa,orchestrator}_server.py`) construct a
loguru file sink at **import time**, so any test that imports them appends fixture strings (`"boom"`,
`"bad"`, `"localhost:59999"`) into the real production log files. Both are fixed: the test now uses
`tmp_path`, and a new `tests/conftest.py` sets an `AMA_KBQA_LOG_DIR` override the server modules
respect. Verified: a full suite run now leaves `logs/` byte-identical and creates no new directories.

**Implication:** do not treat empty `benchmark_results/` directories or unfamiliar strings in
`logs/*.log` as evidence of a failed or anomalous run without first checking whether the test suite
ran against production paths at some point — it did, for every run on `dev` before this commit.

### Correction: `question_id: 0` is not a live defect

An earlier point in this investigation reported `question_id: 0` on every row of certain
`results.json` files as a currently-live bug. **It is not.** That bug was already fixed on `dev`,
before this investigation started, by commits `eacc4c1` (2026-06-14, "Make `--seed` control agent
LLM sampling; fix benchmark progress reporting") and `85c3b0f` (2026-07-18, "Fix review findings on
reconciliation port + add reconciliation ADR"). The all-zero-`question_id` files observed came from
runs dated 2026-06-07, 2026-06-12, and 2026-06-13 — all of which predate the fix. The 2026-06-15 run
onward is unaffected. If this document or any `Tasks/active/` doc is read as asserting all-zero
`question_id` is a current defect, that reading is wrong; this note supersedes it.

### Other `e1ce3bf` fixes (brief)

- `is_run_completed()` (`ama_kbqa/benchmark_agents.py:108`) now requires `is_complete: true` in
  `summary.json` rather than treating the file's mere existence as "done." `summary.json` is written
  after *every* question with `is_complete: false` until the final write, so `--resume` previously
  treated a run that crashed at 5/100 as finished and skipped it forever.
- `FindByAttribute`'s URI-literal branch spliced raw values into a `<...>` fragment guarded only by a
  runtime `!CONTAINS` filter; since the fragment is static query text, any whitespace-bearing value
  (e.g. every ISNI code) failed at SPARQL parse time before the guard could run. The URI branch is
  now gated in Python. The literal branch's escaping, previously handling only double quotes, now
  also escapes backslash, LF, and CR.
- The fewshot exporter (`ama_kbqa/fewshot_generator.py`) hardcoded the KQAPro output directory for
  every agent and did not sanitise qtype labels, so SciQA's newline-joined compound types (e.g.
  `"Factoid\nSuperlative"`) produced filenames with embedded newlines, and shared `_general.json` /
  `_tool_tips.json` sinks merged SciQA insights into the prompt injected for KQAPro. The exporter is
  now agent-aware with sanitised filenames; three mis-filed files moved to
  `db/datasets/sciqa/fewshot-examples/`.
  - **Open caveat:** `SciQAAgent` still reads few-shot examples from the static in-code
    `FEWSHOT_EXAMPLES` dict in `ama_kbqa/agents/sciqa_agent/prompts.py` (~line 877), not from the
    files this exporter now correctly routes. The files are written but nothing consumes them yet.
    This is the intended scope of this fix — the goal was stopping SciQA-KQAPro prompt contamination,
    not wiring SciQA's runtime prompt to the exported bank. Wiring SciQA to consume its own exported
    fewshots is separate follow-on work, not yet tracked as a `Tasks/active/` doc.

None of `e1ce3bf`'s fixes correspond to an existing `Tasks/active/` PRD — they were harness/generator
bugs surfaced opportunistically during analysis, not planned feature work, so nothing in
`Tasks/active/` moves to archive as a result of this commit. Test suite: 479 passing (441 → 479 in
this commit; commit `8d519ca` had already brought it from an earlier baseline to 441).

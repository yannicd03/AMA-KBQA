# SciQAAgent

KBQA agent for Open Research Knowledge Graph (ORKG) scientific Q&A (~160MB,
468 Q&A pairs). Inherits from `BaseKBQAAgent`. Split out of the former
monolithic `agent_system.md` (2026-07-18) — see [Agent Framework](agent_framework.md)
for shared mechanics not repeated here.

## Related Docs
- [Agent System (index)](agent_system.md) — map of all agent-system docs
- [Agent Framework](agent_framework.md) — `BaseKBQAAgent` shared mechanics
- [KQAProAgent](kqapro_agent.md) — the sibling agent for general-knowledge QA
- [Project Architecture](project_architecture.md) — full tool reference and directory tree
- [Decisions/sciqa-raw-sparql-keep-decision.md](../Decisions/sciqa-raw-sparql-keep-decision.md) — A/B evidence behind the raw-SPARQL denylist gate
- [Decisions/sciqa-cross-resource-aggregation.md](../Decisions/sciqa-cross-resource-aggregation.md), [Decisions/sciqa-node-type-filter.md](../Decisions/sciqa-node-type-filter.md), [Decisions/multi-label-fewshot-tolerance.md](../Decisions/multi-label-fewshot-tolerance.md) — tool/prompt ADRs referenced below

---

## File Organization

- `ama_kbqa/agents/sciqa_agent/agent.py` (~190 lines) — `SciQAAgent` class, ORKG-specific overrides
- `ama_kbqa/agents/sciqa_agent/prompts.py` (~890+ lines) — ORKG-specific prompts, 8 strategies with fewshots, predicate dictionary, loop recovery entries

## Agent Lifecycle

Same 4-phase pattern as KQAProAgent (see [kqapro_agent.md](kqapro_agent.md)):

1. **Initialization** — load LLM clients, connect to `sciqa_server.py`
2. **Pre-Agent Hook** — classify question (8 types), extract entities, load type-specific strategy
3. **Main Agent Loop** — call ORKG tools (28 registered tools), track progress; `max_tool_calls` = 25
4. **Post-Agent Hook** — synthesize answer from journal (always exits through synthesis — see [Synthesis Funnel](agent_framework.md#synthesis-funnel-always-exit-through-synthesis))

## Multi-Label Classifier-Output Tolerance

`SciQAAgent._classify_question()` overrides the base class with a two-pass fewshot lookup:

1. **Direct match:** `FEWSHOT_EXAMPLES.get(qtype, "")`.
2. **Split-and-merge:** if empty, split `qtype` on `r"[\n,/+|;]+"` (handles compound labels like `"Factoid\nSuperlative"` that the classifier reproduces from gold annotations), case-normalize, concatenate fewshots from every matching label. `chosen_label` = first match.

Prevents silent empty-fewshot for compound classifier outputs — e.g. the agent still receives Superlative fewshots with `FindFrequentValues` traces even when the LLM outputs `"Factoid\nSuperlative"`. See `Decisions/multi-label-fewshot-tolerance.md`.

## Prompt Architecture (Type-Specific Strategy Loading)

```
[0] SYSTEM: SYSTEM_PROMPT (lean — rules, schema, tools, predicates)
[1] USER: Original question
[2] USER: Analysis Context (post-classification)
    ├── Question Type
    ├── Extracted Entities/Relations
    ├── QTYPE_STRATEGIES[detected_type]
    └── FEWSHOT_EXAMPLES[detected_type]
```

**`SYSTEM_PROMPT`** (~165 lines): critical rules, ORKG schema/prefixes, 5-tier tool catalog, execution strategy, predicate reference dictionary.

**Critical Rules:** (1) no hallucination — verify via tools, one-hop logical inference exception; (2) schema compliance via `GetResourceSummary` on empty results; (3) `ManageJournal` state tracking; (4) pivot on repeated search failure; (5) always follow `FindResource` with a detail/summary/relation call; (6) verify all conditions with `VerifyNumericCondition`; (7) **scope-before-query** — for count/avg/sum/min/max/frequency/top questions, decide single-comparison vs multi-comparison vs global/papers scope before choosing tools; (8) ORKG booleans use `"T"/"t"` for True/present, `"F"/"f"` for False/absent.

**Evidence-First Block:** short traces are acceptable when they establish the right scope, predicate, and answer-producing tool result. For aggregation/count/superlative: inspect schema, then try `AggregateComparisonValues`/`FindFrequentValues` before raw `RunORKGSPARQL`. For multi-filter row questions: `QueryComparisonRows` before raw SPARQL. For paper-metadata frequencies: `FindFrequentValues(scope="papers", value_source="subject")`.

**Deep-dive prompt hardening (commit `7397cc3`, 2026-06-15)** — added after trace analysis of 27 failures in a 100-question cached run surfaced four fixable root causes (see also the server-side fixes below):
- Scope/anchoring rules and P31 direction guidance (which side of `orkgp:P31` is Paper vs Contribution).
- Category-filter rule for narrowing nested rows by label/component.
- **Paired-grouping recipe** — when a grouping key and an intermediate/nested-row path share the same relation path, group by the intermediate node instead of producing a naive cross-join of every label against every value.
- Author/recency ranking recipes (SPARQL patterns for "most recent" / "most cited" style questions).
- **Final-answer contract**: enumerate all groups including rollups (don't drop "all sources"-style rollup rows), preserve full numeric precision (don't round), never invent or summarize list items — return exactly what the tool returned.

## Question Type Classification (8 types)

| Type | Description | Strategy |
|------|-------------|----------|
| **Factoid** | Direct fact lookup | `GetResourceSummary` for exploration, comparison-based factoid guidance, SPARQL domain data tip |
| **Count** | "How many..." | Scope first; `AggregateComparisonValues` for named Comparisons, `FindFrequentValues` for global/paper counts, `DiagnoseComparisonAggregation` when population is unclear, raw SPARQL only for set-difference/unsupported shapes |
| **List** | "Which papers..." | Comparison-based list pattern, `GetComparisonContributions`, multi-hop lists |
| **Boolean** | "Is...", "Does..." | ASK SPARQL for comparison data, `VerifyNumericCondition` for numeric conditions |
| **Comparison** | Compare entities | `CompareResources` for batch comparison, full nested-value navigation |
| **Superlative** | "highest", "most popular X overall" | Decide SCOPE first: (A) single-comparison → `InspectComparisonSchema` → `AggregateComparisonValues`; (B) multi-comparison → both with `comparison_ids`; (C) global/cross-graph → `FindFrequentValues`; (D) paper metadata → `FindFrequentValues(scope="papers", value_source="subject")`. Locking onto a single Comparison for a global-scope question is the highest-leverage failure mode. |
| **Aggregation** | SUM, AVG, total, frequency | Same SCOPE decision as Superlative; prefer high-level tools over `RunORKGSPARQL` for AVG/SUM/MIN/MAX/COUNT/MODE_TOP; use `intermediate_predicate`/`value_predicates`/`DiagnoseComparisonAggregation`/`QueryComparisonRows` before raw SPARQL |
| **General** | Complex/other | `GetResourceSummary` exploration, comparison mention fallback, SPARQL domain data tip |

## Loop Detection (SciQA exercises all 7 layers)

Layers 1-4 and 6-7 as in [kqapro_agent.md](kqapro_agent.md#loop-detection-kqapro-exercises-layers-1-4-and-6-7-layer-5-is-sciqa-specific), plus:

5. **`RunORKGSPARQL` Cap** — 10 calls (`sparql_cap` in domain_settings). Message: "RunORKGSPARQL called N times (cap: 10). Use GetComparisonContributions or GetResourceSummary instead." SciQA-only (KQAPro has no equivalent raw-SPARQL cap tool).

**No-Progress intervention** is a 6-branch decision tree (`NO_PROGRESS_TEMPLATE`): (1) aggregation/count/superlative → state scope + use `AggregateComparisonValues`/`FindFrequentValues`; (2) path found but denominator unclear → `DiagnoseComparisonAggregation`; (3) multi-filter row question → `QueryComparisonRows`; (4) bad anchor → switch resource type or `FindByPredicateValue`; (5) missing predicate → `GetResourceSummary`; (6) scoped attempt + schema discovery found nothing → synthesize from current evidence.

## Env-Gated Tool Denylist: `AMA_SCIQA_DISABLE_RAW_SPARQL`

`SciQAAgent._get_denied_tool_names()` (overriding the base no-op) gates `RunORKGSPARQL` off when `AMA_SCIQA_DISABLE_RAW_SPARQL` is set to a truthy value (`1`/`true`/`yes`/`on`; off/absent by default). Applied via the base class's denylist gate (see [Tool Gating](agent_framework.md#tool-gating-qtype-filter--denylist)) after the qtype filter, so it can suppress the tool even in strategies that would otherwise allow it.

**Why it exists, and why it's *not* a removal:** `RunORKGSPARQL` runs ~65% unproductive in traces, making it an attractive removal target. A seed-42/n=100 A/B (gate OFF vs baseline ON) scored 73/100 vs 77/100 — 7 questions broke, 2 gained, net −5, and token usage did **not** drop (12.10M OFF vs 11.87M ON). The "forcing dedicated tools improves accuracy" hypothesis was rejected; some questions (e.g. global-recency lookups with no dedicated tool path) are genuinely unrecoverable without raw SPARQL. **Decision:** keep the tool in the default tool set, keep the env gate as a permanent, deliberately-idle A/B lever for future re-tests once tool-gap backfills land. See `Decisions/sciqa-raw-sparql-keep-decision.md`.

## SciQA MCP Tools

`sciqa_server.py` registers **28 tools**: 24 user-visible KB tools (Tiers 1-5) + 2 state-management tools + 1 LLM-hidden journal-snapshot tool (`GetJournalStateJSON`) + `VerifyNumericCondition` (Tier 5). See [Project Architecture — §2.1 SciQA MCP Server](project_architecture.md) for the full per-tool reference — not duplicated here to avoid drift between two copies of the same tool list. Server-side fixes from the 2026-06-15 deep-dive (commit `7397cc3`), grounded directly in `sciqa_server.py`:

- **`QueryComparisonRows`** — accepts dict-form filters (previously pydantic-list-only, which cost the agent an extra turn on some questions); supports comma-separated paths with leading `^` inverse hops; matches a single predicate up to two anonymous hops below the contribution by default (comparison cells often live on sub-resources, not the contribution itself); returns each row's paper id/label via the inverse `^P31` hop.
- **`AggregateComparisonValues`** — when `group_by_path`/`group_by_predicate` and an intermediate/nested-row path share the same relation path, the group is now bound to the intermediate node instead of cross-joining every label against every value (previously produced the identical aggregate for every group on affected questions); warns when a grouped numeric aggregate looks degenerate (e.g. one group absorbing everything).
- **`RunORKGSPARQL`** — lints unanchored `?x compareContribution` patterns that would silently scan the whole KG rather than a specific Comparison's contributions, and warns instead of returning a globally-scoped count as if it were locally scoped.

## ORKG Predicate Reference

**Core Navigation Predicates:**

| Predicate | URI | Description | Pattern |
|-----------|-----|-------------|---------|
| P0 | `orkgp:P0` | addresses (problem) | Paper/Contribution → Problem |
| P1 | `orkgp:P1` | yields (result) | Contribution → Result |
| P2 | `orkgp:P2` | employs (method) | Contribution → Method |
| P6 / P27 | `orkgp:P6` / `orkgp:P27` | author | Paper → Author |
| P7 | `orkgp:P7` | affiliation | Author → Organization |
| P10 / P26 | `orkgp:P10` / `orkgp:P26` | DOI | Paper → DOI string |
| P29 | `orkgp:P29` | publication year | Paper → Year |
| P30 | `orkgp:P30` | research field | Paper → ResearchField |
| P31 | `orkgp:P31` | has contribution | Paper → Contribution (**critical path**) |
| P32 | `orkgp:P32` | research problem | Paper → Problem |

**Key navigation pattern (~70% of questions):** `Paper --P31--> Contribution --domain_predicate--> Value`

**Domain-specific predicates (via Contributions):** Energy (P43133 installed capacity, P43135 energy sources, P43247/P43248 upper/lower limit), Chemistry (P35147 Bisphenol A analogue, P35194 SAME_AS), Benchmarks/NLP (P41923 amount of questions, P15585 has benchmark), Biology (P37458 major anion type, P37586 study type), Comparison (P5038 Aggregation, P5039 tool capabilities).

## Batch Processing

See [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) for the unified batch CLI. `llm_judge` postprocessing applies a deterministic numeric-equivalence guard: if the predicted answer contains the gold's numeric value within a small decimal tolerance, harmless rounding/formatting differences don't become false negatives.

```bash
python -m ama_kbqa.benchmark_agents --agents sciqa --n-questions 10 --seed 42 --dataset handcrafted
python -m ama_kbqa.benchmark_agents --agents sciqa --n-questions 50 --dataset auto --postprocessing llm_judge
```

**Dataset (CSV) columns:** `Paraphrase`, `Result` (gold), `Machine-readable query` (SPARQL), `Q Content` (question type), `Research field`.

# AMA KBQA — Documentation Index

**Project:** Multi-agent Knowledge Base Question Answering system over heterogeneous RDF/SPARQL knowledge graphs.  
**Research venue:** SEMANTiCS 2026 Posters & Demos track (`paper/main.tex`).  
**Quick orientation:** See [System/project_architecture.md](System/project_architecture.md) first.

---

## System

| Document | Description |
|----------|-------------|
| [System/project_architecture.md](System/project_architecture.md) | Start here. Full structure, tech stack, MCP servers, CLI, frontend, batch pipeline, config system |
| [System/agent_system.md](System/agent_system.md) | Agent lifecycle, question classification, loop detection (6 layers), journal/scratchpad data model |
| [System/research_corpus.md](System/research_corpus.md) | Paper (`paper/`), dataset docs (`docs/`), and how research artifacts relate to code |
| [System/wikikgqa_agent.md](System/wikikgqa_agent.md) | WikiKGQA subsystem reference: `WikidataAgent`, `wikidata_server.py` MCP tools, `MentionSparqlGenerator`/`AgentSparqlGenerator`, tool-call budget mechanism, R1-R10 answer conventions |

> `System/database_schema.md` — referenced in older READMEs but file is absent; Qdrant/Virtuoso schema is documented inline in `System/project_architecture.md` and `System/agent_system.md`.

---

## SOP

| Document | Description |
|----------|-------------|
| [SOP/running_batch_processing.md](SOP/running_batch_processing.md) | Run single-model or multi-model benchmarks; LLM judge; fewshot generation; CSV export; resume |
| [SOP/adding_new_kqapro_tools.md](SOP/adding_new_kqapro_tools.md) | Checklist for adding KQAPro MCP tools that produce answer values (journal write invariants) |
| [SOP/hetzner_deployment.md](SOP/hetzner_deployment.md) | Hetzner VPS deployment runbook: 3-service compose stack, Qdrant port, frontend container, RDF bootstrap, SSH tunnel |
| [SOP/migrate_add_bm25.md](SOP/migrate_add_bm25.md) | Runbook for `db/migrate_add_bm25.py`: adds BM25 sparse index to existing Qdrant collections (dry-run, then --yes); enable hybrid in config afterward |

> `SOP/database_setup.md` and `SOP/changing_llm_provider.md` — referenced in earlier docs but files are absent; Virtuoso/Qdrant setup is in the root `README.md`; LLM provider config is in `docs/guides/dataset_integration.md` and `config.toml`.

---

## Decisions

| Document | Description |
|----------|-------------|
| [Decisions/hybrid-retrieval-architecture.md](Decisions/hybrid-retrieval-architecture.md) | Hybrid BM25+dense retrieval via Qdrant Query API (RRF/DBSF), cross-encoder reranker, `[retrieval]` config section, AMA_RETRIEVAL_* env overlay, Settings page persistence |
| [Decisions/abstract-operation-contract.md](Decisions/abstract-operation-contract.md) | Four linked decisions from the 2026-06-06 drift audit: KG-native property addressing, per-server envelopes, two-level count contract (11 req + 3 opt), KG-flavored tool names preserved with binding layer comparability |
| [Decisions/kqapro-tool-surface-expansion.md](Decisions/kqapro-tool-surface-expansion.md) | Why CountEntities, SelectExtreme, VerifyFact were added + FilterEntities extensions (closes 42% coverage gap) |
| [Decisions/text-mode-tool-calls.md](Decisions/text-mode-tool-calls.md) | Why and how a client-side `<tool_call>` parser was built for models that can't emit native function calls (minimax-m2.7) |
| [Decisions/llm-judge-concept-extraction-rubric.md](Decisions/llm-judge-concept-extraction-rubric.md) | LLM judge rubric rewrite: concept-extraction first, per-type rules, hard-negatives for tool-call fragments and max-iteration errors (+~2 pp, removes verbose-answer bias) |
| [Decisions/transitive-concept-default-false.md](Decisions/transitive-concept-default-false.md) | Why `transitive_concept` default stays False: n=100 benchmark showed -28 pp Count regression on flat concepts; retry-on-empty guidance keeps hierarchy cases recoverable |
| [Decisions/qualifier-and-format-prompt-hardening.md](Decisions/qualifier-and-format-prompt-hardening.md) | 5-prompt-fix bundle + iter-cap 25→40 targeting n=100 → 0.86–0.90: GetNodeSummary gate, QueryAttr attribute-first fallback, QueryAttrQualifier direction rule + QualifierFilter, QueryRelation format + passive-voice trap. Post-merge: 0.810/0.790 confirmed with v4-pro judge + RULE 0 retry |
| [Decisions/get-qualifier-value-tool.md](Decisions/get-qualifier-value-tool.md) | 26th KQAPro tool: direct single-qualifier projection vs full-dict tools (GetEdgeQualifiers / GetQualifiersByPredicate). Auto-detects relation/attribute and forward/backward direction. Targets QueryAttrQualifier / QueryRelationQualifier accuracy gap |
| [Decisions/sciqa-aggregation-and-coauthor-tools.md](Decisions/sciqa-aggregation-and-coauthor-tools.md) | SciQA tools 19/20: `AggregateComparisonValues` (SPARQL+Python aggregation over Comparison contributions, closes 9/20 SciQA aggregation questions) and `FindCoAuthors` (co-author enumeration, closes Q2). Motivated by n=20 audit + cross-model failure analysis. Deployment to Hetzner pending. |
| [Decisions/qualifier-fewshot-hardening.md](Decisions/qualifier-fewshot-hardening.md) | Fewshot trace rewrite for QueryRelationQualifier (3 + 1 new) and QueryAttrQualifier (1 + 2 new) to demonstrate `GetQualifierValue` directly; closes "do as I show not as I say" gap where model copied old GetEdgeQualifiers traces over prompt-level instruction |
| [Decisions/truncated-tool-call-retry.md](Decisions/truncated-tool-call-retry.md) | Agent loop retry for minimax-m2.7 mid-`<tool_call>` truncation (opener present, block unparseable); `has_truncated_tool_call` classifier; targets the single largest failure mode in n=100 audit (30% of wrongs) |
| [Decisions/classifier-think-prefix-fix.md](Decisions/classifier-think-prefix-fix.md) | `_extract_json_object` static helper on BaseKBQAAgent: strips `<think>` blocks + markdown fences + brace-balanced fallback; classifier max_tokens 300→1500; fixes minimax-m2.7 silently routing all questions to "Query" |
| [Decisions/trace-inspector-frontend-architecture.md](Decisions/trace-inspector-frontend-architecture.md) | 6 ADRs for the Trace Inspector + Graph View feature: stay on Streamlit, ContextVar nesting, mutating-tool-only snapshots, shared recorder for Orchestrator delegation, vis-network via CDN, iteration event vs span. **Status update (v2):** live-updates gap resolved — see next entry. |
| [Decisions/live-trace-and-chat-unification.md](Decisions/live-trace-and-chat-unification.md) | v2 ADR: listener-based TraceRecorder observer, worker-thread asyncio loop + queue + fragment polling, `_SilentPlaceholder`, chat-page tab panel unification, Simplified view toggle, inline-SVG lifecycle figure. v3 update: SVG redesigned to match `fig:agent_flow` (new node IDs), span-tree click-to-select, Graph View removed. |
| [Decisions/fewshot-generator-enrichment.md](Decisions/fewshot-generator-enrichment.md) | Generator driven by `[fewshot_generator]` config section; `load_tool_catalog` injects live MCP tool catalog; `include_full_conversation=true` default; post-hoc runner `run_fewshot_generator.py` with hard guard against live-dir writes |
| [Decisions/kit-outage-watchdog.md](Decisions/kit-outage-watchdog.md) | Structured `httpx.Timeout` (connect/read/write/pool) replacing bulk `timeout=60` in `_create_client` (surfaces socket hangs); 5-consecutive-infra-failure abort in `run_benchmark_for_model_agent` with `[FATAL]` log and partial-output preservation |
| [Decisions/count-entities-not-conditions.md](Decisions/count-entities-not-conditions.md) | `not_conditions` parameter for `CountEntities`: `FILTER NOT EXISTS` via `_attr_condition_sparql` helper; "lacks attribute = KEPT" semantics; motivated by seed=43 Count failures on negation questions (both models guessed raw SPARQL, got wrong count) |
| [Decisions/sciqa-node-type-filter.md](Decisions/sciqa-node-type-filter.md) | `node_type_filter` for `FindResource`: over-fetches 4×top_n then prunes by `rdf:type orkgc:<class>`; fixes wrong-resource-type silently corrupting `AggregateComparisonValues` (Paper returned instead of Comparison); Aggregation fewshot updated |
| [Decisions/sciqa-cross-resource-aggregation.md](Decisions/sciqa-cross-resource-aggregation.md) | New tool `FindFrequentValues` (tool 21) + `comparison_ids` extension to `AggregateComparisonValues`; closes 12-failure global-scope superlative/aggregation class; SCOPE decision tree in Superlative/Aggregation strategies; 25-call hard-stop in NO_PROGRESS_TEMPLATE |
| [Decisions/multi-label-fewshot-tolerance.md](Decisions/multi-label-fewshot-tolerance.md) | Split-and-merge fewshot lookup in `SciQAAgent._classify_question()`: handles compound classifier outputs like `"Factoid\nSuperlative"` that silently yielded empty fewshots and hid `FindFrequentValues` traces from the agent |
| [Decisions/root-cause-tool-generalization-2026-05-14.md](Decisions/root-cause-tool-generalization-2026-05-14.md) | Root-cause fixes from the seed-43 trace audit: answer cleanup, branch-local `CountUnion` relations, qualifier aliases, `QueryComparisonRows`, multi-predicate aggregation, and subject-scope frequencies |
| [Decisions/multiturn-direct-agent-conversation.md](Decisions/multiturn-direct-agent-conversation.md) | Multiturn conversation for directly-selected sub-agents: persistent agent instance + `reset(keep_history=True)`; skip pre-agent hook on follow-ups; fast-path answer recording; asyncio event-loop safety; Orchestrator stays stateless |
| [Decisions/orchestrator-evidence-based-routing.md](Decisions/orchestrator-evidence-based-routing.md) | Orchestrator routing reworked from collapsed heuristic verdict (avg_confidence > 0.7) to evidence-based two-step LLM routing: tool returns raw JSON (terms_probed/matched/avg_score/labels); LLM calls `select_agent(agent, reason)`; `route_reason` on classify span; degraded paths no longer silently KQAPro |
| [Decisions/wikikgqa-2026-adaptation.md](Decisions/wikikgqa-2026-adaptation.md) | ADR for WikiKGQA 2026 challenge adaptation: SPARQL-generate→execute head, no Wikidata embedding (live SPARQL as index), challenge endpoint backend, English with-mentions initial target, new `wikidata/` backend + `ama_kbqa/wikikgqa/` module |
| [Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md](Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md) | Tool-call budget (20) replaces wall-clock (280s) as the binding limit on the agentic generator, fixing empty-output cancellation on over-exploration; per-tool SPARQL timeout with steering message; live `[tool call N/20]` budget counter; shared-framework retry for transient KIT proxy errors + empty-response guard |
| [Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md](Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md) | Synthesis context default reverted to minimal (journal-only): seed-42 rand-50 A/B found full-context 0.781 vs minimal 0.807 F1 (within noise, cheaper); full stays opt-in via `WIKIKGQA_FULL_SYNTHESIS=1` / `--full-synthesis` |
| [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](Decisions/wikikgqa-commit-time-recovery-2026-07-03.md) | `strip_sparql` empty-on-no-match fix + non-empty-prior fallback (0/462 gold answers empty) at commit time; ASK self-consistency voting (`--ask-votes`); `run_manifest.json` per run; open framework-level paths that still discard validated queries |

---

## Tasks

| Document | Status | Description |
|----------|--------|-------------|
| [Tasks/benchmark_persistence_refactor.md](Tasks/benchmark_persistence_refactor.md) | Planned | Decouple benchmark runs from Streamlit session; incremental disk writes + job_id reconnect (see Orca's `batch_queue.py` for the reference pattern) |
| [Tasks/archive/generic-framework-implementation.md](Tasks/archive/generic-framework-implementation.md) | Archived | Generic KBQA framework with BaseKBQAAgent, adapters, 97 tests |
| [Tasks/archive/sciqa-agent-implementation.md](Tasks/archive/sciqa-agent-implementation.md) | Archived | SciQA/ORKG agent with 468-pair ground truth benchmark |
| [Tasks/archive/scratchpad-enforced-agent-loop.md](Tasks/archive/scratchpad-enforced-agent-loop.md) | Archived | Scratchpad-first loop, tool response truncation, journal reflection |

---

## Project Root Documentation

| Document | Description |
|----------|-------------|
| [README.md](../README.md) | Setup instructions, environment variables, database bootstrap |
| [AGENT_ARCHITECTURE.md](../AGENT_ARCHITECTURE.md) | Detailed agent architecture reference (1000+ lines) |
| [TOOLS_REFERENCE.md](../TOOLS_REFERENCE.md) | Complete KQAPro MCP tool reference (25 tools) |
| [CLAUDE.md](../CLAUDE.md) | AI assistant instructions and codebase context |
| [docs/datasets/kqapro.md](../docs/datasets/kqapro.md) | KQAPro dataset structure (~94K Q&A, 1.6M RDF triples) |
| [docs/datasets/sciqa.md](../docs/datasets/sciqa.md) | SciQA/ORKG dataset structure (468 Q&A, 1.1M triples) |
| [docs/guides/dataset_integration.md](../docs/guides/dataset_integration.md) | How to integrate a new KG into the system |
| [paper/main.tex](../paper/main.tex) | SEMANTiCS 2026 submission — AMA-KBQA system paper |

---

## CHANGELOG

See [CHANGELOG.md](CHANGELOG.md).

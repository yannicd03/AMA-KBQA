# Project Architecture

## Overview

AMA KBQA (Knowledge Base Question Answering) is a multi-agent system for answering natural language questions using **multiple knowledge graphs**. The system is designed to prove generalization across different RDF/SPARQL databases.

**Supported Knowledge Graphs:**
- **KQAPro** - Wikidata-derived factoid Q&A (~47K entities) ✅ Active
- **SciQA/ORKG** - Scientific research papers and contributions (~160MB, 468 Q&A pairs) ✅ Active

The system uses a combination of:
- **LLM-based reasoning** (via OpenRouter, KIT)
- **Vector semantic search** (Qdrant)
- **SPARQL queries** (Virtuoso)
- **MCP (Model Context Protocol)** for tool communication

**Platform:** Linux (Hetzner VPS for server deployment; originally developed on Windows).
Commands shown use Linux/bash syntax. Windows users may need path adjustments.

---

## Project Structure

```
ama-kbqa/
├── ama_kbqa/                   # Main Python package
│   ├── cli.py                  # CLI entrypoint (ama-kbqa command)
│   ├── benchmark_agents.py     # Unified batch processing & multi-model benchmarking (includes tool trace export)
│   ├── analyze_benchmark_run.py # Offline run-audit CLI: accuracy, wrong rows, tool adoption, zero-tool/context buckets
│   ├── postprocessing.py       # PostProcessor class (choice/sparql/llm_judge/simple)
│   ├── fewshot_generator.py    # LLM-based fewshot example generator — config-driven, tool-catalog-aware
│   ├── run_fewshot_generator.py # Post-hoc generator runner: replays over saved benchmark dir, shadow output only
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   └── trace_utils.py      # Tool trace extraction & few-shot export
│   ├── framework/              # ✅ Generic KBQA Framework
│   │   ├── __init__.py         # Package exports
│   │   ├── operations.py       # Abstract op contract: ATOMIC_OPERATIONS (11 req + 3 opt), CoverageReport, validate_bindings() ✅ NEW
│   │   ├── deterministic.py    # Shared math core: parse_numeric, compare_numeric (servers wrap, not duplicate) ✅ NEW
│   │   ├── config.py           # Configuration dataclasses
│   │   ├── state.py            # JournalState and JournalManager
│   │   ├── mcp_client.py       # Shared MCPClient class
│   │   ├── base_agent.py       # BaseKBQAAgent ABC (~600 lines)
│   │   ├── text_tool_calls.py  # Text-mode tool-call shim (minimax-m2.7 compat)
│   │   ├── trace.py            # TraceEvent + TraceRecorder (OTel-shaped, ContextVar nesting)
│   │   └── adapters/           # KG-specific adapters
│   │       ├── base_adapter.py # BaseKGAdapter ABC: _create_config, get_operation_bindings (abstract), validate_operation_coverage; resolved_config cached property ✅ UPDATED
│   │       ├── resolve.py      # resolve_graph_config / resolve_vector_config: env → [kg.<code>] toml → legacy toml (kqapro/sciqa only) → adapter default ✅ NEW
│   │       ├── kqapro_adapter.py # KQAPro: 11 required + select_extreme; NS_*/SPARQL_PREFIXES sourced by kqapro_server.py ✅ UPDATED
│   │       └── sciqa_adapter.py  # SciQA/ORKG: 11 required + aggregate/frequent_values; owl: prefix injected ✅ UPDATED
│   ├── agents/                 # Agent implementations
│   │   ├── kqapro_agent/       # KQAPro KBQA agent
│   │   │   ├── agent.py        # KQAProAgent class (~290 lines, inherits BaseKBQAAgent)
│   │   │   └── prompts.py      # All prompts extracted for maintainability (~665 lines)
│   │   ├── sciqa_agent/        # SciQA/ORKG agent ✅ Active
│   │   │   ├── agent.py        # SciQAAgent class (~190 lines, inherits BaseKBQAAgent)
│   │   │   └── prompts.py      # SciQA-specific prompts (~890 lines)
│   │   └── orchestrator_agent/ # Multi-agent router
│   │       └── agent.py        # Orchestrator class
│   ├── server/                 # MCP servers (FastMCP)
│   │   ├── kqapro_server.py    # KQAPro tools (29 tool-decorated functions: 28 user-visible + 1 LLM-hidden `GetJournalStateJSON`; verified by grep 2026-07-18)
│   │   ├── sciqa_server.py     # SciQA/ORKG tools (28 tool-decorated functions: 4 discovery, 6 retrieval, 12 domain, 1 SPARQL, 1 verification, 2 state, 1 hidden snapshot) ✅ Active
│   │   └── orchestrator_server.py # Routing tools
│   ├── frontend/               # Streamlit multi-page app
│   │   ├── app.py              # Main entry point (page config + sidebar)
│   │   ├── pages/              # Streamlit pages
│   │   │   ├── 1_Chat.py       # Interactive Q&A; background-thread run; live SVG lifecycle; Lifecycle/Trace tab panel (Graph removed); Simplified view toggle; persists sub-agent in session_state["persistent_agent"] for multiturn
│   │   │   ├── 2_Batch_Processing.py # Batch runner: live progress, tqdm parsing, console auto-scroll
│   │   │   ├── 3_Evaluation.py # Results dashboard (charts, metrics, per-question details)
│   │   │   ├── 4_Settings.py   # config.toml editor
│   │   │   ├── 5_Trace_Inspector.py  # Thin wrapper: trace selector + render_trace_panel()
│   │   │   └── (6_Graph_View.py deleted — utils kept; see Decisions/live-trace-and-chat-unification.md v3)
│   │   └── utils/              # Shared utilities
│   │       ├── styling.py      # CSS, ansi_to_html, avatars, kind-coloured span pills
│   │       ├── async_helpers.py # run_async() wrapper
│   │       ├── agent_factory.py # Agent creation + metadata
│   │       ├── batch_results_loader.py # Load benchmark result files from benchmark_results/
│   │       ├── config_editor.py # config.toml loading/saving
│   │       ├── trace_render.py  # Pure-Python trace helpers: build_tree, summarise, render_tree_html
│   │       ├── graph_html.py    # journal_to_graph, build_graph_html (vis-network HTML template)
│   │       ├── lifecycle_runner.py # LiveLifecycleState, start_run(is_continuation=), drain_into() — background thread + queue; propagates trace_id on continuation ✅ NEW
│   │       ├── lifecycle_svg.py    # Inline-SVG agent-lifecycle figure (ported from paper TikZ, data-state nodes) ✅ NEW
│   │       ├── lifecycle_mapping.py # SPAN_KIND_TO_NODE, TOOLS_A/TOOLS_B frozensets ✅ NEW
│   │       ├── trace_panel.py   # render_trace_panel() — extracted panel helper shared by Chat + page 5 ✅ NEW
│   │       └── graph_panel.py   # render_graph_panel() — extracted panel helper shared by Chat + page 6 ✅ NEW
│   ├── retrieval/              # ✅ Centralised retrieval module (BM25 hybrid + reranker)
│   │   ├── __init__.py
│   │   ├── embeddings.py       # get_query_embedding() — cached lru_cache query embedding
│   │   ├── search.py           # RetrievalParams, build_retrieval_params, search(), search_terms() — dense + hybrid dispatch
│   │   └── reranker.py         # Lazy CrossEncoder singleton; rerank() overwrites point.score
│   ├── config.py               # Configuration loader
│   ├── benchmark_agents.py     # Unified batch processing & multi-model benchmarking (includes tool trace export)
│   ├── postprocessing.py       # Postprocessing modes (choice, sparql, llm_judge, simple)
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   └── trace_utils.py      # Tool trace extraction & few-shot export
├── tests/                      # Test suite (382 tests total, verified 2026-07-18)
│   ├── framework/               # Framework unit tests (202 tests) — base agent, journal, trace, adapters, operations, synthesis guards, tool loop
│   │   ├── test_operations.py  # Registry shape, two-level semantics, adapter coverage, binding-rot guard (source-level regex)
│   │   ├── test_deterministic.py # parse_numeric variants, compare_numeric correctness/error paths; pins byte-identical output
│   │   ├── test_base_agent_synthesis_guards.py # Synthesis-funnel hard-stop guards: max-iterations→synthesis, malformed tool-args, GetJournalSummary fallback, empty-choices guard (audit B1/B2) ✨ NEW
│   │   ├── test_base_agent_tool_loop.py # Tool-loop gating: qtype filter + denylist interaction ✨ NEW
│   │   ├── test_config.py      # Configuration tests
│   │   ├── test_state.py       # State management tests
│   │   ├── test_adapters.py    # Adapter tests
│   │   ├── test_trace.py       # TraceRecorder: nesting, contextvar isolation, sync/async parity, JSONL roundtrip, listeners
│   │   ├── test_trace_render.py # trace_render helpers: tree-building, HTML, journal→graph
│   │   └── test_base_agent_multiturn.py # reset semantics + first-turn vs follow-up hook skipping (multiturn)
│   ├── agents/                  # Agent-level tests (33 tests) — orchestrator routing, sciqa denylist gate
│   ├── server/                  # MCP server tests (41 tests) — kqapro_server.py, sciqa_server.py tool behavior
│   ├── retrieval/                # Retrieval module tests (54 tests) — embeddings, search, reranker, BM25 hybrid
│   ├── frontend/                # Frontend unit tests (42 tests)
│   │   ├── test_lifecycle_svg.py    # SVG generation, node state transitions
│   │   ├── test_lifecycle_mapping.py # SPAN_KIND_TO_NODE, TOOLS_A/B coverage
│   │   └── test_lifecycle_runner.py  # start_run/drain_into with stub agent + threading.Event; continuation through real worker
│   └── test_seed_and_progress.py # --seed → get_chat_seed() threading; benchmark progress-reporting fixes (10 tests) ✨ NEW
├── db/                         # Database utilities
│   ├── docker-compose.yml      # Virtuoso + Qdrant + frontend (3 services on Hetzner)
│   ├── populate_vector_db.py   # KQAPro Qdrant initialization
│   ├── populate_sciqa_vectors.py # SciQA Qdrant initialization
│   ├── migrate_add_bm25.py     # Crash-safe 2-stage migration: adds BM25 sparse index to existing collections without re-embedding
│   └── datasets/
│       ├── kqapro/
│       │   ├── kb.json         # KQAPro knowledge base
│       │   ├── convert_kb_to_nt.py
│       │   └── fewshot-examples/
│       │       ├── Count.json              # Per-qtype tool-trace examples (includes genericized UNION counting example with placeholders)
│       │       ├── QueryRelationQualifier.json  # 3 qualifier selection examples (point_in_time vs for_work, role, ceremony)
│       │       ├── Query.json              # and 8 other qtype files
│       │       ├── _general.json           # Cross-type general guidance (5 entries: verify constraints, trust KB, match qualifier, use FindByAttribute, use SPARQL)
│       │       └── _tool_tips.json         # Tool-specific tips (GetEdgeQualifiers updated 2026-02-13: "For what" marked AMBIGUOUS - check both for_work AND ceremony)
│       └── SciQA/
│           ├── ORKG RDF dump 14.02.2023.nt  # ORKG knowledge graph
│           ├── Handcrafted/    # 100 expert Q&A pairs
│           └── Autogenerated/  # 368 auto Q&A pairs
├── benchmark_results/          # Output from batch processing & benchmark runs
│   └── <YYYY-MM-DD-N>/         # Date-based benchmark run with incrementing number
│       ├── overview.json       # Multi-model leaderboard (multi-model mode only)
│       ├── benchmark_results.csv # CSV export (if --export-csv)
│       └── <agent>/            # Per-agent results (kqapro, sciqa)
│           └── <model>/        # Per-model results (or "default")
│               ├── results.json        # Per-question results
│               ├── summary.json        # Aggregate statistics
│               ├── console_output.txt  # Full console log
│               ├── detailed_log.txt    # Intermediate thinking
│               ├── judgments.json      # LLM judge evaluations (if llm_judge mode)
│               ├── generated_fewshot.json  # LLM-generated fewshot audit log (if --generate-fewshot)
│               └── tool_traces/        # Full conversation traces
│                   ├── question_000.json
│                   ├── question_001.json
│                   └── question_002.json
├── logs/                       # Server logs
├── config.toml                 # Central configuration
├── .env                        # API keys (not committed)
├── pyproject.toml              # Dependencies
├── CLAUDE.md                   # AI assistant instructions
├── AGENT_ARCHITECTURE.md       # Detailed agent documentation
└── TOOLS_REFERENCE.md          # Complete tool reference
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.12+ | Primary implementation |
| **Agent Framework** | openai-agents, FastMCP | Tool-calling agents with MCP |
| **Vector DB** | Qdrant (`qdrant-client>=1.17.0`; image `v1.17.1`) | Semantic entity/relation search; hybrid BM25+dense via Query API (RRF/DBSF); server-side BM25 inference requires >= 1.15.2 |
| **Graph DB** | Virtuoso 7 | RDF triple store, SPARQL queries |
| **LLM Access** | OpenAI SDK + `chatkit` (git dep, pinned tag `v1.0.0`) | Compatible with OpenRouter, LMStudio, KIT; `chatkit.retry.TransientRetry` wraps every sampled agent LLM call to catch KIT's non-5xx transient errors — see [agent_framework.md](agent_framework.md#transient-llm-retry-chatkitretrytransientretry) |
| **Embedding** | qwen/qwen3-embedding-8b (4096 dim) | Entity/relation embeddings |
| **Frontend** | Streamlit | Multi-page web UI (Chat, Batch, Evaluation, Settings) |
| **Configuration** | TOML + dotenv | Centralized config management |
| **Containerization** | Docker Compose | Database services |

---

## Core Components

### 0. Generic KBQA Framework (`ama_kbqa/framework/`)

The framework provides abstract base classes (`BaseKBQAAgent`, `BaseKGAdapter`) that both KQAProAgent and SciQAAgent inherit from, reducing code duplication by ~70% and ensuring consistent behavior — tool gating, synthesis funnel hard-stops, trace instrumentation, text-tool-call mode, MCP client pattern, token/duration tracking, reset patterns, and LLM sampling seed control.

**This section moved to a dedicated doc during the 2026-07-18 `agent_system.md` split** (the framework module table, `BaseKBQAAgent` provides-list, and all mechanics detail live there now): see **[System/agent_framework.md](agent_framework.md)**. The 11 framework files (`operations.py`, `deterministic.py`, `config.py`, `state.py`, `mcp_client.py`, `base_agent.py`, `text_tool_calls.py`, `trace.py`, `adapters/base_adapter.py`, `adapters/kqapro_adapter.py`, `adapters/sciqa_adapter.py`) are listed in the directory tree above and detailed in that doc's Framework Files table.

### 1. KQAProAgent (`ama_kbqa/agents/kqapro_agent/agent.py`)

The main KBQA agent that answers questions by inheriting from `BaseKBQAAgent`:

1. **Pre-Agent Hook** - Classifies question type and extracts entities
2. **Iterative Tool Loop** - Calls MCP tools to gather information
3. **Post-Agent Hook** - Synthesizes final answer from journal

**Key Features (inherited from BaseKBQAAgent):**
- Question type classification (10 types: Count, Verify, Select, SelectBetween, SelectAmong, QueryAttr, QueryAttrQualifier, QueryRelation, QueryRelationQualifier, QueryName)
- 7-layer loop detection (KQAPro exercises layers 1-4 and 6-7; see [System/kqapro_agent.md](kqapro_agent.md#loop-detection-kqapro-exercises-layers-1-4-and-6-7-layer-5-is-sciqa-specific))
- Automatic journal tracking (visited nodes, found values)
- Separate synthesis model configuration
- Token usage tracking

**File Organization:**
- `agent.py` (~290 lines) - Implements abstract methods, KQAPro-specific logic
- `prompts.py` (~665 lines) - All prompts extracted for maintainability

**Full deep-dive (lifecycle diagram, journal data model, message history format):** [System/kqapro_agent.md](kqapro_agent.md)

### 1.1 Prompts Module (`ama_kbqa/agents/kqapro_agent/prompts.py`)

All prompts are centralized in a separate module for easier maintenance:

| Prompt | Purpose |
|--------|---------|
| `QTYPE_STRATEGIES` | 10 question-type-specific reasoning strategies. Count: FilterEntities for attribute filtering + RunSPARQL COUNT for large sets + OR/UNION section. Verify: VerifyNumericCondition for numeric/date, VerifyString for text. QueryAttrQualifier: QualifierFilter for narrowing by qualifier conditions + question-word-to-qualifier mapping (When→point_in_time, Where→location). QueryRelationQualifier: QualifierFilter + qualifier mapping (When→point_in_time/start_time, Where→location, What role→object_has_role, ceremony, For what→AMBIGUOUS). QueryName: FilterEntities for type+attribute conditions. Query: prepositional phrase cross-reference in step 2b. |
| `SYSTEM_PROMPT` | Main agent system prompt with 8 KBQA rules. Tool tier section updated with T1.5 Filtering (FilterEntities, QualifierFilter) and T4 Verify (VerifyNumericCondition, VerifyString). Includes prepositional phrase disambiguation rule #7 and DO NOT BACKTRACK journal confidence rule. |
| `CLASSIFICATION_PROMPT_TEMPLATE` | Question classification prompt |
| `ENTITY_EXTRACTION_PROMPT` | Entity/relation extraction prompt |
| `ANALYSIS_CONTEXT_TEMPLATE` | Pre-analysis injection template |
| `SYNTHESIS_PROMPT_TEMPLATE` | Final answer synthesis prompt |
| `TOOL_LOOP_GUIDANCE` | Tool-specific loop recovery guidance |
| `LOOP_INTERVENTION_TEMPLATE` | Loop detection intervention message |
| `GENERAL_GUIDANCE_TEMPLATE` | Cross-type insights from _general.json (5 entries: verify constraints, trust KB data, match qualifier, use FindByAttribute, use SPARQL) |
| `TOOL_TIPS_TEMPLATE` | Tool-specific tips from _tool_tips.json (max 10 entries loaded) |

### 1.2 SciQAAgent (`ama_kbqa/agents/sciqa_agent/agent.py`)

The SciQA agent for Open Research Knowledge Graph (ORKG) scientific QA, inheriting from `BaseKBQAAgent`:

1. **Pre-Agent Hook** - Classifies question type (8 types), extracts entities, loads type-specific strategy
2. **Iterative Tool Loop** - Calls SciQA MCP tools (28 tool-decorated functions) to gather research information
3. **Post-Agent Hook** - Synthesizes final answer from journal

**Key Features (inherited from BaseKBQAAgent):**
- Question type classification (8 types: Factoid, Count, List, Boolean, Comparison, Superlative, Aggregation, General)
- Type-specific strategy loading (only the relevant strategy is injected after classification)
- Lean SYSTEM_PROMPT with strategy-specific content in QTYPE_STRATEGIES
- 7-layer loop detection, same journal system as KQAPro, plus a `RunORKGSPARQL` cap (layer 5) and an env-gated denylist (`AMA_SCIQA_DISABLE_RAW_SPARQL`) — see [System/sciqa_agent.md](sciqa_agent.md#env-gated-tool-denylist-ama_sciqa_disable_raw_sparql)
- Separate synthesis model configuration
- Token and tool call duration tracking

**File Organization:**
- `agent.py` (~190 lines) - Implements abstract methods, SciQA-specific logic
- `prompts.py` (~890 lines) - ORKG-specific prompts, 8 enriched strategies with few-shot examples (author search, negation, aggregation scoping, energy domain, boolean comparison-embedded values), predicate dictionary, 9 loop recovery entries. Current guidance is evidence-first: high-level aggregation tools before raw SPARQL for supported count/superlative/aggregation shapes. Deep-dive fixes (nested row extraction, paired grouping, anchor lint, final-answer contract) landed 2026-06-15, commit `7397cc3`.

**Full deep-dive (lifecycle, multi-label classifier tolerance, tool fixes, predicate reference):** [System/sciqa_agent.md](sciqa_agent.md)

### 1.3 Prompts Module (`ama_kbqa/agents/sciqa_agent/prompts.py`)

SciQA-specific prompts for scientific domain (~890 lines). Follows a type-specific strategy loading pattern matching KQAPro:

| Prompt | Purpose |
|--------|---------|
| `QTYPE_STRATEGIES` | 8 question-type-specific strategies (Factoid, Count, List, Boolean, Comparison, Superlative, Aggregation, General) with comparison patterns, HAS_VALUE nested patterns, SPARQL templates, and decision trees. Only the relevant strategy is loaded after classification. |
| `SYSTEM_PROMPT` | Lean agent system prompt (~165 lines) with ORKG rules, evidence-first requirement, scope-before-query aggregation routing, 5-tier tool listing, 10-step execution strategy, and predicate reference dictionary. Strategy-specific content lives in QTYPE_STRATEGIES. |
| `CLASSIFICATION_PROMPT_TEMPLATE` | Scientific question classification (8 types) |
| `ENTITY_EXTRACTION_PROMPT` | Extract papers, authors, contributions, fields |
| `FEWSHOT_EXAMPLES` | 8 type-specific few-shot example sets loaded alongside strategies (enhanced with author search, negation queries, aggregation scoping, energy domain distinctions, boolean comparison-embedded values) |
| `SYNTHESIS_PROMPT_TEMPLATE` | Final answer synthesis |
| `TOOL_LOOP_GUIDANCE` | 9 tool-specific loop recovery entries (covers the high-traffic tool classes, includes RunORKGSPARQL 10-call cap warning) |

### 2. MCP Server (`ama_kbqa/server/kqapro_server.py`)

Provides 29 tool-decorated functions (28 user-visible + 1 LLM-hidden) for knowledge graph interaction, organized by tier. `NS_*` namespace constants and `SPARQL_PREFIXES` are sourced from `_ADAPTER.config.namespaces` at module level (not hardcoded); `VIRTUOSO_ENDPOINT`, `COLLECTION_ENTITIES`, and `COLLECTION_RELATIONS` are sourced from `_ADAPTER.resolved_config` (adapter defaults + `config.toml`/`AMA_KBQA_*` overrides — see [Decisions/kg-adapter-config-resolution.md](../Decisions/kg-adapter-config-resolution.md)). The KQAPro adapter's `get_operation_bindings()` maps each of these tools to an abstract operation name; see `framework/operations.py` and `Decisions/abstract-operation-contract.md`.

**T1 Discovery:**
- `FindNode` - Semantic entity search (deduplicates via `visited_nodes` cache)
- `FindByAttribute` - Reverse lookup by attribute value (prefer for unique IDs/codes/URLs)

**T1.5 Filtering:** ✨ NEW TIER
- `FilterEntities` - Unified filtering by concept type and/or attribute value with operator support. Handles string, numeric, date, and year auto-detection. Supports chaining via `entity_ids` parameter. Parameters: `concept`, `attribute_name`, `attribute_value`, `operator` (=,!=,<,>,<=,>=,contains), `entity_ids`, `limit`. Returns `SearchResponse`.
- `QualifierFilter` - Filters entities by qualifier values on reified RDF statements (facts about facts). Handles QFilterStr/QFilterNum/QFilterYear/QFilterDate KoPL patterns. Parameters: `entity_ids`, `relation_or_attribute`, `qualifier_name`, `qualifier_value`, `operator`, `limit`. Returns `SearchResponse`.

**T2 Retrieval:**
- `GetNodeSummary` - Complete node data in one call
- `GetAttributeDetails` - Specific attribute values
- `GetRelationDetails` - Connected entities via relation. Writes to **both** `verified_facts` (provenance) and `found_values` (answer visibility to synthesis). Before April 2026 it only wrote to `verified_facts`, making relation-based answers invisible to the synthesis step.
- `GetRelationBetween` - Deterministically inspects all predicates connecting two known endpoints in both directions. Returns `subject_to_object`, `object_to_subject`, and `preferred_answer` so QueryRelation questions can preserve direction instead of inferring it from node summaries.

**T3 Qualifiers:**
- `GetEdgeQualifiers` - Attribute statement qualifiers (uses RDF reification pattern); returns full qualifier dict — use for discovery. Relation-style calls with `predicate` + target Q-id delegate through the private relation-qualifier helper rather than calling the decorated public tool object.
- `GetQualifiersByPredicate` - Relation statement qualifiers; returns full qualifier dict — use for discovery. Internally backed by `_get_qualifiers_by_predicate_impl` so other server tools can reuse it safely.
- `GetQualifierValue` - Direct projection of a single qualifier value (`subject_id`, `predicate`, `target`, `qualifier_name`). Auto-detects relation vs attribute from target shape (Q-id → relation; literal → attribute); auto-retries backward direction; unwraps qualifier bnodes via `rdf:value`; supports slash qualifier URIs such as `number_of_matches_played/races/starts` through aliases `number_of_matches`, `matches_played`, and `appearances`; also normalizes subscriber/follower wording to `number_of_subscribers`/`number_of_followers`. Logs to `verified_facts` as `type="qualifier_value"` and to `found_values` under `predicate.qualifier_name`. **Preferred over full-dict tools once qualifier name is known.** ✨ UPDATED
- `GetAttributeWithQualifiers` - Attribute values with all context (dual-method SPARQL: blank-node pattern + RDF reification pattern)
- `TemporalAttributeQuery` - Date-specific attribute lookup

**T4 Verify:**
- `VerifyNumericCondition` - Deterministic math/date comparison (TRUE/FALSE/ERROR)
- `VerifyString` - Deterministic string comparison with 4 modes: `exact`, `case_insensitive`, `contains`, `normalize` (default — strips diacritics, punctuation, whitespace). Parameters: `value1`, `value2`, `mode`. Returns `StringComparisonResponse`. ✨ NEW

**Complex:**
- `RunSPARQL` - Raw SPARQL queries. Stores a bounded preview in `found_values["sparql_result_N"]` to survive message truncation; SELECT responses include `result_count`, `returned_count`, `truncated`, and a refinement note when Virtuoso returns more rows than the agent context should receive.
- `CountUnion` - Branch-aware `COUNT(DISTINCT ?entity)` over heterogeneous OR conditions via SPARQL `UNION`. Prevents relation-derived candidate leakage and double-counting in Count questions that combine multiple concept/attribute branches. Branches can now include relation-local filters (`relation_name`, `relation_target_id(s)`, `relation_direction`) so relation constraints stay inside the correct OR branch instead of becoming global intersections.
- `CompareEntities` - Compare attribute across multiple entities
- `FindEntitiesByRelationPath` - Multi-hop entity discovery

**Utility:**
- `GetNodeLabel` - Quick label lookup for an entity ID. Also backfills `found_values` entries whose `related_id` matches the resolved node (upgrades opaque IDs to human labels in the rendered summary).
- `BatchGetNodeLabels` - Batch label resolution. Same `found_values` backfill behavior as `GetNodeLabel`.
- `GetSchemaForAttribute` - Discover how an attribute is encoded in the KB
- `ManageJournal` - Scratchpad management (actions: `set_question`, `update_plan`, `update`)
- `GetJournalSummary` - Summary of all discoveries
- `ExploreNeighborhood` - Explore entity's full neighborhood (deprecated)

**New Pydantic Model:**
- `StringComparisonResponse` - Response model for `VerifyString` with `verdict` (TRUE/FALSE/ERROR), `explanation`, and `mode_used` fields.

**CORE_TOOLS (always available regardless of qtype):**
`FindNode`, `FindByAttribute`, `GetAttributeDetails`, `GetRelationDetails`, `GetNodeSummary`, `RunSPARQL`, `GetNodeLabel`, `BatchGetNodeLabels`, `ManageJournal`, `GetJournalSummary`, `FilterEntities`

**LLM-hidden internal tools (filtered from the tool list sent to the LLM):**
- `GetJournalStateJSON()` — returns `session_journal.model_dump()` as a JSON string. Called only by the agent's snapshot path (after journal-mutating tool calls) to capture structured journal state for the Graph View. Not exposed to the LLM. ✅ NEW

**Key implementation details:**
- `JournalState` imported from `ama_kbqa.framework.state` (not defined locally)
- `FindNode` deduplicates via `visited_nodes`: skips Qdrant search if label already cached (case-insensitive)
- `KQAProAgent._extract_exact_attribute_constraints()` detects exact attribute/value constraints such as `official name`, `date of birth`, `IAB code`, `ICD-10-CM`, `UMLS CUI`, `ISWC/ISNI`, and `known under <identifier>`. These constraints are injected into pre-analysis and can drive fast-path `FindByAttribute`, preventing same-name semantic matches from overriding explicit identifiers.
- `RunSPARQL`, `CountUnion`, `GetRelationBetween`, and `GetQualifierValue` store answer-grade values in `found_values` so they survive message truncation and are visible to synthesis
- Public MCP tool functions should not call other decorated public tool names directly. Shared behavior that needs server-internal reuse lives in private helpers such as `_batch_get_node_labels_impl` and `_get_qualifiers_by_predicate_impl`.
- `GetRelationDetails` writes to both `verified_facts` (provenance) and `found_values` (required for synthesis visibility)
- `GetNodeLabel` / `BatchGetNodeLabels` backfill `value` fields in `found_values` entries after resolving a node's label
- `GetJournalSummary` renders `found_values` as `📊 DISCOVERED VALUES` and `verified_facts` as `🔗 VERIFIED FACTS` (up to 15 triples); IDs are resolved to labels via `visited_nodes` at render time
- `completed_steps` capped at 20 entries; `failed_attempts` capped at 10 entries; all append sites use `add_completed_step()` / `add_failed_attempt()` helpers
- Synthesis runs with minimal context (system + journal + query only); `_run_synthesis` detects data presence by checking for literal substrings `"discovered values"`, `"verified facts"`, `"partial answer:"`, `"orkgr:"` in the rendered summary

### 2.1 SciQA MCP Server (`ama_kbqa/server/sciqa_server.py`)

Provides 28 tool-decorated functions for ORKG knowledge graph interaction: 24 user-visible KB tools, 2 state-management tools, 1 verification tool, and 1 LLM-hidden journal snapshot tool. `NS_*` constants and `SPARQL_PREFIXES` (including `owl:`) are sourced from `_ADAPTER.config.namespaces` at module level. `VIRTUOSO_ENDPOINT`, `SCIQA_GRAPH`, `COLLECTION_ENTITIES`, and `COLLECTION_RELATIONS` are sourced from `_ADAPTER.resolved_config` — see [Decisions/kg-adapter-config-resolution.md](../Decisions/kg-adapter-config-resolution.md). The SciQA adapter's `get_operation_bindings()` covers the 11 required abstract operations plus optional `aggregate` and `frequent_values`; see `Decisions/abstract-operation-contract.md`.

**Tier 1 - Discovery (4 tools):**
- `FindResource` - Semantic vector search for papers, authors, contributions, with high-confidence lexical label promotion when a query contains a resource title plus extra words; short title-like queries run strict token-coverage label lookup before broad token fallback
- `FindPredicate` - Find ORKG predicate names by description
- `FindByPredicateValue` - Reverse lookup: find resources by predicate value (exact/contains/greater/less)
- `FindAuthorPapers` - SPARQL-based author name search (case-insensitive partial match, better than vector search for proper nouns) ✨ NEW

**Tier 2 - Retrieval (6 tools):**
- `GetResourceDetails` - Full resource details (label, type, relations)
- `GetResourceSummary` - ALL predicates in one SPARQL call (preferred for exploration)
- `GetRelationTargets` - Follow specific relations with **reverse lookup fallback** (if direct query returns 0 results, tries finding resources that reference the target) ✨ UPDATED
- `GetResourceLabel` - Quick label lookup
- `BatchGetResourceLabels` - Batch label resolution
- `CompareResources` - Compare a predicate across multiple resources (returns sorted)

**Tier 3 - Domain-Specific (12 tools):**
- `GetPaperContributions` - Get contributions for a paper (P31)
- `GetPaperAuthors` - Get authors (P6/P27)
- `GetContributionMethods` - Get methods used (P2)
- `GetResearchFieldPapers` - List papers in field (P30)
- `GetComparisonContributions` - Navigate Comparison -> Contribution pattern with predicate discovery mode, now **stores values in journal's found_values** ✨ UPDATED
- `FollowRelationPath` - Multi-hop relation navigation in one SPARQL call
- `InspectComparisonSchema` - Compact schema map for Comparison resources. Lists direct contribution predicates, one-hop nested contribution -> row-object -> metric paths, and two-hop nested contribution -> node -> row-object -> metric paths with labels, counts, sample values, numeric/HAS_VALUE evidence, unit samples, rollup-like intermediate labels when present, and `AggregateComparisonValues` usage hints. Deeper paths include `intermediate_path="P1,P2"` hints so agents can stay inside wrapped graph operations instead of enumerating rows or writing raw SPARQL. Use before choosing predicates or nested row filters for aggregation. ✨ UPDATED
- `QueryComparisonRows` - Multi-predicate row selector for Comparison contributions. Applies one filter per column/constraint, projects several return predicates, and writes the matched rows to `found_values`; use before raw SPARQL for "column A = X and column B = Y, return metrics C/D/E" questions. ✨ NEW
- `AggregateComparisonValues` - Single SPARQL + Python aggregation over Comparison contributions. Handles HAS_VALUE/label indirection; supports avg/sum/min/max/count/count_distinct/mode_top/all_values plus aliases like frequency/mean/total; optional grouping, `group_by_path` for grouping keys reached through relation paths such as contribution → scenario → goal → time frame, `group_by_intermediate` for nested-row labels as a grouping axis, pre-filtering, `value_via_group` 2-hop switch, `comparison_ids` CSV for multi-Comparison union mode without a redundant `comparison_id`, `value_parser` for embedded numeric strings, `return_predicate` for companion values on min/max rows, `intermediate_predicate` for nested rows such as contribution → energy source → electricity generation, `intermediate_path` for deeper contribution → node → row paths, `intermediate_filter_value` for restricting nested row objects by label/ID including rollup rows like "all sources", and `value_predicates` for unioning sibling metric predicates even when `value_predicate` is empty. Ungrouped, unfiltered nested numeric aggregations return denominator hints, rollup candidates, and when a rollup exists a `recommended_follow_up` plus an `ambiguous_denominator` journal record instead of promoting row-level aggregates as final. Grouped table outputs are left as answer-grade grouped rows. ✨ UPDATED
- `DiagnoseComparisonAggregation` - Wrapped SPARQL diagnostics for denominator/scope ambiguity after schema discovery. Reports total scope contributions, matched value rows, distinct contributions, nested/group populations, row-level numeric summaries, per-contribution sum/mean candidates, and per-intermediate/per-group candidates. Use before answering averages/counts where the wording could mean all rows, per study/contribution, or per category. ✨ NEW
- `FindFrequentValues` - Cross-resource aggregation for global-scope questions ("most popular X", "largest Y across the papers"). No Comparison anchor required. Scope tiers: research_field_id → P30/P31; comparison_ids → VALUES union; default → all compareContribution subjects; `scope="papers"` scans paper contributions. `value_source="subject"` reads values from scoped Paper/Comparison resources themselves, covering paper metadata like top research fields. Supports the same agg, `value_parser`, and `return_predicate` modes as AggregateComparisonValues; writes results to `found_values`; hard cap `limit_subjects=5000`. ✨ UPDATED
- `FindCoAuthors` - Finds co-authors of papers by a seed author (case-insensitive partial match; handles both resource-URI and literal-string author predicates P6/P27); returns co-authors sorted by shared-paper count. Closes Q2 co-author pattern.

**Tier 4 - Raw SPARQL (1 tool):**
- `RunORKGSPARQL` - Raw SPARQL queries (prefixes auto-injected, **capped at 10 calls per question** to prevent runaway SPARQL spirals). SELECT responses are compacted to a bounded row preview with `result_count`, `returned_count`, `truncated`, and a refinement note so broad exploratory queries cannot overflow the LLM context window. ✨ UPDATED

**Tier 5 - Verification (1 tool):**
- `VerifyNumericCondition` - Deterministic math/date comparison (TRUE/FALSE/ERROR)

**State Management (2 tools):**
- `ManageJournal` - Scratchpad management
- `GetJournalSummary` - Summary of discoveries

**LLM-hidden internal tools:**
- `GetJournalStateJSON()` — structured journal snapshot for Graph View (not exposed to LLM). ✅ NEW

### 3. Configuration System (`ama_kbqa/config.py`)

Centralized configuration loaded from `config.toml`:

```python
from ama_kbqa.config import (
    get_chat_client,           # OpenAI-compatible client
    get_embedding_client,      # For vector embeddings
    get_chat_model_name,       # Current chat model
    get_synthesis_client,      # Synthesis-specific client
    get_synthesis_enabled,     # bool — whether to run the synthesis LLM step (default True)
    get_auto_inject_journal,   # bool — whether to auto-push journal into tool loop (default True)
    get_qdrant_host,           # Database connection
    get_virtuoso_endpoint,     # SPARQL endpoint — deprecated for the two MCP servers (see below), still used by ama_kbqa/utils/artifact_golds.py and db/migrate_add_bm25.py
    get_retrieval_config,      # Returns the full [retrieval] section as a dict
    get_hybrid_enabled,        # bool — enable BM25 hybrid search (default False)
    get_fusion,                # str — "rrf" or "dbsf" (default "rrf")
    get_prefetch_limit,        # int — candidates per branch in hybrid prefetch (default 20)
    get_reranker_enabled,      # bool — enable cross-encoder reranking (default False)
    get_reranker_model,        # str — HuggingFace model id for cross-encoder
    get_rerank_candidates,     # int — number of candidates to rerank (default 20)
    get_rerank_threshold,      # float | None — minimum cross-encoder score filter
)
```

**`[retrieval]` config section** — controls the `ama_kbqa/retrieval/` module:

```toml
[retrieval]
hybrid_enabled = false          # off by default; requires BM25 migration first
fusion = "rrf"                  # "rrf" (Reciprocal Rank Fusion) or "dbsf" (Distribution-Based Score Fusion)
prefetch_limit = 20             # candidates fetched per branch (dense + BM25) before fusion
reranker_enabled = false        # requires uv sync --extra rerank
reranker_model = "Alibaba-NLP/gte-reranker-modernbert-base"
rerank_candidates = 20
```

Environment variable overlay: `AMA_RETRIEVAL_<KEY>` (upper-cased key, typed) overrides the TOML value. Env vars are the secondary channel for headless/benchmark use; the Settings page uses the disk/session-cache path. `RETRIEVAL_ENV_PREFIX = "AMA_RETRIEVAL_"`.

**`synthesis_enabled` config key** (`[synthesis]` section in `config.toml`):
- `true` (default) — run the dedicated synthesis LLM call after the tool loop
- `false` — return the agent's own last assistant message directly; saves one LLM call but loses deterministic answer shaping
- Exposed as a toggle in the Settings UI (`pages/4_Settings.py`, "Run synthesis step")

**`auto_inject_journal` config key** (`[agent]` section in `config.toml`):
- `true` (default) — `_run_tool_loop` periodically injects a journal refresh every N iterations and appends an "answer now" prompt whenever the agent calls `GetJournalSummary`
- `false` — both automatic injections are suppressed; `GetJournalSummary` remains available as a tool the agent can call voluntarily
- Exposed as a toggle in the Settings UI (`pages/4_Settings.py`, "Auto-inject journal into context", under Agent Configuration)

**KG endpoint / named-graph / vector-collection resolution** (2026-07, `ama_kbqa/framework/adapters/resolve.py`): the two MCP servers no longer call `get_virtuoso_endpoint` / `get_collection_entities` / `get_collection_relations` / `get_sciqa_*` directly. They read `_ADAPTER.resolved_config`, which layers deployment overrides on top of the adapter's declared `GraphConfig`/`VectorConfig` defaults, highest precedence first:
1. `AMA_KBQA_<CODE>_<FIELD>` env var (e.g. `AMA_KBQA_SCIQA_ENDPOINT`) — generic, works for any adapter code.
2. `[kg.<code>]` section in `config.toml` — generic; **this is how a new KG should be onboarded**, not by adding a getter to `ama_kbqa/config.py`.
3. Legacy per-KG `config.toml` keys — closed table, `kqapro`/`sciqa` only, kept for backward compat with existing `config.toml`/`config.docker.toml`.
4. The adapter's own default (`_create_config()`).

`config.toml` and `config.docker.toml` were not modified by this change — both still resolve via tier 3. The six legacy getters in `ama_kbqa/config.py` remain, with deprecation docstrings, for `ama_kbqa/utils/artifact_golds.py` and `db/migrate_add_bm25.py`. Full rationale, rejected alternatives, and two known footguns (silently-swallowed malformed env values; a new KG defaulting to localhost inside Docker if it skips the `[kg.<code>]` section): [Decisions/kg-adapter-config-resolution.md](../Decisions/kg-adapter-config-resolution.md).

### 4. Orchestrator Agent (`ama_kbqa/agents/orchestrator_agent/agent.py`)

Multi-agent router that routes each question to `KQAProAgent` or `SciQAAgent` (the only two implemented sub-agents; `_agent_config` currently has exactly the `kqapro`/`sciqa` keys — CodeAgent/MathAgent placeholders mentioned in earlier docs were never implemented) in a single LLM round-trip: a direct MCP probe call (`analyze_query_recommend_db`) followed by one forced `select_agent` tool call. Falls back to KQAProAgent only if the router returns an unrecognised agent key.

**Full deep-dive (evidence contract, `route_reason` span, design invariants):** [System/orchestrator_routing.md](orchestrator_routing.md)

### 5. CLI Entrypoint (`ama_kbqa/cli.py`)

Command-line interface for interacting with the KBQA system:

```bash
# Ask a question using the orchestrator (default)
ama-kbqa ask "Who directed Inception?"

# Ask using KQAPro subagent directly
ama-kbqa ask --subagent kqapro "Who directed Inception?"
ama-kbqa ask -s kqapro "How many films did Christopher Nolan direct?"

# Ask using SciQA subagent for scientific questions
ama-kbqa ask --subagent sciqa "What papers address text classification?"
ama-kbqa ask -s sciqa "Who are the authors of papers on machine learning?"

# Run KQAPro benchmarking
ama-kbqa benchmark --subagent kqapro -n 10 --seed 42
ama-kbqa benchmark -s kqapro -n 50 --seed 123 --postprocessing llm_judge

# Run SciQA benchmarking
ama-kbqa benchmark --subagent sciqa -n 10 --seed 42 --dataset handcrafted
ama-kbqa benchmark -s sciqa -n 50 --dataset auto --postprocessing simple
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `ask` | Ask a question to the KBQA system |
| `benchmark` | Run benchmarking on validation dataset |

**Ask Options:**

| Option | Description |
|--------|-------------|
| `query` | The question to ask (positional argument) |
| `--subagent, -s` | Specific subagent: `kqapro` or `sciqa`. If not specified, uses orchestrator |

**Benchmark Options:**

| Option | Description |
|--------|-------------|
| `--subagent, -s` | **Required.** The subagent to benchmark (`kqapro` or `sciqa`) |
| `-n, --n_questions` | Number of questions to sample (default: 10) |
| `--seed` | Random seed. Controls **both** dataset sampling (which questions are picked) **and**, as of 2026-06-15, the LLM `seed=` parameter on every sampled agent call (classify, tool-loop, text-only, orchestrator probe/judge) via `config.get_chat_seed()` / `AMA_LLM_SEED` — see [agent_framework.md](agent_framework.md#reproducibility-llm-sampling-seed). The judge call stays unseeded (`temperature=0`). (default: 42) |
| `-p, --postprocessing` | Evaluation method: `choice`, `sparql`, `llm_judge`, or `simple` (default: `llm_judge`). `llm_judge` keeps the model judge but applies a deterministic single-number numeric-equivalence guard for harmless decimal formatting differences. |
| `-d, --dataset` | SciQA only: `handcrafted` or `auto` (default: handcrafted) |

**Note:** the direct multi-model batch CLI (`python -m ama_kbqa.benchmark_agents`, used by the frontend and `SOP/running_batch_processing.md`) additionally exposes `--concurrency N` (default 1 = strictly serial): an opt-in question-level parallelism path that spins up `N` fully isolated agents (each its own MCP subprocess) and processes questions concurrently. Orthogonal to the always-on within-question concurrent tool-call execution described in [agent_framework.md](agent_framework.md). Resolution order: CLI flag > `config.toml [benchmark] concurrency` > 1.

### 6. Fewshot Generator (`ama_kbqa/fewshot_generator.py`)

LLM-based generator that analyzes benchmark results to produce fewshot learning material:

**Three Output Types:**

1. **Per-qtype examples** - Tool-trace examples saved to `db/datasets/kqapro/fewshot-examples/<QType>.json`
2. **General guidance** - Cross-type insights saved to `_general.json` (max 10 entries)
3. **Tool tips** - Tool-specific gotchas saved to `_tool_tips.json` (max 20 entries)

**Generator Features:**
- Model, temperature, max_tokens, and context flags driven by `[fewshot_generator]` in `config.toml` (defaults: `deepseek/deepseek-v4-pro`, temp 1.0, max_tokens 16000). Old hardcoded constants removed.
- `include_tool_descriptions` (default `true`): spawns the agent's MCP server, calls `list_tools`, formats via `build_text_mode_tool_catalog`, and injects as `## Available Tools` in the generator prompt. Cached per agent. Falls back to trace extraction then empty string.
- `include_full_conversation` (default `true`): passes the complete untruncated conversation trace. Generator can judge path optimality against the full tool catalog.
- `GENERATOR_SYSTEM_PROMPT` instructs: even on correct answers, check whether a more direct tool existed; surface as `pitfall` or `tool_tip`.
- Analyzes both correct (argumentation_score >= 4) and incorrect answers
- For correct answers: extracts successful patterns and notes inefficiencies
- For incorrect answers: diagnoses mistakes and proposes corrected tool traces
- Deduplicates by question/title/tool+pattern before saving
- Audit log written to `benchmark_results/<YYYY-MM-DD-N>/<agent>/<model>/generated_fewshot.json`
- Enabled via `--generate-fewshot` CLI flag (defaults to `false` in config.toml)
- Runs after llm_judge evaluation completes

**`[fewshot_generator]` config section:**

```toml
[fewshot_generator]
provider = "openrouter"
model = "deepseek/deepseek-v4-pro"
temperature = 1.0
max_tokens = 16000
include_tool_descriptions = true
include_full_conversation = true
max_messages = 20         # used when include_full_conversation = false
max_result_chars = 500    # used when include_full_conversation = false
```

**Post-Hoc Runner (`ama_kbqa/run_fewshot_generator.py`):**

Standalone script that replays the generator over a saved benchmark result directory without re-running the agent. Loads `results.json` + `tool_traces/question_NNN.json`, constructs `ReplayResult` shims, and monkey-patches `fewshot_generator.FEWSHOT_DIR` to a shadow directory before calling `generate_and_save_fewshot_examples`.

```bash
uv run python -m ama_kbqa.run_fewshot_generator \
    --result-dir benchmark_results/<run-name> \
    --agent kqapro \
    --output-dir db/datasets/kqapro/fewshot-examples.generated-YYYY-MM-DD
```

Hard safety guard: aborts if `--output-dir` resolves to the live `FEWSHOT_DIR`. All generator output must be reviewed before promotion to the live fewshot corpus.

**Pydantic Models:**
- `FewshotQTypeExample` - Per-qtype tool-trace examples with lessons/pitfalls
- `FewshotGeneralExample` - Cross-type guidance with applicability tags
- `ToolTip` - Tool-specific problem patterns and guidance
- `FewshotGeneratorOutput` - Combined output with optional fields

**Integration:**
- Agent loads general guidance via `_load_general_guidance()` (top 5 entries)
- Agent loads tool tips via `_load_tool_tips()` (top 10 entries)
- Injected into analysis context via `GENERAL_GUIDANCE_TEMPLATE` and `TOOL_TIPS_TEMPLATE`

**Evaluation pipeline layers (as of 2026-05-09):**

```
benchmark run → llm_judge evaluation → fewshot generator (inline or post-hoc)
                                              ↓
                                     shadow output dir
                                              ↓
                               human review → promote to live fewshot-examples/
```

### 7. Frontend (Streamlit Multi-Page App)

The frontend provides a web-based interface for all major system features. Built with Streamlit, it offers four pages accessible via the sidebar.

**Running the Frontend:**

```bash
streamlit run ama_kbqa/frontend/app.py
```

#### Pages

**1. Chat (`pages/1_Chat.py`)**

Interactive question-answering interface with:
- Agent selector (Orchestrator, KQAPro, SciQA)
- Agent metadata cards showing databases and tool counts
- Suggested example questions per agent
- Token usage display
- Reasoning trace viewer (collapsible)
- Chat history with restart button
- After each `ask()` returns: reads `agent.recorder.to_dicts()` and `agent.journal_snapshots`, mirrors them into `st.session_state["traces"]` (capped at 30 most recent)
- 🔍 Trace and 🕸️ Graph buttons under each assistant message that `st.switch_page` to the Trace Inspector / Graph View with `pinned_trace_id` set

**2. Batch Processing (`pages/2_Batch_Processing.py`)**

Run batch benchmarks through the UI:
- Configuration form (agent, sample size, seed, evaluation method, dataset)
- Few-shot toggle
- Models multi-select (for multi-model benchmarking)
- Advanced options expander (timeout, output directory, resume, export CSV, dry run)
- Live progress output with ANSI color rendering
- Subprocess execution (non-blocking)
- Recent batch results list

**3. Evaluation (`pages/3_Evaluation.py`)**

Results dashboard for batch runs:
- Summary metrics (accuracy, total questions, avg duration, total tokens)
- Accuracy by question type (bar chart)
- Question type distribution (bar chart)
- Duration distribution (histogram)
- Tool call breakdown (bar chart)
- Per-question details table (expandable, shows question, predicted/gold answers, correctness, duration, tokens, tool calls)
- Judgment details viewer (LLM judge reasoning if available)

**4. Settings (`pages/4_Settings.py`)**

Configuration editor for `config.toml`:
- LLM configuration (provider, model, temperature, max_tokens)
- Synthesis configuration (model, temperature, max_tokens)
- Embedding configuration (provider, model, dimensions)
- Search configuration (top_k limits)
- **Retrieval Configuration** — controls the `[retrieval]` config section: hybrid toggle, RRF/DBSF fusion selectbox (shown when hybrid is on), prefetch limit, reranker toggle (disabled with a `uv sync --extra rerank` hint when `sentence_transformers` is not importable), reranker model, rerank candidates. Mutates `edited["retrieval"]`; persisted via the same save flow as all other sections.
- **Model field** (chat/synthesis/judge): for "fetchable" providers (currently KIT) the model is a **dropdown populated live from the provider's `/models` endpoint** (`utils/settings_ui.py::model_field` → `config_editor.fetch_provider_models`, cached 5 min, with a 🔄 refresh button); other providers (OpenRouter, llamacpp) keep a free-text input. A failed fetch (no API key, network/HTTP error) falls back to free text with a warning so the page never blocks.
- Two save modes:
  - **Session only** — `config_editor.apply_to_session` sets `_config_cache` for the running process; no disk write. MCP subprocesses (spawned per question turn) will not see the change until saved to disk.
  - **Save to file** — `config_editor.save_config` writes the full config (including `[retrieval]`) to `config.toml`; creates `.bak` backup. MCP subprocesses pick it up on the next spawn.

**5. Trace Inspector (`pages/5_Trace_Inspector.py`)** ✅ NEW

Langfuse-style hierarchical span view for a completed agent run:
- Trace selector (from `st.session_state["traces"]`) + summary header
- Two-pane layout (`st.columns([0.45, 0.55])`): left = clickable span tree (native `st.button`s styled as dark tree rows via the `st-key-tracetree-*` container, select by websocket rerun — no page reload); right = tabbed detail keyed off span kind (Messages / Args+Result / Attributes / Payload / Raw)
- JSONL export for offline analysis
- Populated after each Chat page `ask()` call via `agent.recorder.to_dicts()`

**6. Graph View (`pages/6_Graph_View.py`)** ✅ NEW

Interactive visualisation of the agent's discovered KG subgraph:
- Trace selector + journal snapshot scrubber (one snapshot per journal-mutating tool call)
- vis-network embedded via CDN through `st.components.v1.html` — no new Python deps
- Wrapped in `@st.fragment` so unrelated Streamlit reruns don't remount the iframe
- Viewport pan/zoom persisted in iframe `localStorage` keyed per `trace_id`
- Toggle for literals + new-since-previous-snapshot highlighting
- Stats row + JSON side panels
- Populated from `agent.journal_snapshots` captured during `ask()`

#### Shared Utilities (`frontend/utils/`)

| Module | Purpose |
|--------|---------|
| `styling.py` | CSS injection, ANSI-to-HTML converter, avatar constants, HTML capture for stdout, kind-coloured span pills |
| `async_helpers.py` | `run_async()` wrapper for running async agent methods in Streamlit |
| `agent_factory.py` | Agent metadata dict, suggestion prompts, `create_agent()` factory function |
| `batch_results_loader.py` | Functions to list and load benchmark result JSON files from `benchmark_results/` |
| `config_editor.py` | Load/save config.toml with backup creation, apply edits to session |
| `trace_render.py` | Pure-Python helpers: `build_tree`, `summarise`, `render_tree_html`, `render_summary_html`, `format_duration` (unit-tested) ✅ NEW |
| `graph_html.py` | `journal_to_graph(state)` and `build_graph_html(nodes, edges, ...)` returning a complete vis-network HTML doc ✅ NEW |

**Key Features:**
- All German labels translated to English
- ANSI color codes rendered as HTML for batch output
- Agent selection persists across chat turns
- Batch runs execute in subprocess with live streaming output
- Settings changes can be previewed in session before committing to file
- Automatic .bak backup created when saving config.toml

---

### 8. CI / Linting (`.github/workflows/ci.yml`, `pyproject.toml`)

**Added 2026-07-18** (commit `9788f47`, landed concurrently with this doc update): a single `lint-and-test` job runs on pushes to `main`/`dev` and on all pull requests:

1. `actions/checkout` + `astral-sh/setup-uv` (cache enabled)
2. `uv sync --group dev` — `pytest`/`ipykernel` moved into a `dev` uv dependency group (previously top-level), `ruff` added as a dependency
3. **Lint:** `uv run ruff check .`
4. **Test:** `uv run pytest -q` — the full 382-test suite is hermetic (fakes/mocks stand in for Qdrant/Virtuoso everywhere); verified to pass with no Docker services running and no `.env`/env vars set. If a future test genuinely needs a live Virtuoso/Qdrant instance, the convention is to mark and deselect it in the CI step rather than adding service containers.

**Ruff config (`pyproject.toml` `[tool.ruff]`):** `target-version = "py312"`; lint rule set restricted to `["E4", "E7", "E9", "F"]` (pycodestyle errors + pyflakes — not the full default set, and no formatting/style rules like line length). `[tool.ruff.lint.per-file-ignores]` suppresses the pre-existing findings in the two large MCP server modules the user has asked not to be touched: `F401`/`F541`/`F841` in `ama_kbqa/server/sciqa_server.py`, and those plus `E402`/`F811`/`E722` in `ama_kbqa/server/kqapro_server.py`; violations there are suppressed rather than fixed in place (`tests/retrieval/test_search.py` also carries an `E402` ignore for a documented import-order workaround). Any new/other file is held to the full selected rule set. The same commit also fixed pre-existing lint findings elsewhere (unused imports/variables, bare excepts, placeholder-less f-strings, one-line compound statements) across several agent/server/frontend files.

---

## Data Flow

```
User Question
     │
     ▼
┌─────────────────────────┐
│   Orchestrator Agent    │  ← Classifies and routes
│   (owns recorder +      │  ← opens "agent_run" root span
│    journal_snapshots)   │
└──────────┬──────────────┘
           │  _delegate(agent_name, query)
           │  shares recorder + parent_span_id
           ▼
┌────────────────────────────────────┐
│  KQAPro Agent / SciQA Agent        │  ← Main KBQA logic
│  (appends to shared recorder)      │
│                                    │
│  ┌──────────────────────────────┐  │
│  │ Pre-Hook  [classify span]    │  │  ← Question classification
│  │ - Qtype                      │  │  ← Entity extraction
│  │ - Strategy / Few-shot        │  │
│  └──────────────────────────────┘  │
│                │                   │
│                ▼                   │
│  ┌──────────────────────────────┐  │
│  │ Tool Loop                    │◄─┼──── MCP Server (stdio)
│  │  [llm_call span per call]    │  │          │
│  │  [tool_call span per tool]   │  │          ├── Qdrant (vector)
│  │  [tool_loop_iter event]      │  │          │
│  │  [journal_refresh event]     │  │          └── Virtuoso (SPARQL)
│  │  After JOURNAL_MUTATING_TOOLS│  │
│  │    → GetJournalStateJSON     │  │  ← snapshot captured (LLM-hidden)
│  │    → journal_snapshots[]     │  │
│  └──────────────────────────────┘  │
│                │                   │
│                ▼                   │
│  ┌──────────────────────────────┐  │
│  │ Post-Hook  [synthesis span]  │  │  ← Synthesis with journal data
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
           │
           ▼
     Final Answer
           │
           ▼  (via background-thread queue + @st.fragment polling)
┌────────────────────────────────────────────────────────────────┐
│  Chat Page  (1_Chat.py)  — background-thread run               │
│                                                                │
│  Worker thread (daemon)          Main Streamlit thread         │
│  ┌──────────────────────┐        ┌──────────────────────────┐  │
│  │ asyncio loop         │        │ @st.fragment(run_every=  │  │
│  │ agent.ask(query)     │        │   0.4)  drain_into()     │  │
│  │   │                  │        │   ├─ SVG node highlight   │  │
│  │   └─ TraceRecorder   │──────► │   ├─ Lifecycle tab live  │  │
│  │     listener push    │queue   │   └─ answer render       │  │
│  └──────────────────────┘        └──────────────────────────┘  │
│                                                                │
│  After run completes:                                          │
│    st.session_state["traces"] ← recorder.to_dicts()           │
│    Lifecycle | Trace | Graph tabs (in-page panel)             │
│      ├─ Lifecycle: inline SVG from lifecycle_svg.py            │
│      ├─ Trace: render_trace_panel() → Trace Inspector         │
│      └─ Graph: render_graph_panel() → vis-network KG view     │
│                                                                │
│  Pages 5 / 6 remain as standalone historical-trace viewers    │
│  (trace selector + sidebar + delegate to same panel helpers)  │
└────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### External APIs

| API | Purpose | Configuration |
|-----|---------|---------------|
| OpenRouter | Cloud LLM access | `OPENROUTER_API_KEY` in `.env` |
| KIT | KIT AI Toolbox | `KIT_API_KEY` in `.env` |

### Database Connections

| Service | Port | Purpose |
|---------|------|---------|
| Virtuoso HTTP | 8890 | SPARQL endpoint |
| Virtuoso SQL | 1111 | Direct SQL access |
| Qdrant | 6333 (default) / **6335 on Hetzner** | Vector database (host port offset on Hetzner to avoid Orca collision) |

### MCP Protocol

Agents communicate with MCP servers via **stdio**:

```python
# Agent connects to MCP server
client = MCPClient(server_path, agent_name)
await client.start()

# Call tools
result = await client.call_tool("FindNode", {"semantic_node_name": "Boston"})
```

---

## Key Algorithms

### 1. Question Classification

Uses LLM with structured JSON output. KQAPro classifies into 10 types (see [kqapro_agent.md](kqapro_agent.md#question-type-classification-10-types)); SciQA classifies into 8 types (see [sciqa_agent.md](sciqa_agent.md#question-type-classification-8-types)). Both use the shared `_extract_json_object` think-prefix/fence-stripping parser (see [agent_framework.md](agent_framework.md#classification-json-parsing-_extract_json_object)).

### 2. Loop Detection (7 Layers)

1. **Identical Calls** - Same tool+params 3x in a row
2. **Oscillation** - A-B-A-B or A-B-C-A-B-C patterns
3. **Tool Spam** - Same tool 5/6 times (different params)
4. **FindResource/FindNode Cap** - 8 calls
5. **RunORKGSPARQL Cap** - 10 calls (SciQA only)
6. **No Progress** - Journal unchanged for 5 iterations (6-branch intervention tree)
7. **Max Tool Calls** - domain-configured hard cap; exits through synthesis, not an error

Full detail: [kqapro_agent.md](kqapro_agent.md#loop-detection-kqapro-exercises-layers-1-4-and-6-7-layer-5-is-sciqa-specific), [sciqa_agent.md](sciqa_agent.md#loop-detection-sciqa-exercises-all-7-layers)

### 3. Hybrid Search

FindNode uses two-phase search:

1. **Exact Filter** - Try exact ID match in Qdrant payload
2. **Vector Search** - Fallback to semantic similarity

### 4. Deterministic Synthesis

Post-agent hook ensures consistent final answers:

1. Fetch complete journal summary
2. Inject synthesis prompt with all data
3. Use separate synthesis LLM configuration
4. Generate grounded answer

---

## Related Documentation

- [Agent System (index)](agent_system.md) - orientation map for the agent-system docs below
- [Agent Framework](agent_framework.md) - `BaseKBQAAgent` shared mechanics (tool gating, synthesis funnel, trace, reset patterns, seed control)
- [Orchestrator Routing](orchestrator_routing.md) - evidence-based routing deep-dive
- [KQAProAgent](kqapro_agent.md) / [SciQAAgent](sciqa_agent.md) - per-agent lifecycle, classification, journal, tools deep-dive
- `database_schema.md` - referenced in older READMEs but file is absent; Qdrant/Virtuoso schema is documented inline in this doc and in the per-agent docs above
- `TOOLS_REFERENCE.md` / `AGENT_ARCHITECTURE.md` - referenced in older docs but removed from the repo root in commit `d197a31` (2026-02-10); their content lives in this doc and the per-agent docs (`agent_framework.md`, `kqapro_agent.md`, `sciqa_agent.md`, `orchestrator_routing.md`) instead

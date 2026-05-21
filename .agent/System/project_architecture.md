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
│   │   ├── types.py            # Response types (EntityMatch, NodeDetails, etc.)
│   │   ├── config.py           # Configuration dataclasses
│   │   ├── state.py            # JournalState and JournalManager
│   │   ├── mcp_client.py       # Shared MCPClient class
│   │   ├── base_agent.py       # BaseKBQAAgent ABC (~600 lines)
│   │   ├── text_tool_calls.py  # Text-mode tool-call shim (minimax-m2.7 compat)
│   │   ├── trace.py            # TraceEvent + TraceRecorder (OTel-shaped, ContextVar nesting)
│   │   └── adapters/           # KG-specific adapters
│   │       ├── base_adapter.py # BaseKGAdapter ABC
│   │       ├── kqapro_adapter.py # KQAPro configuration
│   │       └── sciqa_adapter.py  # SciQA/ORKG configuration
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
│   │   ├── kqapro_server.py    # KQAPro tools (28 tools)
│   │   ├── sciqa_server.py     # SciQA/ORKG tools (27 registered: 4 discovery, 6 retrieval, 12 domain, 1 SPARQL, 1 verification, 2 state, 1 hidden snapshot) ✅ Active
│   │   └── orchestrator_server.py # Routing tools
│   ├── frontend/               # Streamlit multi-page app
│   │   ├── app.py              # Main entry point (page config + sidebar)
│   │   ├── pages/              # Streamlit pages
│   │   │   ├── 1_Chat.py       # Interactive Q&A; background-thread run; live SVG lifecycle; Lifecycle/Trace/Graph tab panel; Simplified view toggle ✅ UPDATED
│   │   │   ├── 2_Batch_Processing.py # Batch runner: live progress, tqdm parsing, console auto-scroll
│   │   │   ├── 3_Evaluation.py # Results dashboard (charts, metrics, per-question details)
│   │   │   ├── 4_Settings.py   # config.toml editor
│   │   │   ├── 5_Trace_Inspector.py  # Thin wrapper: trace selector + render_trace_panel() ✅ UPDATED
│   │   │   └── 6_Graph_View.py       # Thin wrapper: trace selector + render_graph_panel() ✅ UPDATED
│   │   └── utils/              # Shared utilities
│   │       ├── styling.py      # CSS, ansi_to_html, avatars, kind-coloured span pills
│   │       ├── async_helpers.py # run_async() wrapper
│   │       ├── agent_factory.py # Agent creation + metadata
│   │       ├── batch_results_loader.py # Load benchmark result files from benchmark_results/
│   │       ├── config_editor.py # config.toml loading/saving
│   │       ├── trace_render.py  # Pure-Python trace helpers: build_tree, summarise, render_tree_html
│   │       ├── graph_html.py    # journal_to_graph, build_graph_html (vis-network HTML template)
│   │       ├── lifecycle_runner.py # LiveLifecycleState, start_run(), drain_into() — background thread + queue ✅ NEW
│   │       ├── lifecycle_svg.py    # Inline-SVG agent-lifecycle figure (ported from paper TikZ, data-state nodes) ✅ NEW
│   │       ├── lifecycle_mapping.py # SPAN_KIND_TO_NODE, TOOLS_A/TOOLS_B frozensets ✅ NEW
│   │       ├── trace_panel.py   # render_trace_panel() — extracted panel helper shared by Chat + page 5 ✅ NEW
│   │       └── graph_panel.py   # render_graph_panel() — extracted panel helper shared by Chat + page 6 ✅ NEW
│   ├── config.py               # Configuration loader
│   ├── benchmark_agents.py     # Unified batch processing & multi-model benchmarking (includes tool trace export)
│   ├── postprocessing.py       # Postprocessing modes (choice, sparql, llm_judge, simple)
│   ├── utils/                  # Shared utilities
│   │   ├── __init__.py
│   │   └── trace_utils.py      # Tool trace extraction & few-shot export
├── tests/                      # Test suite (176 tests total)
│   ├── framework/              # Framework unit tests (138 tests)
│   │   ├── test_types.py       # Response type tests
│   │   ├── test_config.py      # Configuration tests
│   │   ├── test_state.py       # State management tests
│   │   ├── test_adapters.py    # Adapter tests
│   │   ├── test_trace.py       # TraceRecorder: nesting, contextvar isolation, sync/async parity, JSONL roundtrip; TestRecorderListeners (4 new)
│   │   └── test_trace_render.py # trace_render helpers: tree-building, HTML, journal→graph
│   └── frontend/               # Frontend unit tests (38 tests) ✅ NEW
│       ├── test_lifecycle_svg.py    # SVG generation, node state transitions (12 tests)
│       ├── test_lifecycle_mapping.py # SPAN_KIND_TO_NODE, TOOLS_A/B coverage (19 tests)
│       └── test_lifecycle_runner.py  # start_run/drain_into with stub agent + threading.Event (3 tests)
├── db/                         # Database utilities
│   ├── docker-compose.yml      # Virtuoso + Qdrant + frontend (3 services on Hetzner)
│   ├── populate_vector_db.py   # KQAPro Qdrant initialization
│   ├── populate_sciqa_vectors.py # SciQA Qdrant initialization
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
| **Vector DB** | Qdrant (`qdrant-client>=1.17.0`) | Semantic entity/relation search via `query_points` API |
| **Graph DB** | Virtuoso 7 | RDF triple store, SPARQL queries |
| **LLM Access** | OpenAI SDK | Compatible with OpenRouter, LMStudio, etc. |
| **Embedding** | qwen/qwen3-embedding-8b (4096 dim) | Entity/relation embeddings |
| **Frontend** | Streamlit | Multi-page web UI (Chat, Batch, Evaluation, Settings) |
| **Configuration** | TOML + dotenv | Centralized config management |
| **Containerization** | Docker Compose | Database services |

---

## Core Components

### 0. Generic KBQA Framework (`ama_kbqa/framework/`)

The framework provides abstract base classes that both KQAProAgent and SciQAAgent inherit from. This reduces code duplication by ~70% and ensures consistent behavior.

**Framework Components:**

| Module | Purpose | Lines |
|--------|---------|-------|
| `types.py` | Response dataclasses (EntityMatch, NodeDetails, etc.) | ~280 |
| `config.py` | Configuration classes (NamespaceConfig, KnowledgeGraphConfig) | ~220 |
| `state.py` | JournalState (Pydantic BaseModel, single source of truth with caps/helpers) and JournalManager | ~340 |
| `mcp_client.py` | Shared MCPClient for MCP server communication | ~120 |
| `base_agent.py` | BaseKBQAAgent ABC with full tool-calling loop + span instrumentation | ~600 |
| `text_tool_calls.py` | Text-mode shim for models that can't emit native function calls | ~150 |
| `trace.py` | `TraceEvent` (OTel-shaped dataclass) + `TraceRecorder` (ContextVar nesting, async/sync spans, point-in-time events, JSONL export, `add_listener`/`remove_listener` observer hooks for live streaming) | ~200 |
| `adapters/base_adapter.py` | BaseKGAdapter ABC for KG configuration | ~200 |
| `adapters/kqapro_adapter.py` | KQAPro-specific adapter | ~150 |
| `adapters/sciqa_adapter.py` | SciQA/ORKG-specific adapter | ~180 |

**BaseKBQAAgent provides:**
- MCP client management
- Pre-agent hooks (classification, entity extraction)
- KG-specific exact-attribute constraint hook (`_extract_exact_attribute_constraints`) and pre-analysis constraint injection
- 6-layer loop detection (includes FindResource cap and RunORKGSPARQL cap)
- Scratchpad-enforced tool-calling loop:
  - Forced reflection after each non-journal tool
  - Tool response truncation (2000 chars, except last 2 and journal tools)
  - Journal refresh at index 1 (primacy bias, replacement mode)
- Post-agent synthesis
- Token and tool call tracking
- `soft_reset()` for batch processing
- `self.recorder: TraceRecorder` — OTel-shaped span instrumentation (spans: `agent_run`, `classify`, `fast_path`, `llm_call`, `tool_call`, `synthesis`, `delegate`; events: `tool_loop_iter`, `journal_refresh`, `loop_detected`, `context_trim`, `intervention`)
- `self.journal_snapshots: list` — structured KG snapshots captured after journal-mutating tool calls (bounds extra MCP RPCs to ~mutation count, not per-iteration)
- `parent_recorder` + `parent_span_id` kwargs for sub-agent nesting under Orchestrator

### 1. KQAProAgent (`ama_kbqa/agents/kqapro_agent/agent.py`)

The main KBQA agent that answers questions by inheriting from `BaseKBQAAgent`:

1. **Pre-Agent Hook** - Classifies question type and extracts entities
2. **Iterative Tool Loop** - Calls MCP tools to gather information
3. **Post-Agent Hook** - Synthesizes final answer from journal

**Key Features (inherited from BaseKBQAAgent):**
- Question type classification (10 types: Count, Verify, Select, SelectBetween, SelectAmong, QueryAttr, QueryAttrQualifier, QueryRelation, QueryRelationQualifier, QueryName)
- Multi-layered loop detection (4 detection patterns)
- Automatic journal tracking (visited nodes, found values)
- Separate synthesis model configuration
- Token usage tracking

**File Organization:**
- `agent.py` (~290 lines) - Implements abstract methods, KQAPro-specific logic
- `prompts.py` (~665 lines) - All prompts extracted for maintainability

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
2. **Iterative Tool Loop** - Calls SciQA MCP tools (27 registered tools) to gather research information
3. **Post-Agent Hook** - Synthesizes final answer from journal

**Key Features (inherited from BaseKBQAAgent):**
- Question type classification (8 types: Factoid, Count, List, Boolean, Comparison, Superlative, Aggregation, General)
- Type-specific strategy loading (only the relevant strategy is injected after classification)
- Lean SYSTEM_PROMPT with strategy-specific content in QTYPE_STRATEGIES
- Same loop detection and journal system as KQAPro
- Separate synthesis model configuration
- Token and tool call duration tracking

**File Organization:**
- `agent.py` (~190 lines) - Implements abstract methods, SciQA-specific logic
- `prompts.py` (~890 lines) - ORKG-specific prompts, 8 enriched strategies with few-shot examples (author search, negation, aggregation scoping, energy domain, boolean comparison-embedded values), predicate dictionary, 9 loop recovery entries. Current guidance is evidence-first: high-level aggregation tools before raw SPARQL for supported count/superlative/aggregation shapes.

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

Provides 28 tools for knowledge graph interaction, organized by tier:

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
- `RunSPARQL` - Raw SPARQL queries (stores results in `found_values["sparql_result_N"]` to survive message truncation)
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

Provides 27 registered tools for ORKG knowledge graph interaction: 24 user-visible KB tools, 2 state-management tools, and 1 LLM-hidden journal snapshot tool.

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
- `InspectComparisonSchema` - Compact schema map for Comparison resources. Lists direct contribution predicates and nested contribution -> row-object -> metric paths with labels, counts, sample values, numeric/HAS_VALUE evidence, unit samples when present, and `AggregateComparisonValues` usage hints. Use before choosing predicates or nested row filters for aggregation. ✨ NEW
- `QueryComparisonRows` - Multi-predicate row selector for Comparison contributions. Applies one filter per column/constraint, projects several return predicates, and writes the matched rows to `found_values`; use before raw SPARQL for "column A = X and column B = Y, return metrics C/D/E" questions. ✨ NEW
- `AggregateComparisonValues` - Single SPARQL + Python aggregation over Comparison contributions. Handles HAS_VALUE/label indirection; supports avg/sum/min/max/count/count_distinct/mode_top/all_values plus aliases like frequency/mean/total; optional grouping, pre-filtering, `value_via_group` 2-hop switch, `comparison_ids` CSV for multi-Comparison union mode without a redundant `comparison_id`, `value_parser` for embedded numeric strings, `return_predicate` for companion values on min/max rows, `intermediate_predicate` for nested rows such as contribution → energy source → electricity generation, `intermediate_filter_value` for restricting nested row objects by label/ID, and `value_predicates` for unioning sibling metric predicates even when `value_predicate` is empty. ✨ UPDATED
- `DiagnoseComparisonAggregation` - Wrapped SPARQL diagnostics for denominator/scope ambiguity after schema discovery. Reports total scope contributions, matched value rows, distinct contributions, nested/group populations, row-level numeric summaries, per-contribution sum/mean candidates, and per-intermediate/per-group candidates. Use before answering averages/counts where the wording could mean all rows, per study/contribution, or per category. ✨ NEW
- `FindFrequentValues` - Cross-resource aggregation for global-scope questions ("most popular X", "largest Y across the papers"). No Comparison anchor required. Scope tiers: research_field_id → P30/P31; comparison_ids → VALUES union; default → all compareContribution subjects; `scope="papers"` scans paper contributions. `value_source="subject"` reads values from scoped Paper/Comparison resources themselves, covering paper metadata like top research fields. Supports the same agg, `value_parser`, and `return_predicate` modes as AggregateComparisonValues; writes results to `found_values`; hard cap `limit_subjects=5000`. ✨ UPDATED
- `FindCoAuthors` - Finds co-authors of papers by a seed author (case-insensitive partial match; handles both resource-URI and literal-string author predicates P6/P27); returns co-authors sorted by shared-paper count. Closes Q2 co-author pattern.

**Tier 4 - Raw SPARQL (1 tool):**
- `RunORKGSPARQL` - Raw SPARQL queries (prefixes auto-injected, **capped at 10 calls per question** to prevent runaway SPARQL spirals) ✨ UPDATED

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
    get_virtuoso_endpoint,     # SPARQL endpoint
)
```

**`synthesis_enabled` config key** (`[synthesis]` section in `config.toml`):
- `true` (default) — run the dedicated synthesis LLM call after the tool loop
- `false` — return the agent's own last assistant message directly; saves one LLM call but loses deterministic answer shaping
- Exposed as a toggle in the Settings UI (`pages/4_Settings.py`, "Run synthesis step")

**`auto_inject_journal` config key** (`[agent]` section in `config.toml`):
- `true` (default) — `_run_tool_loop` periodically injects a journal refresh every N iterations and appends an "answer now" prompt whenever the agent calls `GetJournalSummary`
- `false` — both automatic injections are suppressed; `GetJournalSummary` remains available as a tool the agent can call voluntarily
- Exposed as a toggle in the Settings UI (`pages/4_Settings.py`, "Auto-inject journal into context", under Agent Configuration)

### 4. Orchestrator Agent (`ama_kbqa/agents/orchestrator_agent/agent.py`)

Multi-agent router that selects appropriate sub-agents:

- Uses LLM to classify query type
- Routes to KQAProAgent, CodeAgent, MathAgent, etc.
- Falls back to KQAProAgent for knowledge base queries

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
| `--seed` | Random seed for reproducibility (default: 42) |
| `-p, --postprocessing` | Evaluation method: `choice`, `sparql`, `llm_judge`, or `simple` (default: `llm_judge`) |
| `-d, --dataset` | SciQA only: `handcrafted` or `auto` (default: handcrafted) |

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
- Two save modes:
  - **Session only** - Apply changes to current session without writing to disk
  - **Save to file** - Write to config.toml (creates .bak backup automatically)

**5. Trace Inspector (`pages/5_Trace_Inspector.py`)** ✅ NEW

Langfuse-style hierarchical span view for a completed agent run:
- Trace selector (from `st.session_state["traces"]`) + summary header
- Two-pane layout (`st.columns([0.45, 0.55])`): left = hierarchical tree HTML + radio for span selection; right = tabbed detail keyed off span kind (Messages / Args+Result / Attributes / Payload / Raw)
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

Uses LLM with structured JSON output to classify into 9 types:

- Count, Verify, SelectBetween, SelectAmong
- QueryAttr, QueryAttrQualifier
- QueryRelation, QueryRelationQualifier
- QueryName

### 2. Loop Detection (4 Layers)

1. **Identical Calls** - Same tool+params 3x in a row
2. **Oscillation** - A-B-A-B or A-B-C-A-B-C patterns
3. **Tool Spam** - Same tool 5/6 times (different params)
4. **No Progress** - Journal unchanged for 5 iterations

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

- [Agent System](agent_system.md) - Detailed agent architecture
- [Database Schema](database_schema.md) - Qdrant and Virtuoso schemas
- [../TOOLS_REFERENCE.md](../../TOOLS_REFERENCE.md) - Complete tool reference
- [../AGENT_ARCHITECTURE.md](../../AGENT_ARCHITECTURE.md) - Full agent lifecycle

# AMA KBQA Documentation Index

Welcome to the AMA KBQA project documentation. This folder contains all the critical information needed to understand and work with the codebase.

**Project Goal:** Build a multi-agent system that answers natural language questions over **multiple knowledge graphs**, proving generalization across different RDF/SPARQL databases.

**Supported Knowledge Graphs:**
- **KQAPro** - Wikidata-derived factoid Q&A (~47K entities) ✅ Active
- **SciQA/ORKG** - Scientific research papers and contributions (~1.1M triples, 468 Q&A) ✅ Active

---

## Quick Start

1. **New to the project?** Start with [System/project_architecture.md](System/project_architecture.md)
2. **Need to set up databases?** See [SOP/database_setup.md](SOP/database_setup.md) (covers both KQAPro and SciQA)
3. **Want to run benchmarks?** Check [SOP/running_batch_processing.md](SOP/running_batch_processing.md)
4. **Comparing multiple LLM models?** See [Multi-Model Benchmarking](#multi-model-benchmarking) section in SOP
5. **Adding new tools?** Follow [SOP/adding_new_tools.md](SOP/adding_new_tools.md)
6. **Working on SciQA agent?** See [Tasks/sciqa-agent-implementation.md](Tasks/sciqa-agent-implementation.md)
7. **Adding a new knowledge graph?** See [Tasks/generic-framework-implementation.md](Tasks/generic-framework-implementation.md) and [generic-framework.md](../generic-framework.md)

---

## Documentation Index

### System Documentation

Core documentation about the current state of the system.

| Document | Description |
|----------|-------------|
| [project_architecture.md](System/project_architecture.md) | **Start here.** Project structure, tech stack, data flow, integration points |
| [agent_system.md](System/agent_system.md) | Agent lifecycle, classification, loop detection, journal system |
| [database_schema.md](System/database_schema.md) | Qdrant collections, Virtuoso RDF schema, SPARQL patterns |

### Standard Operating Procedures (SOP)

Step-by-step guides for common tasks.

| Document | Description |
|----------|-------------|
| [database_setup.md](SOP/database_setup.md) | Setting up Virtuoso and Qdrant databases |
| [running_batch_processing.md](SOP/running_batch_processing.md) | Running batch benchmarks with LLM judge |
| [adding_new_tools.md](SOP/adding_new_tools.md) | How to add new MCP tools to the server |
| [changing_llm_provider.md](SOP/changing_llm_provider.md) | Configuring different LLM providers |

### Tasks (PRD & Implementation Plans)

Feature requests and implementation tracking.

| Document | Status | Description |
|----------|--------|-------------|
| [generic-framework-implementation.md](Tasks/generic-framework-implementation.md) | ✅ **Complete** | Generic KBQA framework with BaseKBQAAgent, adapters, 97 tests |
| [sciqa-agent-implementation.md](Tasks/sciqa-agent-implementation.md) | ✅ **Complete** | SciQA/ORKG scientific KG agent with 468 ground truth Q&A |
| [scratchpad-enforced-agent-loop.md](Tasks/scratchpad-enforced-agent-loop.md) | ✅ **Complete** | Scratchpad-first agent loop with tool response truncation and journal reflection |

---

## Project Root Documentation

Additional documentation in the project root:

| Document | Description |
|----------|-------------|
| [CLAUDE.md](../CLAUDE.md) | AI assistant instructions and project context |
| [AGENT_ARCHITECTURE.md](../AGENT_ARCHITECTURE.md) | Detailed agent architecture (1000+ lines) |
| [TOOLS_REFERENCE.md](../TOOLS_REFERENCE.md) | Complete reference for KQAPro MCP tools (21 tools) |
| [CONFIG_GUIDE.md](../CONFIG_GUIDE.md) | Configuration guide (if exists) |
| [README.md](../README.md) | Project README with setup instructions |

---

## Key Files Reference

### Configuration

| File | Purpose |
|------|---------|
| `config.toml` | Central configuration (LLM providers, database, search) |
| `.env` | API keys (not committed) |
| `.env_example` | Template for .env file |

### Framework Code (NEW)

| File | Purpose |
|------|---------|
| `ama_kbqa/framework/base_agent.py` | BaseKBQAAgent ABC (~600 lines) |
| `ama_kbqa/framework/mcp_client.py` | Shared MCPClient class (~120 lines) |
| `ama_kbqa/framework/types.py` | Response types (EntityMatch, NodeDetails, etc.) |
| `ama_kbqa/framework/config.py` | Configuration dataclasses |
| `ama_kbqa/framework/state.py` | JournalState and JournalManager |
| `ama_kbqa/framework/adapters/` | KG-specific adapters (KQAPro, SciQA) |
| `tests/framework/` | Framework unit tests (97 tests) |

### Main Code

| File | Purpose |
|------|---------|
| `ama_kbqa/cli.py` | CLI entrypoint (`ama-kbqa` command) |
| `ama_kbqa/benchmark_agents.py` | Unified batch processing & multi-model benchmarking (~1400 lines) |
| `ama_kbqa/postprocessing.py` | PostProcessor class with choice/sparql/llm_judge/simple modes (~850 lines) |
| `ama_kbqa/utils/trace_utils.py` | Tool trace extraction & few-shot export (~330 lines) |
| `ama_kbqa/agents/kqapro_agent/agent.py` | KQAPro agent (inherits BaseKBQAAgent, ~290 lines) |
| `ama_kbqa/agents/kqapro_agent/prompts.py` | KQAPro prompts (~665 lines) |
| `ama_kbqa/agents/sciqa_agent/agent.py` | SciQA agent (inherits BaseKBQAAgent, ~190 lines) |
| `ama_kbqa/agents/sciqa_agent/prompts.py` | SciQA/ORKG prompts (~890 lines) |
| `ama_kbqa/server/kqapro_server.py` | KQAPro MCP server (21 tools) |
| `ama_kbqa/server/sciqa_server.py` | SciQA MCP server (18 tools) |
| `ama_kbqa/config.py` | Configuration loader module |

### Frontend

| File | Purpose |
|------|---------|
| `ama_kbqa/frontend/app.py` | Main Streamlit entry point |
| `ama_kbqa/frontend/pages/1_Chat.py` | Interactive Q&A page with agent selector |
| `ama_kbqa/frontend/pages/2_Batch_Processing.py` | Batch runner with live progress |
| `ama_kbqa/frontend/pages/3_Evaluation.py` | Results dashboard (charts, metrics) |
| `ama_kbqa/frontend/pages/4_Settings.py` | config.toml editor |
| `ama_kbqa/frontend/utils/styling.py` | CSS, ANSI-to-HTML, avatars |
| `ama_kbqa/frontend/utils/agent_factory.py` | Agent creation + metadata |
| `ama_kbqa/frontend/utils/batch_results_loader.py` | Load batch result files |
| `ama_kbqa/frontend/utils/config_editor.py` | config.toml loading/saving |

### Database

| File | Purpose |
|------|---------|
| `db/docker-compose.yml` | Virtuoso + Qdrant containers |
| `db/populate_vector_db.py` | KQAPro Qdrant initialization |
| `db/populate_sciqa_vectors.py` | SciQA/ORKG Qdrant initialization |
| `db/generate_question_datasets.py` | Generate reproducible questionnaire JSON files |
| `db/datasets/kqapro/convert_kb_to_nt.py` | KQAPro JSON to N-Triples converter |
| `db/datasets/SciQA/` | SciQA/ORKG dataset files |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Query                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│                  (Routes to appropriate KG)                  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   KQAPro Agent    │ │   SciQA Agent     │ │   Future Agents   │
│   (Factoid Q&A)   │ │   (Scientific)    │ │   (Extensible)    │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                 │                       │
            ▼                 ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Shared Infrastructure                     │
│  ┌─────────────────────┐     ┌─────────────────────┐        │
│  │      Qdrant         │     │      Virtuoso       │        │
│  │  - kqapro-entities  │     │  - KQAPro graph     │        │
│  │  - kqapro-relations │     │  - SciQA graph      │        │
│  │  - sciqa-entities   │     │  - Future graphs    │        │
│  │  - sciqa-relations  │     │                     │        │
│  └─────────────────────┘     └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## Common Tasks

### Setup New Development Environment

1. Clone repository
2. `pip install -e .` (or `uv pip install -e .`)
3. Copy `.env_example` to `.env`, add API keys
4. Follow [SOP/database_setup.md](SOP/database_setup.md)

### Using the CLI (Recommended)

After installation, the `ama-kbqa` command is available globally:

```bash
# Ask a question using the orchestrator (default)
ama-kbqa ask "Who directed Inception?"

# Ask using the KQAPro agent directly (bypasses orchestrator)
ama-kbqa ask --subagent kqapro "Who directed Inception?"
ama-kbqa ask -s kqapro "How many films did Christopher Nolan direct?"

# Ask using the SciQA agent for scientific questions
ama-kbqa ask --subagent sciqa "What papers address text classification?"
ama-kbqa ask -s sciqa "Who are the authors of papers on machine learning?"

# Run KQAPro benchmarking on validation dataset
ama-kbqa benchmark --subagent kqapro -n 10 --seed 42
ama-kbqa benchmark -s kqapro -n 50 --seed 123 --postprocessing llm_judge

# Run SciQA benchmarking
ama-kbqa benchmark --subagent sciqa -n 10 --seed 42 --dataset handcrafted
ama-kbqa benchmark -s sciqa -n 50 --dataset auto --postprocessing simple

# Get help
ama-kbqa --help
ama-kbqa ask --help
ama-kbqa benchmark --help
```

**CLI Commands:**

| Command | Description |
|---------|-------------|
| `ask` | Ask a question (uses orchestrator by default, or specific subagent with `-s`) |
| `benchmark` | Run benchmarking with configurable sample size, seed, and postprocessing method |

**Ask Options:**
- `--subagent, -s`: Subagent to use (`kqapro` or `sciqa`)

**Benchmark Options:**
- `-n, --n_questions`: Number of questions to sample (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)
- `-p, --postprocessing`: Evaluation method: `choice`, `sparql`, `llm_judge`, or `simple` (default: `llm_judge`)
- `-d, --dataset`: SciQA only - `handcrafted` or `auto` (default: handcrafted)

### Using the Frontend (Streamlit Web UI)

The frontend provides a visual interface for all major features:

```bash
streamlit run ama_kbqa/frontend/app.py
```

The app opens in your browser with four pages:

1. **Chat** - Ask questions to Orchestrator, KQAPro, or SciQA agents
2. **Batch Processing** - Configure and run batch benchmarks with live progress
3. **Evaluation** - View results dashboards with charts and per-question details
4. **Settings** - Edit config.toml (LLM provider, models, search parameters)

**Features:**
- Agent selector with metadata cards
- Example question suggestions per agent
- Token usage and reasoning trace display
- Live batch processing output with ANSI colors
- Results dashboard with accuracy charts
- Config editor with session-only or file save modes (auto .bak backup)

### Run Single Question (Programmatic)

```python
import asyncio
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent

async def main():
    agent = KQAProAgent()
    try:
        answer = await agent.ask("Who directed Inception?")
        print(answer)

        # Optional: Get tool call duration summary
        summary = agent.get_tool_call_summary()
        print(f"Tool calls: {summary['total_calls']}, Duration: {summary['total_duration_seconds']}s")
    finally:
        await agent.close()  # Always close MCP connection

asyncio.run(main())
```

### Run Batch Benchmark (Programmatic)

```bash
# Generate reproducible questionnaire files first (recommended)
python db/generate_question_datasets.py --dataset both --n 100 --seed 42

# Run with pre-generated questionnaire (single model)
python -m ama_kbqa.benchmark_agents --agents kqapro --questionnaire db/kqapro_questionnaire.json
python -m ama_kbqa.benchmark_agents --agents sciqa --questionnaire db/sciqa_questionnaire.json

# Or run with on-the-fly sampling
python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 10 --seed 42
```

### Multi-Model Benchmarking

```bash
# Preview what would run (dry-run)
python -m ama_kbqa.benchmark_agents --dry-run

# Test single model on single agent
python -m ama_kbqa.benchmark_agents --models minimax-m2.1 --agents kqapro --n-questions 3

# Full benchmark with selected models on both agents
python -m ama_kbqa.benchmark_agents --models minimax-m2.1 glm-4.7 --export-csv

# Resume interrupted benchmark
python -m ama_kbqa.benchmark_agents --resume
```

### Change LLM Model

Edit `config.toml`:
```toml
[openrouter]
chat_model = "your/model-name"
```

---

## Getting Help

- **Technical issues:** Check troubleshooting sections in relevant SOP
- **Architecture questions:** See [System/project_architecture.md](System/project_architecture.md)
- **Tool behavior:** See [TOOLS_REFERENCE.md](../TOOLS_REFERENCE.md)
- **Agent behavior:** See [AGENT_ARCHITECTURE.md](../AGENT_ARCHITECTURE.md)

---

## Contributing

When updating documentation:

1. Keep documents focused on single topics
2. Update this README index when adding new documents
3. Cross-reference related documents with "Related Documentation" sections
4. Use consistent formatting and structure

---

## Recent Changes

- **2026-02-12**: ✅ **Frontend Batch Processing UI Fixes**
  - Fixed "missing ScriptRunContext" warning by moving background thread state to plain shared dict (synced on main thread)
  - Fixed progress bar parsing with improved regex patterns: `(\d+)%\|` for percentage and `\|\s*(\d+)/(\d+)\s*\[` for fraction
  - Fixed garbled tqdm output by adding `_resolve_cr()` function to collapse carriage-return overwrites to latest value
  - Added console auto-scroll JS snippet to pin output to bottom during batch runs
  - Updated [ama_kbqa/frontend/pages/2_Batch_Processing.py](ama_kbqa/frontend/pages/2_Batch_Processing.py)
- **2026-02-12**: ✅ **Batch Processing Refactor**
  - Unified batch processing: merged three scripts (kqapro_agent/batch_runner.py, sciqa_agent/batch_runner.py, benchmark_agents.py) into single `ama_kbqa/benchmark_agents.py`
  - Created `ama_kbqa/postprocessing.py` with PostProcessor class (choice/sparql/llm_judge/simple modes)
  - Created `ama_kbqa/utils/trace_utils.py` for tool trace extraction and few-shot export
  - Deleted old batch_runner.py files from agent directories
  - Frontend updated: batch_results_loader.py now scans `benchmark_results/<timestamp>/<agent>/<model>/` directories
  - Batch Processing page now calls ama_kbqa.benchmark_agents with unified CLI args (--agents, --n-questions, --postprocessing, --seed, --stratified, --questionnaire, --dataset)
  - Evaluation page handles superset summary.json schema with dual field names (e.g., accuracy/accuracy_rate)
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-12**: ✅ **Code Cleanup**
  - Removed unused `ama_kbqa/agents/placeholder_agent/` directory (template agent)
  - Removed duplicate `ama_kbqa/agents/placeholder_agent copy/` directory
  - Removed unused `ama_kbqa/server/subagent_server.py` (prototype server)
  - Removed superseded `ama_kbqa/agents/kqapro_agent/prompt_alternative.txt`
  - Fixed triple-underscore typo: `ama_kbqa/___init__.py` → `ama_kbqa/__init__.py`
  - Updated [System/project_architecture.md](System/project_architecture.md)
- **2026-02-09**: ✅ **SciQA Agent Accuracy Improvements**
  - **FindAuthorPapers**: Added UNION clause for string literal authors (papers storing authors as plain strings `orkgp:P27 "Kurt Thomas"` are now matched via `isLiteral()` filter, in addition to resource-URI author matching)
  - **GetComparisonContributions**: Added automatic 4-hop value resolution via `OPTIONAL { ?value orkgp:HAS_VALUE ?nestedValue }` - response now includes `nested_value` field when present (Comparison → Contribution → intermediate_resource → HAS_VALUE → actual_value)
  - **Prompt enhancements**: Added boolean encoding guidance (ORKG "T"/"F" convention), comparison verification guidance, strengthened minimum-effort enforcement with WARNING block, multi-hop SPARQL patterns for nested HAS_VALUE queries, reverse-link boolean few-shot example
  - Updated [System/agent_system.md](System/agent_system.md) and [System/database_schema.md](System/database_schema.md)
- **2026-02-09**: ✅ **Multi-Page Frontend Refactor**
  - Upgraded single-page German Streamlit chat UI to multi-page English app with 4 pages:
    - **Chat** (pages/1_Chat.py) - Agent selector (Orchestrator/KQAPro/SciQA), English labels, token display
    - **Batch Processing** (pages/2_Batch_Processing.py) - Config form + subprocess runner with live progress
    - **Evaluation** (pages/3_Evaluation.py) - Results dashboard with charts (accuracy, duration, tool calls)
    - **Settings** (pages/4_Settings.py) - config.toml editor (session-only or file save with .bak backup)
  - Created shared utilities layer (frontend/utils/):
    - `styling.py` - Shared CSS, ansi_to_html(), avatars, HTML capture
    - `async_helpers.py` - run_async() wrapper
    - `agent_factory.py` - AGENT_INFO dict, create_agent() factory, suggestion prompts
    - `batch_results_loader.py` - list_batches(), load_summary(), load_results(), load_judgments()
    - `config_editor.py` - load_config_raw(), save_config() with .bak backup, apply_to_session()
  - app.py reduced to minimal entry point (page config + sidebar branding + CSS injection)
  - Frontend now exposes batch processing, evaluation dashboards, multi-agent selection, and settings editing that were previously CLI-only
  - Updated [System/project_architecture.md](System/project_architecture.md)
- **2026-02-09**: ✅ **SciQA Server SPARQL Wrapping Fix**
  - Fixed `RunORKGSPARQL` auto-wrapping behavior to correctly handle solution modifiers
  - Problem: LIMIT, ORDER BY, OFFSET, GROUP BY clauses were trapped inside GRAPH block (invalid SPARQL)
  - Solution: Extract trailing modifiers before wrapping with GRAPH, re-append outside the block (per SPARQL spec)
  - Merged duplicate if/elif branches for WHERE and ASK queries into single condition
  - Updated RunORKGSPARQL docstring with auto-wrapping behavior and modifier extraction details
  - No behavior change for queries that already contain explicit GRAPH clauses
- **2026-02-09**: ✅ **Few-Shot Redesign: Tool-Trace Format**
  - Replaced judge-evaluation few-shot examples with abbreviated tool-usage traces (question → tool sequence → answer)
  - Old format stored LLM judge reasoning ("The predicted answer matches...") which diluted attention from QTYPE_STRATEGIES and hurt performance
  - New format shows which tools to call and in what order, directly actionable for the agent
  - Added `extract_tool_trace()`, `abbreviate_tool_args()`, `abbreviate_tool_result()` to batch_runner.py
  - Renamed `export_fewshot_examples_from_judgments()` → `export_fewshot_examples_from_traces()` (filters by correctness + efficiency, not judge score)
  - Added `use_fewshot` parameter to BaseKBQAAgent, KQAProAgent, SciQAAgent, and benchmark_agents.py
  - Added `--no-fewshot` CLI flag to both batch runners and benchmark script for ablation studies
  - Reduced max examples per type from 10 → 3 (loaded) / 5 (stored), sorted by efficiency
  - Cleared all old judge-evaluation examples from `db/datasets/kqapro/fewshot-examples/*.json`
  - Bonus: Added `ManageJournal("clear")` action, `VerifyNumericCondition` now accepts int/float inputs
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) and [System/agent_system.md](System/agent_system.md)
- **2026-02-08**: ✅ **SciQA Agent Bug Fixes and Loop Prevention**
  - **New tool**: Added `FindAuthorPapers` to sciqa_server.py (SPARQL-based author name search, better than vector search for proper nouns)
  - **Tool improvements**:
    - `GetComparisonContributions` now stores values in journal's `found_values` (was missing journal update)
    - `GetRelationTargets` added reverse lookup fallback when direct query returns 0 results (finds resources that reference the target)
  - **RunORKGSPARQL call cap**: Limited to 10 calls per question (configurable via `sparql_cap` in domain_settings) to prevent runaway SPARQL spirals
  - **Few-shot examples**: Enhanced with examples for author search, negation queries, aggregation scoping, energy domain distinctions, and boolean comparison-embedded values
  - **Loop recovery guidance**: Improved RunORKGSPARQL loop guidance to mention the 10-call cap
  - Updated [System/agent_system.md](System/agent_system.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-06**: ✅ **Agent Behavior Enhancements**
  - Increased tool response truncation limit (1000 → 2000 chars) to preserve property lists from GetNodeSummary
  - Added completion gates to SelectBetween and Count strategies (prevents premature termination)
  - Added anti-premature-termination rule #7 to KQAPro system prompt
  - Fixed few-shot example loading: added missing "Select" and "Query" qtypes to loading list
  - Populated SelectBetween.json with 3 worked examples (was empty)
  - Updated [System/agent_system.md](System/agent_system.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-06**: ✅ **Scratchpad-Enforced Agent Loop**
  - Implemented forced reflection after each non-journal tool call
  - Added tool response truncation (1000 chars, preserve last 2 full responses)
  - Journal refresh now replaces at index 1 (primacy bias, single state in history)
  - Added `_force_journal_reflection()`, `_truncate_tool_response()`, `_compact_message_history()` to base_agent.py
  - Constants: `TOOL_RESPONSE_MAX_CHARS=1000`, `PRESERVE_FULL_RESPONSES_COUNT=2`, `NEVER_TRUNCATE_TOOLS`, `JOURNAL_REFRESH_INDEX=1`
  - Reflection uses text-only LLM call to extract LEARNED/PLAN/NEXT, stores in journal
  - Updated system prompts in both KQAPro and SciQA agents with SCRATCHPAD-FIRST WORKFLOW section
  - Updated [System/agent_system.md](System/agent_system.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-05**: ✅ **SciQA Prompts Refactor (Type-Specific Strategy Loading)**
  - Slimmed SYSTEM_PROMPT (~193 → ~165 lines) by moving strategy-specific content out
  - Removed COMPARISON-BASED QUESTIONS section, nested HAS_VALUE pattern, and SPARQL TIP from SYSTEM_PROMPT
  - Enriched all 8 QTYPE_STRATEGIES entries with moved content (comparison patterns, HAS_VALUE, SPARQL tips)
  - Only the relevant strategy is now loaded after classification (matching KQAPro pattern)
  - SYSTEM_PROMPT retains only general-purpose content: rules, schema, tools, predicates
  - Updated [System/agent_system.md](System/agent_system.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-05**: ✅ **SciQA Agent Enhancement (Accuracy Improvement)**
  - Added 5 new tools to `sciqa_server.py` (13 → 18 tools):
    - `VerifyNumericCondition` - Deterministic math/date comparison (KG-agnostic)
    - `GetResourceSummary` - All-in-one resource exploration (all predicates in one call)
    - `FindByPredicateValue` - Reverse lookup by predicate value (exact/contains/greater/less)
    - `CompareResources` - Compare predicate across multiple resources (sorted)
    - `FollowRelationPath` - Multi-hop relation navigation in one SPARQL call
  - Added 3 new Pydantic models: `NumericComparisonResponse`, `ComparisonResult`, `CompareResourcesResponse`
  - Expanded system prompt (~85 → ~160 lines, ~1700 tokens) with:
    - Schema introspection guidance
    - Constraint verification rule
    - One-hop inference exception
    - 5-tier tool organization
    - 10-step execution strategy
    - Full ORKG Predicate Reference dictionary (core + domain-specific predicates)
  - Updated all 6 QTYPE_STRATEGIES with decision trees and new tool references
  - Expanded TOOL_LOOP_GUIDANCE (4 → 9 entries covering all tools)
  - Updated [System/agent_system.md](System/agent_system.md) and [System/project_architecture.md](System/project_architecture.md)
- **2026-02-05**: ✅ **Added KIT Direct Benchmark Model**
  - New benchmark models using KIT endpoints directly (bypass OpenRouter):
    - `gpt-oss-120b-kit` (model_id: `gpt-oss:120b`) — KIT Ollama at `/ollama/v1`
    - `qwen3-vl-235b-kit` (model_id: `qwen3-vl:235b-a22b-instruct`) — KIT Ollama at `/ollama/v1`
    - `gpt-4.1-mini-kit` (model_id: `azure.gpt-4.1-mini`) — KIT AI Toolbox at `/api/v1`
  - KIT Ollama models use `KIT_OLLAMA_TOKEN`, AI Toolbox models use `AIFB_API_KEY`
  - Note: KIT's `/ollama/api` is native Ollama (NOT OpenAI-compatible); `/ollama/v1` is the OpenAI-compatible endpoint
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) with new model
- **2026-02-05**: ✅ **Enhanced Multi-Model Benchmarking**
  - Added 2 new models: `qwen3-32b` (qwen/qwen3-32b:nitro), `nemotron-3-nano-30b` (nvidia/nemotron-3-nano-30b-a3b:nitro)
  - Updated `gpt-oss-120b` to use `openai/gpt-oss-120b:nitro` model ID
  - Console now displays both gold and predicted answers for each question
  - Summary JSON includes new `avg_tokens` metric (average tokens per question)
  - LLM judge now uses **JSON mode** for reliable response parsing with automatic fallback
  - Stricter judge prompt to reduce false positives (marks INCORRECT when in doubt)
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) with new models and features
- **2026-02-05**: ✅ **Added AIFB LLM Provider**
  - New provider "aifb" using KIT AI Toolbox API at `https://ki-toolbox.scc.kit.edu/api/v1`
  - Default model: `azure.gpt-4.1-mini`
  - Requires `AIFB_API_KEY` environment variable
  - Updated [SOP/changing_llm_provider.md](SOP/changing_llm_provider.md) with AIFB configuration
- **2026-02-04**: ✅ **LLM-as-Judge Evaluation for Multi-Model Benchmarking**
  - Updated `benchmark_agents.py` to use DeepSeek v3.2 as an LLM judge for answer evaluation
  - Replaces rule-based string matching with semantic equivalence checking
  - Judge token usage is NOT counted towards benchmarked model's token count
  - Falls back to string matching if judge API unavailable
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) with evaluation details
- **2026-02-02**: ✅ **Multi-Model Benchmarking Script Added**
  - Created `ama_kbqa/benchmark_agents.py` (~750 lines) for comparing multiple LLM models
  - Tests 10 models: minimax-m2.1, glm-4.7, kimi-k2.5, deepseek-v3.2, gpt-oss-120b, qwen3-32b, nemotron-3-nano-30b (via OpenRouter), gpt-oss-120b-kit, qwen3-vl-235b-kit, gpt-4.1-mini-kit (via KIT direct)
  - Benchmarks both KQAPro and SciQA agents using pre-generated questionnaires
  - Features: progress bars, resume capability, error recovery, cost estimation, CSV export
  - Generates leaderboards by accuracy, speed, and cost-efficiency
  - Dynamically overrides config cache to switch models at runtime
  - Updated [SOP/running_batch_processing.md](SOP/running_batch_processing.md) with multi-model section
  - Updated project README.md with usage documentation
- **2026-02-01**: ✅ **Question Dataset Generator Added**
  - Created `db/generate_question_datasets.py` for reproducible questionnaire generation
  - Supports both KQAPro and SciQA datasets with configurable sampling
  - Generates standardized JSON files with metadata (seed, source, timestamp)
  - Updated batch runners with `--questionnaire` argument for loading pre-generated files
  - SciQA generator supports handcrafted, autogenerated, or both source types
- **2026-02-01**: ✅ **Generic KBQA Framework Implementation Complete**
  - Created `ama_kbqa/framework/` package with:
    - `base_agent.py` - BaseKBQAAgent ABC (~600 lines)
    - `mcp_client.py` - Shared MCPClient class (~120 lines)
    - `types.py` - Response types (EntityMatch, NodeDetails, etc.)
    - `config.py` - Configuration dataclasses
    - `state.py` - JournalState and JournalManager
    - `adapters/` - BaseKGAdapter, KQAProAdapter, SciQAAdapter
  - Refactored both agents to inherit from BaseKBQAAgent:
    - KQAProAgent: ~1,118 → ~290 lines (**74% reduction**)
    - SciQAAgent: ~810 → ~190 lines (**77% reduction**)
  - Created 97 unit tests in `tests/framework/`
- **2026-01-27**: Generic Framework Task Created
  - Created [Tasks/generic-framework-implementation.md](Tasks/generic-framework-implementation.md)
  - Defines base classes, configuration layer, and adapter pattern for any RDF knowledge graph
- **2026-01-27**: **SciQA Agent Implementation Complete**
  - Created `ama_kbqa/agents/sciqa_agent/` with agent.py, prompts.py, batch_runner.py
  - Created `ama_kbqa/server/sciqa_server.py` with 13 ORKG tools
  - Updated CLI to support `sciqa` subagent and `--dataset` option
  - Added SciQA config functions to `config.py`
- **2026-01-17**: SciQA/ORKG fully loaded into Virtuoso (`http://sciqa.org/kg`) with ~1.1M triples
- **2026-01-17**: Updated vector population scripts with overwrite protection (prompts before overwriting existing collections)
- **2026-01-17**: Changed `populate_sciqa_vectors.py` to use hardcoded config instead of CLI args
- **2026-01-17**: Added SciQA/ORKG as second knowledge graph to prove multi-KG generalization
- **2026-01-17**: Created `db/populate_sciqa_vectors.py` for SciQA vector population

---

*Last updated: February 5, 2026 (SciQA Prompts Refactor - type-specific strategy loading)*


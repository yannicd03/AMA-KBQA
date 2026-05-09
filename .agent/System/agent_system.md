# Agent System Documentation

## Overview

The AMA KBQA system uses a multi-agent architecture with:

1. **Orchestrator Agent** - Routes queries to appropriate sub-agents
2. **KQAPro Agent** - KBQA agent for general knowledge (Wikidata-derived) ✅ Active
3. **SciQA Agent** - KBQA agent for scientific research (ORKG) ✅ Active
4. **Placeholder Agents** - Domain-specific fallback agents (code, math)

All agents communicate with their MCP servers via **stdio protocol**.

---

## Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│   ama_kbqa/agents/orchestrator_agent/agent.py               │
│                                                              │
│   - Routes queries based on LLM classification              │
│   - Falls back to KQAProAgent for knowledge queries         │
│   - Uses orchestrator_server.py for routing tools           │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  KQAPro Agent   │  │   SciQA Agent   │  │   Code Agent    │
│  (General KB)   │  │  (Scientific)   │  │  (Placeholder)  │
│                 │  │                 │  │                 │
│  Wikidata-like  │  │  ORKG Papers,   │  │  Python, Algo   │
│  Factoid Q&A    │  │  Authors, etc.  │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                     │
        ▼                     ▼
┌─────────────────┐  ┌─────────────────┐
│ kqapro_server   │  │ sciqa_server    │
│ (26 tools)      │  │ (20 tools)      │
└─────────────────┘  └─────────────────┘
```

---

## Generic KBQA Framework

Both KQAProAgent and SciQAAgent inherit from `BaseKBQAAgent` in the framework package. This provides:

**Framework Files (`ama_kbqa/framework/`):**

| File | Contents | Line Count |
|------|----------|------------|
| `base_agent.py` | BaseKBQAAgent ABC with full agent lifecycle (includes Detection 5: RunORKGSPARQL cap); text-tool-call init + loop hooks; `_extract_json_object` static helper | ~600 |
| `mcp_client.py` | Shared MCPClient class | ~120 |
| `types.py` | Response types (EntityMatch, NodeDetails, etc.) | ~280 |
| `config.py` | Configuration dataclasses | ~220 |
| `state.py` | JournalState (Pydantic BaseModel, single source of truth) and JournalManager — caps, helpers, `to_str()`/`to_summary_str()` | ~340 |
| `text_tool_calls.py` | Text-mode tool-call shim: `needs_text_tool_calls`, `build_text_mode_tool_catalog`, `parse_text_tool_calls`, `TEXT_TOOL_CALL_INSTRUCTION` | ~150 |
| `adapters/base_adapter.py` | BaseKGAdapter ABC | ~200 |
| `adapters/kqapro_adapter.py` | KQAPro-specific config | ~150 |
| `adapters/sciqa_adapter.py` | SciQA-specific config (includes sparql_cap: 10) | ~180 |

**BaseKBQAAgent provides:**
- Abstract methods: `get_config()`, `get_mcp_server_path()`
- Template methods: `_get_system_prompt()`, `_classify_question()`, `_extract_entities()`
- Concrete methods: `ask()`, `_run_tool_loop()`, `_detect_loops()`, `reset()`, `soft_reset()`, `close()`

### Classification JSON Parsing (`_extract_json_object`)

`BaseKBQAAgent._extract_json_object(content: str) -> Optional[Dict]` is a static helper used by both `_classify_question` and SciQAAgent's equivalent. It handles LLM responses that include reasoning prefixes or markdown fences around the JSON payload:

1. Strip `<think>...</think>` blocks via `re.sub(..., flags=re.DOTALL)` — minimax-m2.7 emits these even with `response_format=json_object`.
2. Strip markdown code fences.
3. `json.loads()` on the cleaned string (fast path).
4. Brace-balanced fallback: scan from the first `{`, count depth, extract the balanced block, parse.

The classifier `max_tokens` was also bumped from 300 → 1500 to give models with think-prefix room to complete both the reasoning block and the JSON output.

See `Decisions/classifier-think-prefix-fix.md` for the full analysis and empirical context.

### Text-Tool-Call Mode

Some models (e.g., `minimax-m2.7`) cannot emit OpenAI-structured `tool_calls` — they output prose or `<tool_call>` XML instead. The framework handles this transparently:

1. **Detection:** `text_tool_calls.needs_text_tool_calls(model_name)` — substring match; currently covers `minimax-m2.7`.
2. **System prompt injection (init):** Two extra messages prepended: `TEXT_TOOL_CALL_INSTRUCTION` (mandates `<tool_call>{"name":..., "arguments":...}</tool_call>` format) + a plain-text tool catalog from `build_text_mode_tool_catalog`.
3. **API call:** `tools=None` — no native function-call schema is sent to the endpoint.
4. **Response parsing (loop hook):** If `message.tool_calls` is empty but content contains `<tool_call>` blocks, `parse_text_tool_calls` constructs synthetic OpenAI-shaped tool_calls; the agent loop continues unchanged.
5. **Tool results:** Appended as `role=user` prose wrapped in `<tool_result name="X">...</tool_result>` (not `role=tool`, which these models were not trained on).
6. **Truncated-block retry:** `has_truncated_tool_call(content)` detects a `<tool_call>` opener with no parseable closing block (model stopped mid-emission). On hit: the broken assistant turn is persisted, a corrective user message is injected ("re-emit as one complete block"), and the loop continues. Reuses the `zero_tool_call_retry_max` budget. Trigger conditions for the "no tool calls" retry branch: (a) zero tool calls, no opener → zero-tool-call (RULE 0) retry; (b) zero tool calls, opener present but unparseable → truncated-block retry.

See `Decisions/text-mode-tool-calls.md` for the failure-mode analysis and rationale. See `Decisions/truncated-tool-call-retry.md` for the truncated-block retry ADR.

---

## KQAProAgent Deep Dive

**Files:**
- `ama_kbqa/agents/kqapro_agent/agent.py` - Inherits BaseKBQAAgent (~290 lines)
- `ama_kbqa/agents/kqapro_agent/prompts.py` - All prompts (~665 lines)

### File Organization

The KQAProAgent inherits from BaseKBQAAgent and implements KQAPro-specific methods:

| File | Contents | Line Count |
|------|----------|------------|
| `agent.py` | KQAProAgent class, overrides for classification/extraction | ~290 |
| `prompts.py` | All prompt strings and templates | ~665 |

**Prompts Module (`prompts.py`) exports:**
- `QTYPE_STRATEGIES` - Dict of 10 question-type-specific strategies (includes completion gates for Count and SelectBetween, OR/UNION counting section for Count with genericized SPARQL UNION pattern using placeholders, prepositional phrase note in QueryAttr, qualifier selection guidance in QueryRelationQualifier step 3 and QueryAttrQualifier step 4, prepositional phrase cross-reference in Query step 2b, FilterEntities references in Count/QueryName, QualifierFilter references in QueryAttrQualifier/QueryRelationQualifier, VerifyString reference in Verify)
- `SYSTEM_PROMPT` - Main agent system prompt (includes 8 critical rules: anti-premature-termination rule, prepositional phrase disambiguation rule #7, DO NOT BACKTRACK journal confidence rule at lines 300-301; tool tier listing updated to include T1.5 Filtering tier and T4 Verify tier)
- `CLASSIFICATION_PROMPT_TEMPLATE` - Question classification (uses `{question}`)
- `ENTITY_EXTRACTION_PROMPT` - Entity/relation extraction
- `ANALYSIS_CONTEXT_TEMPLATE` - Pre-analysis context (uses `{qtype}`, `{formatted_entities}`, etc.)
- `FEWSHOT_EXAMPLES_TEMPLATE` - Few-shot examples section (loads all 10 qtypes: Count, Verify, Select, SelectBetween, SelectAmong, QueryAttr, QueryAttrQualifier, QueryRelation, QueryRelationQualifier, QueryName, Query)
- `GENERAL_GUIDANCE_TEMPLATE` - Cross-type insights from `_general.json` (top 5 entries: verify constraints, trust KB data, match qualifier, use FindByAttribute, use SPARQL)
- `TOOL_TIPS_TEMPLATE` - Tool-specific tips from `_tool_tips.json` (top 10 entries; GetEdgeQualifiers entry updated 2026-02-13: "For what" marked AMBIGUOUS - must check both for_work AND ceremony)
- `ANALYSIS_CONTEXT_SUFFIX` - Closing text for analysis
- `JOURNAL_REFRESH_TEMPLATE` - Periodic memory refresh (uses `{iteration_count}`, `{journal_refresh}`)
- `NO_PROGRESS_TEMPLATE` - No progress intervention
- `SYNTHESIS_PROMPT_TEMPLATE` - Final synthesis (uses `{journal_summary}`, `{query}`)
- `JOURNAL_SUMMARY_ANSWER_PROMPT` - Force answer after GetJournalSummary
- `TOOL_LOOP_GUIDANCE` - Dict of tool-specific recovery guidance
- `GENERIC_LOOP_GUIDANCE` - Fallback recovery guidance
- `LOOP_INTERVENTION_TEMPLATE` - Loop detection message

### Agent Lifecycle

```
┌────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
│    - Load chat LLM client from config.toml                     │
│    - Load synthesis LLM client from config.toml                │
│    - Initialize MCP server connection                          │
│    - Load system prompt with KBQA guidelines                   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. PRE-AGENT HOOK (Deterministic Classification)              │
│    - Classify question type (9 types)                          │
│    - Extract entities and relations                            │
│    - Load question-type-specific strategy                      │
│    - Load tool-trace few-shot examples (if enabled)            │
│    - Load general guidance from _general.json (top 5)          │
│    - Load tool tips from _tool_tips.json (top 10)              │
│    - Inject pre-analysis context into message history          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. MAIN AGENT LOOP (Scratchpad-Enforced)                      │
│    REPEAT until answer found OR max iterations (50):           │
│     - Call LLM with tools                                      │
│     - Track token usage                                        │
│     - Execute tool calls via MCP                               │
│     - TRUNCATE tool responses (keep last 2 full, rest 2000ch)  │
│     - FORCED REFLECTION after each non-journal tool:           │
│       • Inject REFLECTION_PROMPT (text-only LLM call)          │
│       • Extract LEARNED/PLAN/NEXT from response                │
│       • Store reflection in journal via ManageJournal          │
│     - LOOP DETECTION: Check for infinite patterns              │
│     - If loop detected: Inject intervention message            │
│     - Every 5 iterations: Inject journal refresh at index 1    │
│       (REPLACE mode after first refresh for primacy bias)      │
│     - PROGRESS CHECK: Compare journal state                    │
│     - If GetJournalSummary called: Force answer next turn      │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. POST-AGENT HOOK (Deterministic Synthesis)                  │
│    - Fetch complete GetJournalSummary                          │
│    - Inject synthesis prompt with all discovered data          │
│    - Make final LLM call using SYNTHESIS client/model          │
│    - Track synthesis token usage                               │
│    - Validate and return final answer                          │
└────────────────────────────────────────────────────────────────┘
```

### Question Type Classification

The agent classifies questions into 10 types, each with a specific strategy:

| Type | Description | Strategy | Completion Gate |
|------|-------------|----------|-----------------|
| **Count** | "How many..." | Use FilterEntities for concept/attribute filtering; RunSPARQL with COUNT() for large sets; OR/UNION conditions use genericized UNION pattern with COUNT(DISTINCT) to avoid double-counting | ✅ Property discovery + COUNT query execution + numeric result |
| **Verify** | "Is...", "Does..." | Use VerifyNumericCondition for numeric/date TRUE/FALSE; use VerifyString for text comparisons (never guess string equality) | - |
| **Select** | General selection | Entity identification and attribute lookup | - |
| **SelectBetween** | Compare 2 entities | Use CompareEntities, verify constraints | ✅ Both entity values retrieved + comparison made + answer identified |
| **SelectAmong** | Superlative (most, least) | Use RunSPARQL with ORDER BY LIMIT 1 | - |
| **QueryAttr** | Direct attribute lookup | Use GetAttributeDetails (note: prepositional phrase disambiguation applies) | - |
| **QueryAttrQualifier** | Attribute with context | Use `GetQualifierValue` once qualifier name is known (preferred); `GetEdgeQualifiers` for discovery; `QualifierFilter` to narrow entities by qualifier condition (When→point_in_time, Where→location) | - |
| **QueryRelation** | Relationship identification | Use GetRelationDetails | - |
| **QueryRelationQualifier** | Relation with context | Use `GetQualifierValue` once qualifier name is known (preferred); `GetQualifiersByPredicate` for discovery; `QualifierFilter` to narrow entities by qualifier condition (When→point_in_time/start_time, Where→location, What role→object_has_role, ceremony, For what→AMBIGUOUS: check both for_work and ceremony) | - |
| **QueryName** | Reverse lookup | Use FindByAttribute, FilterEntities for type+attribute conditions, or RunSPARQL | - |
| **Query** | General query | Multi-step reasoning (step 2b: prepositional phrase cross-reference to Rule #7) | - |

**Completion Gates:** Some question types have explicit completion gates to prevent premature termination. The agent must complete all checkpoints in the gate before stopping. This prevents the agent from concluding "I cannot answer" before attempting all necessary steps.

### Loop Detection Mechanisms

**5 Detection Layers:**

1. **Identical Repeated Calls** (3x same tool+params)
   ```
   FindNode("Boston") → FindNode("Boston") → FindNode("Boston")
   → LOOP DETECTED
   ```

2. **Oscillating Pattern** (A-B-A-B or A-B-C-A-B-C)
   ```
   GetAttr → FindNode → GetAttr → FindNode → GetAttr → FindNode
   → LOOP DETECTED (A-B-A-B-A-B pattern)
   ```

3. **Tool Spam** (5/6 calls same tool, different params)
   ```
   RunSPARQL(q1) → RunSPARQL(q2) → X → RunSPARQL(q3) → RunSPARQL(q4) → RunSPARQL(q5)
   → LOOP DETECTED (RunSPARQL 5/6 times)
   ```

4. **FindResource Cap** (8 calls for KQAPro, 8 for SciQA)
   - Configured via `find_resource_cap` in domain_settings
   - Prevents semantic search exhaustion
   - Message: "FindResource called N times (cap: 8). Switch to SPARQL or other structured queries."

5. **RunORKGSPARQL Cap** (10 calls for SciQA, configurable per agent)
   - Configured via `sparql_cap` in domain_settings (default: 10)
   - Prevents runaway SPARQL query spirals
   - Message: "RunORKGSPARQL called N times (cap: 10). Use GetComparisonContributions or GetResourceSummary instead."
   - Only applies to agents with RunORKGSPARQL tool (e.g., SciQA)

6. **No Progress** (Journal unchanged for 5 iterations)
   - Checked during periodic journal refresh
   - Triggers strong intervention if journal state identical

### Journal (Scratchpad) System

**File:** `ama_kbqa/framework/state.py` (single source of truth — Pydantic `BaseModel`)

`kqapro_server.py` and `sciqa_server.py` both import `JournalState` from `ama_kbqa.framework.state` instead of defining their own. The class has `ClassVar` constants for caps and two serialization methods: `to_str()` (compact MCP server format) and `to_summary_str()` (framework / synthesis format).

The journal tracks agent progress:

```python
class JournalState(BaseModel):
    # Question context
    question_text: str          # set via ManageJournal("set_question", ...)
    question_type: str
    target_entities: list[str]

    # Exploration tracking
    visited_nodes: dict[str, str]      # {node_id: node_name}
    verified_facts: list[dict]
    failed_attempts: list[str]         # capped at MAX_FAILED_ATTEMPTS = 10

    # CRITICAL - Discovered values
    found_values: dict[str, dict]      # {entity_id: {attr: value}}
                                       # also receives sparql_result_N entries from RunSPARQL

    # Progress
    current_plan: list[str]            # set as a list via ManageJournal("update_plan", ...)
    completed_steps: list[str]         # capped at MAX_COMPLETED_STEPS = 20
    partial_answer: str

    # Class-level caps (ClassVar)
    MAX_COMPLETED_STEPS: ClassVar[int] = 20
    MAX_FAILED_ATTEMPTS: ClassVar[int] = 10
```

**Helper methods on JournalState:**
- `add_completed_step(step)` - Appends and caps to last 20 entries
- `add_failed_attempt(attempt)` - Appends and caps to last 10 entries

#### Journal Data Model: Three Distinct Buckets

The three fields `found_values`, `verified_facts`, and `visited_nodes` serve distinct roles. Understanding the difference is critical for writing correct tools.

| Field | Type | Role | Rendered in summary as |
|-------|------|------|------------------------|
| `found_values` | `dict[entity_id, dict[attr, list]]` | **Answer-bearing values.** The synthesis step receives ONLY the rendered journal summary (minimal-context synthesis: system prompt + journal + query, no conversation history). `GetJournalSummary` builds the "DISCOVERED VALUES" block from `found_values`. If an answer is not here, synthesis cannot see it. | `📊 DISCOVERED VALUES` section |
| `verified_facts` | `list[dict]` | Relation triples confirmed as true (subject → relation → target). Used for provenance tracking and the `🔗 VERIFIED FACTS` block. Before April 2026, only `verified_facts` was populated by `GetRelationDetails`, causing synthesis to miss relation-based answers entirely. | `🔗 VERIFIED FACTS` section (up to 15 triples with labels) |
| `visited_nodes` | `dict[node_id, node_name]` | **Label resolution map.** Records every node explored as `{id: human_label}`. Used at render time by `GetJournalSummary` to resolve opaque IDs (like `Q3012`) into readable labels (like `Ulm`). Both the `DISCOVERED VALUES` and `VERIFIED FACTS` sections resolve IDs via this dict at render time. | Referenced inline to resolve IDs |

**Architectural invariant — every answer-producing tool MUST write to `found_values`:**

Synthesis runs with minimal context (system prompt + journal summary + query only — no conversation history). Any value that is only in `verified_facts` or only returned in a tool response is invisible to synthesis. This was the root cause of a class of bugs where relation-resolved answers would succeed in the tool loop but fail at synthesis with "⚠️ NO VALUES DISCOVERED YET". If you add a new tool that produces an answer value, it must write to `found_values`.

**Auto-Updates:**
- `visited_nodes` - Updated by FindNode (with dedup: skips Qdrant search if label already in `visited_nodes`), GetNodeLabel, BatchGetNodeLabels
- `found_values` - Updated by: GetAttributeDetails, GetNodeSummary, RunSPARQL (stores results as `sparql_result_N` with `{"query": query[:200], "results": simplified_rows[:10]}`), **GetRelationDetails** (stores relation targets so synthesis can see them — see below), FilterEntities
- `verified_facts` - Updated by GetRelationDetails and other relation tools
- `failed_attempts` - Logged when tools fail (via `add_failed_attempt()` helper)
- `completed_steps` - Logged on tool success (via `add_completed_step()` helper)
- `question_text` - Set by `ManageJournal("set_question", content)`

#### GetRelationDetails: Dual Journal Write

`GetRelationDetails` now writes to **both** `verified_facts` and `found_values`:

```python
# verified_facts entry (for provenance)
{"subject": base_node_id, "relation": relation_name,
 "related_id": related_id, "direction": direction, "source": "GetRelationDetails"}

# found_values entry (so synthesis can see the answer)
found_values[base_node_id][relation_name] = [
    {"value": related_label_or_id, "related_id": related_id, "direction": direction},
    ...
]
```

The `value` field is pre-populated with the label from `visited_nodes` if already resolved, or falls back to the raw ID. Calling `GetNodeLabel` / `BatchGetNodeLabels` afterwards will backfill the `value` field in any matching `found_values` entry.

#### GetNodeLabel / BatchGetNodeLabels: `found_values` Backfill

After resolving a label for a node ID, both tools scan `session_journal.found_values` for any list entry with a matching `related_id` and upgrade its `value` field from the opaque ID to the human label. This means calling `GetNodeLabel("Q3012")` after `GetRelationDetails` automatically upgrades `"value": "Q3012"` to `"value": "Ulm"` in the rendered summary.

#### GetJournalSummary: Rendered Output Format

The summary text that synthesis receives contains these sections (in order):

1. **`📊 DISCOVERED VALUES`** — one block per entity in `found_values`, with attribute/relation names and values. IDs resolved via `visited_nodes` at render time. Shows up to 3 entries per attribute.
2. **`🔗 VERIFIED FACTS`** — up to 15 subject→relation→target triples from `verified_facts`, with subjects and targets resolved via `visited_nodes`.
3. **`📈 PROGRESS`** — node/fact/step counts.
4. **`🔍 EXPLORED NODES`** — first 5 visited nodes with a `✓ HAS DATA` / `○ no data yet` marker.
5. **`💭 PARTIAL ANSWER`** — if set.

The empty-state marker `⚠️ NO VALUES DISCOVERED YET` appears only when `found_values` is empty. `_run_synthesis` in `base_agent.py` keys off the literal strings emitted by this renderer (not internal field names) to determine whether the journal has data.

**ManageJournal actions:**
- `"set_question"` - Sets `session_journal.question_text = content` ✨ NEW
- `"update_plan"` - Splits `content` on newlines into a list (`[s.strip() for s in content.split("\n") if s.strip()]`); agent passes multi-step plans as newline-separated text ✨ UPDATED (was: wrapped content in a 1-element list)
- `"update"` - General journal update (reflection notes)

**Scratchpad-Enforced Reflection:**

After each non-journal tool call, the agent is forced to reflect by:
1. Injecting REFLECTION_PROMPT (text-only LLM call, no tools)
2. Extracting structured reflection: LEARNED, PLAN, NEXT
3. Storing reflection in journal via ManageJournal(action="update", content="reflection")

This ensures the agent maintains a running mental model and doesn't lose context as tool responses are truncated.

**Tool Response Truncation:**
- Tool responses over 2000 characters are truncated (except journal tools)
- Last 2 tool responses preserved in full (for immediate context)
- All earlier responses compacted after each tool batch
- Reduces context window bloat while preserving critical recent context
- Increased from 1000 to 2000 chars to preserve complete property lists from GetNodeSummary

**Journal Refresh Placement:**
- Journal refresh injected at index 1 (after system prompt, before user question)
- First refresh uses INSERT mode, subsequent refreshes use REPLACE mode
- This exploits primacy bias: LLM sees journal state first
- Only 1 journal refresh exists in message history at any time

### Message History Format

OpenAI-compatible conversation format with scratchpad-enforced loop:

```python
[
    {"role": "system", "content": "[System prompt with KBQA rules]"},
    {"role": "user", "content": "[JOURNAL REFRESH - replaces on subsequent refreshes]"},
    {"role": "user", "content": "What is the population of Boston?"},
    {"role": "user", "content": "[PRE-ANALYSIS: Question Type, Strategy, etc.]"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "FindNode", "content": "[TRUNCATED if >2000 chars]"},
    {"role": "assistant", "content": "[REFLECTION: LEARNED/PLAN/NEXT]"},  # Forced reflection
    {"role": "tool", "tool_call_id": "...", "name": "ManageJournal", "content": "Journal updated"},
    # ... more iterations ...
    {"role": "user", "content": "[SYNTHESIS PROMPT with journal data]"},
    {"role": "assistant", "content": "The population of Boston is..."}
]
```

**Key features:**
- Journal refresh at index 1 (primacy bias)
- Tool responses truncated (except last 2 and journal tools)
- Forced reflection after each non-journal tool
- Journal state replaces on subsequent refreshes (not appended)

---

## SciQAAgent Deep Dive

**Files:**
- `ama_kbqa/agents/sciqa_agent/agent.py` - Inherits BaseKBQAAgent (~190 lines)
- `ama_kbqa/agents/sciqa_agent/prompts.py` - ORKG-specific prompts (~890 lines)

### File Organization

| File | Contents | Line Count |
|------|----------|------------|
| `agent.py` | SciQAAgent class, overrides for ORKG-specific behavior | ~190 |
| `prompts.py` | ORKG-specific prompt strings, strategies, predicate dictionary | ~890 |

### Agent Lifecycle

Follows the same 4-phase pattern as KQAProAgent:

1. **Initialization** - Load LLM clients, connect to sciqa_server.py
2. **Pre-Agent Hook** - Classify question (8 types), extract entities, load type-specific strategy
3. **Main Agent Loop** - Call ORKG tools (20 tools), track progress
4. **Post-Agent Hook** - Synthesize answer from journal

### Prompt Architecture (Type-Specific Strategy Loading)

The SciQA prompts follow the same pattern as KQAPro: a lean SYSTEM_PROMPT with type-specific strategies loaded after classification.

```
Message History:
[0] SYSTEM: SYSTEM_PROMPT (lean - rules, schema, tools, predicates)
[1] USER: Original question
[2] USER: Analysis Context (injected after classification)
    ├── Question Type
    ├── Extracted Entities/Relations
    ├── QTYPE_STRATEGIES[detected_type]  <-- only the relevant strategy
    └── FEWSHOT_EXAMPLES[detected_type]  <-- only the relevant examples
```

**SYSTEM_PROMPT** (~165 lines) contains only general-purpose content: critical rules, ORKG schema/prefixes, 5-tier tool catalog, execution strategy, and predicate reference dictionary.

**Key Prompt Rules (CRITICAL RULES section):**
1. No hallucination - all facts must be verified via tools (with one-hop logical inference exception)
2. Schema compliance - use GetResourceSummary if predicates return no results
3. State management - use ManageJournal to track progress
4. Pivot logic - if search fails twice, switch strategies
5. Complete retrieval - always follow up FindResource with GetResourceDetails/GetResourceSummary/GetRelationTargets
6. Constraint verification - verify ALL conditions with VerifyNumericCondition before including items
7. Minimum effort - must use at least 10 tool calls before concluding data unavailable
8. **Boolean values in ORKG** - Many predicates use "T"/"t" for True/present and "F"/"f" for False/absent. When filtering for presence of a property (e.g., therapeutic effect), filter for "T" not "F".

**WARNING Block (Minimum Effort Enforcement):**
- If fewer than 5 tool calls made, MUST NOT give final answer
- Recovery strategies when stuck:
  - If FindAuthorPapers returns 0 results, try RunORKGSPARQL with REGEX on author labels
  - If FindResource returns irrelevant results, try different search terms or FindByPredicateValue
  - If GetComparisonContributions returns 0 contributions, try other FindResource results
  - If a predicate returns empty, use GetResourceSummary to discover available predicates

**QTYPE_STRATEGIES** (8 entries) contain type-specific guidance including comparison patterns, SPARQL templates, HAS_VALUE nested patterns, and decision trees. Only the relevant strategy is loaded per question.

**Enhanced Comparison/Superlative Strategies:**
- **Comparison verification guidance**: After finding a Comparison resource with FindResource, ALWAYS call GetComparisonContributions(comparison_id) in schema discovery mode first. If it returns 0 contributions, the resource may not be a real Comparison - try other results from FindResource or search with a different query.
- **Multi-hop SPARQL patterns for nested HAS_VALUE**: Some contributions use intermediate resources with HAS_VALUE for their values (Comparison → Contribution → domain_pred → IntermediateResource → HAS_VALUE → actual_value). GetComparisonContributions now includes `OPTIONAL { ?value orkgp:HAS_VALUE ?nestedValue }` and returns `nested_value` field when present.
- **Reverse-link boolean few-shot example**: Added example showing ASK pattern when entity is referenced BY a contribution (reverse direction), with "t"/"T" for True and "f"/"F" for False in boolean predicates.

### Question Type Classification

SciQA classifies questions into 8 types:

| Type | Description | Strategy |
|------|-------------|----------|
| **Factoid** | Direct fact lookup | GetResourceSummary for exploration, comparison-based factoid guidance, SPARQL domain data tip |
| **Count** | "How many..." | RunORKGSPARQL with COUNT(), comparison-based counting with HAS_VALUE nested pattern |
| **List** | "Which papers..." | Comparison-based list pattern, GetComparisonContributions, multi-hop lists |
| **Boolean** | "Is...", "Does..." | ASK SPARQL for comparison data, VerifyNumericCondition for numeric conditions |
| **Comparison** | Compare entities | CompareResources for batch comparison, full comparison navigation with nested value pattern |
| **Superlative** | "highest", "lowest", "boundaries" | SPARQL ORDER BY + LIMIT, comparison-based superlatives with HAS_VALUE nested pattern |
| **Aggregation** | SUM, AVG, total | RunORKGSPARQL with aggregation functions, comparison-based aggregation with HAS_VALUE |
| **General** | Complex/other | GetResourceSummary for exploration, comparison mention as fallback, SPARQL domain data tip |

### SciQA MCP Tools

The sciqa_server.py provides 20 tools organized by tier:

**Tier 1 - Discovery (4 tools):**
- `FindResource(semantic_query)` - Vector search for ORKG resources
- `FindPredicate(semantic_query)` - Find ORKG predicate names
- `FindByPredicateValue(predicate_id, value, match_type)` - Reverse lookup by predicate value (exact/contains/greater/less)
- `FindAuthorPapers(author_name)` - SPARQL-based author name search with **UNION clause for both resource-URI authors and string literal authors** (handles `orkgp:P27 ?author` with `?author rdfs:label` OR `orkgp:P27 ?authorLabel` with `isLiteral()` filter). Case-insensitive partial match, better than vector search for proper nouns.

**Tier 2 - Retrieval (6 tools):**
- `GetResourceDetails(resource_id)` - Full resource info
- `GetResourceSummary(resource_id)` - ALL predicates in one call (preferred for exploration)
- `GetRelationTargets(resource_id, predicate)` - Follow specific relations with **reverse lookup fallback** (if direct query returns 0 results, tries finding resources that reference the target) ✨ UPDATED
- `GetResourceLabel(resource_id)` - Quick label lookup
- `BatchGetResourceLabels(resource_ids)` - Batch resolution
- `CompareResources(resource_ids, predicate_id)` - Compare predicate across resources (sorted)

**Tier 3 - Domain-Specific (8 tools):**
- `GetPaperContributions(paper_id)` - Paper contributions via P31
- `GetPaperAuthors(paper_id)` - Authors via P6/P27
- `GetContributionMethods(contribution_id)` - Methods via P2
- `GetResearchFieldPapers(field_name)` - Papers in field via P30
- `GetComparisonContributions(comparison_id, domain_predicate, filter_value, filter_type)` - Navigate Comparison -> Contribution pattern with predicate discovery mode. Now includes **automatic 4-hop value resolution**: query includes `OPTIONAL { ?value orkgp:HAS_VALUE ?nestedValue }` and response includes `nested_value` field when present (Comparison → Contribution → intermediate_resource → HAS_VALUE → actual_value). Stores values in journal's found_values.
- `FollowRelationPath(start_resource_id, relation_path)` - Multi-hop navigation in one SPARQL call
- `AggregateComparisonValues(comparison_id, value_predicate, agg, group_by_predicate?, filter_predicate?, filter_value?, filter_match?, top_n?, value_via_group?)` - SPARQL + Python aggregation over a Comparison's contributions. Handles HAS_VALUE/label indirection. `agg` options: `avg|sum|min|max|count|count_distinct|mode_top|all_values`. `value_via_group=True` adds a 2-hop: `?contrib group_pred ?group . ?group value_pred ?scalar` (needed for grouped energy-domain questions). Smoke-tested against gold answers for Q3/Q11/Q13/Q16/Q18. ✨ NEW
- `FindCoAuthors(author_name, top_n?)` - Finds co-authors of papers by a seed author (partial case-insensitive match; handles resource-URI authors via P27/P6 + rdfs:label and literal-string authors via isLiteral() filter); returns co-authors sorted by shared-paper count descending. Smoke-tested against Q2 gold. ✨ NEW

**Tier 4 - Raw SPARQL (1 tool):**
- `RunORKGSPARQL(query)` - Raw SPARQL (prefixes auto-injected, **capped at 10 calls per question** to prevent runaway SPARQL spirals) ✨ UPDATED

**Tier 5 - Verification (1 tool):**
- `VerifyNumericCondition(value1, operator, value2, unit)` - Deterministic math/date comparison (TRUE/FALSE/ERROR)

**State Management (2 tools):**
- `ManageJournal(action, content)` - Scratchpad management
- `GetJournalSummary()` - Summary of discoveries

### ORKG Predicate Reference

**Core Navigation Predicates:**

| Predicate | URI | Description | Pattern |
|-----------|-----|-------------|---------|
| P0 | `orkgp:P0` | addresses (problem) | Paper/Contribution -> Problem |
| P1 | `orkgp:P1` | yields (result) | Contribution -> Result |
| P2 | `orkgp:P2` | employs (method) | Contribution -> Method |
| P6 | `orkgp:P6` | author | Paper -> Author |
| P27 | `orkgp:P27` | author (alternative) | Paper -> Author |
| P7 | `orkgp:P7` | affiliation | Author -> Organization |
| P10 | `orkgp:P10` | DOI | Paper -> DOI string |
| P26 | `orkgp:P26` | has DOI | Paper -> DOI string |
| P29 | `orkgp:P29` | publication year | Paper -> Year |
| P30 | `orkgp:P30` | research field | Paper -> ResearchField |
| P31 | `orkgp:P31` | has contribution | Paper -> Contribution (**CRITICAL PATH**) |
| P32 | `orkgp:P32` | research problem | Paper -> Problem |

**Key Navigation Pattern (~70% of questions):**
```
Paper --P31--> Contribution --domain_predicate--> Value
```

**Domain-Specific Predicates (via Contributions):**

| Domain | Predicate | Description |
|--------|-----------|-------------|
| Energy | P43133 | installed capacity |
| Energy | P43135 | energy sources |
| Energy | P43247/P43248 | upper/lower limit |
| Chemistry | P35147 | Bisphenol A analogue |
| Chemistry | P35194 | SAME_AS (alt names) |
| Benchmarks/NLP | P41923 | amount of questions |
| Benchmarks/NLP | P15585 | has benchmark |
| Biology | P37458 | major anion type |
| Biology | P37586 | study type |
| Comparison | P5038 | Aggregation |
| Comparison | P5039 | tool capabilities |

### SciQA Batch Processing

**See:** [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) for complete unified batch processing documentation.

```bash
# Run on handcrafted dataset (100 Q&A)
python -m ama_kbqa.benchmark_agents --agents sciqa --n-questions 10 --seed 42 --dataset handcrafted

# Run on autogenerated dataset (368 Q&A)
python -m ama_kbqa.benchmark_agents --agents sciqa --n-questions 50 --dataset auto --postprocessing llm_judge
```

**Dataset Format (CSV):**

| Column | Description |
|--------|-------------|
| `Paraphrase` | Question text |
| `Result` | Ground truth answer |
| `Machine-readable query` | SPARQL query |
| `Q Content` | Question type |
| `Research field` | Domain |

**Output Structure:**
```
benchmark_results/<YYYY-MM-DD-N>/
├── sciqa/
│   └── <model-name>/       # or "default" for single-model mode
│       ├── results.json    # Per-question results
│       ├── summary.json    # Accuracy statistics
│       └── judgments.json  # LLM judge evaluations (if llm_judge mode)
```

---

## MCP Client Pattern

**File:** `ama_kbqa/framework/mcp_client.py` (shared MCPClient class)

```python
class MCPClient:
    def __init__(self, server_path: str, agent_name: str):
        self.server_path = Path(server_path)
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None

    async def start(self):
        # Connect via stdio
        client_gen = stdio_client(StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_path)]
        ))
        read, write = await self.exit_stack.enter_async_context(client_gen)
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def list_tools(self) -> List[McpTool]:
        return (await self.session.list_tools()).tools

    async def call_tool(self, name: str, args: Dict) -> str:
        result = await self.session.call_tool(name, arguments=args)
        return result.content[0].text

    async def close(self):
        await self.exit_stack.aclose()
```

---

## Token Tracking

Tokens are tracked across all LLM calls in the KQAProAgent:

```python
self.token_usage = {
    "prompt_tokens": 0,       # Input to LLM
    "completion_tokens": 0,   # Output from LLM
    "total_tokens": 0         # Sum
}
```

**Tracked LLM Calls (5 total):**

| Method | Location (line) | Purpose |
|--------|-----------------|---------|
| `_classify_question()` | ~305 | Question type classification |
| `_extract_entities()` | ~350 | Entity/relation extraction |
| `_llm_call()` | ~876 | Main agent loop (tool calls) |
| `_llm_call_text_only()` | ~903 | Fallback when no tools |
| `_llm_call_synthesis()` | ~923 | Final answer synthesis |

Each location uses the same pattern:
```python
if response.usage:
    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
    self.token_usage["completion_tokens"] += response.usage.completion_tokens
    self.token_usage["total_tokens"] += response.usage.total_tokens
```

**Note:** Token tracking in postprocessing functions (LLM judge, answer selection, SPARQL synthesis) is not currently aggregated into the agent's token counts.

Reported in batch results JSON:
```json
{
    "prompt_tokens": 15234,
    "completion_tokens": 856,
    "total_tokens": 16090,
    "duration_seconds": 12.45,
    "agent_turns": 8
}
```

---

## Tool Call Duration Tracking

Each tool call is timed and tracked for performance analysis:

```python
self.tool_call_durations: List[Dict[str, Any]] = []
```

**Tracked Data Per Tool Call:**

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Name of the tool called |
| `duration_seconds` | float | How long the call took (rounded to 3 decimals) |
| `success` | bool | Whether the call succeeded |
| `timestamp` | str | ISO format timestamp |
| `error` | str | Error message (only if failed) |

**Console Output:**
Tool calls now show duration in logs:
```
🔙 Result (SearchEntities) [0.234s]: {"matches": [...]}
```

**Summary at Question End:**
```
⏱️  TOOL CALL SUMMARY: 15 calls, total 4.567s
   📊 SearchEntities: 5x, total 1.234s, avg 0.247s
   📊 ExecuteSPARQL: 3x, total 0.891s, avg 0.297s
```

**Accessing Summary Programmatically:**
```python
summary = agent.get_tool_call_summary()
# Returns:
{
    "total_calls": 15,
    "total_duration_seconds": 4.567,
    "tool_breakdown": {
        "SearchEntities": {
            "count": 5,
            "total_duration": 1.234,
            "avg_duration": 0.247,
            "success_count": 5,
            "failure_count": 0
        },
        # ...
    },
    "calls": [...]  # Individual call records
}
```

**Batch Processing Aggregation:**
In batch results, tool statistics are aggregated across all questions:
```json
{
    "statistics": {
        "total_tool_calls": 150,
        "total_tool_duration_seconds": 45.234,
        "avg_tool_calls_per_question": 15.0,
        "tool_breakdown": {
            "SearchEntities": {"count": 50, "total_duration": 12.345, "avg_duration": 0.247}
        }
    }
}
```

---

## Synthesis Configuration

The agent uses **separate LLM configuration** for final answer synthesis:

```toml
# config.toml
[synthesis]
synthesis_enabled = true          # Set false to skip synthesis and use agent's final message
synthesis_provider = "openrouter"
synthesis_model = "deepseek/deepseek-v3.2"
synthesis_temperature = 0.2
synthesis_max_tokens = 8000
synthesis_mode = "benchmark"      # "benchmark" (short exact-match) or "conversational"
```

**`synthesis_enabled` flag (added April 2026):**

When `synthesis_enabled = false`, `_run_tool_loop` returns the agent's own last assistant message directly, bypassing the second LLM call entirely. Falls back to synthesis if the agent produced no final content (e.g., max-iteration abort).

- **Getter:** `get_synthesis_enabled()` in `ama_kbqa/config.py` (defaults to `True`)
- **Frontend toggle:** "Run synthesis step" in `pages/4_Settings.py` → Synthesis Configuration section
- **Tradeoff:** saves one LLM call per question; answer shape is less deterministic (no explicit "give a short, exact answer" shaping)

**`auto_inject_journal` flag (added April 2026):**

Controls whether `_run_tool_loop` automatically pushes journal state into the conversation. Configured under the new `[agent]` TOML section; defaults to `true`.

Two injection sites in `_run_tool_loop` are guarded by this flag:

1. **Periodic refresh** — every `journal_refresh_interval` iterations (default: every 5), `_inject_journal_refresh` is called. It fetches `GetJournalSummary`, applies the `WORKING MEMORY REFRESH (Iteration N)` template (or the `WARNING: NO PROGRESS DETECTED` template when the journal state is unchanged), strips any prior refresh message from history (replace-not-append strategy), and appends the new one. When `auto_inject_journal = false` this call is skipped entirely.

2. **Post-GetJournalSummary answer prompt** — when the agent voluntarily calls `GetJournalSummary` as a tool, `_execute_tool_calls` sets a flag and `_run_tool_loop` injects `"Now provide your final answer. Do NOT call more tools."` immediately after the tool result. When `auto_inject_journal = false` this prompt is suppressed.

`GetJournalSummary` remains available as a callable tool regardless of this flag. The toggle only disables *automatic* pushes; the agent can still consult the journal on its own initiative.

- **Getter:** `get_auto_inject_journal()` in `ama_kbqa/config.py` (defaults to `True`)
- **Frontend toggle:** "Auto-inject journal into context" in `pages/4_Settings.py` → Agent Configuration section (above Synthesis Configuration)
- **Tradeoff:** disabling yields a leaner conversation (no periodic refresh tokens), but the agent must track its own progress solely from tool-result history; useful when the model has strong working memory or when minimizing context size is critical

**How synthesis works (when enabled):**

1. `_run_synthesis` calls `GetJournalSummary` to obtain the rendered journal text.
2. A minimal two-message conversation is built: `[system_prompt, synthesis_prompt(journal + query)]`. No conversation history — this is the primary token saving (30-80k tokens per question).
3. If synthesis returns a failure phrase ("cannot answer", "not found", etc.) but the journal contains `discovered values` or `verified facts` markers, a re-prompt is issued with the rendered journal data.
4. The `has_data` check keys off literal strings emitted by the renderer: `"discovered values"` (excluding `"no values discovered yet"`), `"verified facts"`, `"partial answer:"`, `"orkgr:"`. Do not change these marker strings in `GetJournalSummary` without updating the heuristic in `_run_synthesis`.

**Benefits:**
- Use faster/cheaper model for synthesis
- Different temperature for consistency
- Separate token limits
- Model specialization (summarization vs reasoning)

---

## Batch Processing

**See:** [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) for complete unified batch processing documentation.

```bash
# Run 10 questions with default seed
python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 10 --seed 42

# Run with LLM judge evaluation
python -m ama_kbqa.benchmark_agents --agents kqapro --n-questions 10 --postprocessing llm_judge
```

**Output Structure:**
```
benchmark_results/<YYYY-MM-DD-N>/
├── kqapro/
│   └── <model-name>/           # or "default" for single-model mode
│       ├── results.json        # Per-question results
│       ├── summary.json        # Accuracy statistics
│       └── judgments.json      # LLM judge evaluations (if llm_judge mode)
```

### LLM Judge

Evaluates answer quality with structured output:

```python
class AnswerJudgment:
    is_correct: bool                # Semantic equivalence
    correctness_reasoning: str      # Detailed comparison
    argumentation_quality: str      # Reasoning assessment
    argumentation_score: int        # 1-5 score
    suggested_improvement: str      # Actionable feedback
```

---

## Agent Reset Pattern

**Three reset methods available:**

### 1. `reset(keep_mcp_open=False)` - Full Reset (Default)

```python
await agent.reset()

# What gets reset:
# - Message history (except system prompt)
# - Token usage counters
# - Tool call duration tracking
# - Loop detection tracking
# - MCP server connection (closed and reopened)

# What persists:
# - LLM client configuration
# - System prompt
```

### 2. `soft_reset()` - Batch Processing (MCP Preserved)

**Recommended for batch processing** - Keeps MCP server running for efficiency:

```python
await agent.soft_reset()

# What gets reset:
# - Message history (except system prompt)
# - Token usage counters
# - Tool call duration tracking
# - Loop detection tracking
# - Journal/scratchpad state (cleared via ManageJournal tool)

# What persists:
# - MCP server connection (stays open!)
# - LLM client configuration
# - System prompt
```

### 3. `close()` - Explicit MCP Shutdown

**Call at the end of batch processing** to properly close MCP:

```python
await agent.close()  # Closes MCP server connection
```

### Batch Processing Pattern

```python
agent = KQAProAgent()

try:
    for question in questions:
        answer = await agent.ask(question)
        results.append(answer)
        await agent.soft_reset()  # Keep MCP open between questions
finally:
    await agent.close()  # Clean shutdown at the end
```

**Benefits of MCP Persistence:**
- Avoids subprocess startup overhead per question
- Maintains warm connections to Qdrant/Virtuoso
- Significantly faster batch processing
- Proper cleanup with `finally` block

---

## Tracing System

Color-coded console output for debugging:

| Color | Meaning |
|-------|---------|
| BLUE | General agent messages |
| GREEN | Tool calls/results, success |
| RED | Errors |
| YELLOW | Warnings, tool calls |
| CYAN | Important state changes, reasoning |

```python
def trace(agent_name: str, msg: str, color: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{agent_name}] -> {msg}")
```

---

## LLM-Based Fewshot Generator

**Module:** `ama_kbqa/fewshot_generator.py`

After benchmark runs with llm_judge evaluation, the system can optionally generate fewshot learning material by analyzing successful and failed attempts.

### Generator Workflow

1. **Qualification Filter** - Selects results worth learning from:
   - Correct answers with `argumentation_score >= 4`
   - Any incorrect answer (to learn from mistakes)

2. **LLM Analysis** - Model and parameters from `[fewshot_generator]` in `config.toml` (defaults: `deepseek/deepseek-v4-pro`, temp 1.0, max_tokens 16000). Analyzes:
   - Full conversation trace (`include_full_conversation=true` by default — no truncation)
   - Agent's live tool catalog (`include_tool_descriptions=true` by default — spawns MCP server to fetch `list_tools`, formatted via `build_text_mode_tool_catalog`)
   - Tool trace overview
   - Judge verdict (correctness reasoning, argumentation quality, score)
   - Question metadata (type, gold answer, predicted answer)

3. **Output Generation** - Produces up to 3 optional outputs per question:
   - **QType Example** - Reusable strategy pattern for this question type. Even for correct answers, the generator checks path optimality against the available tools.
   - **General Example** - Cross-type insight applicable to multiple types
   - **Tool Tip** - Tool-specific gotcha or usage pattern (generated when a more direct tool existed than the one the agent used)

4. **Deduplication & Storage**:
   - Per-qtype examples: deduplicated by question text, saved to `<QType>.json` (max 5)
   - General guidance: deduplicated by title, saved to `_general.json` (max 10)
   - Tool tips: deduplicated by tool_name+problem_pattern, saved to `_tool_tips.json` (max 20)

5. **Audit Logging** - All generated examples (pre-dedup) saved to `generated_fewshot.json`

6. **Post-Hoc Replay** - `ama_kbqa/run_fewshot_generator.py` replays the generator over a saved benchmark dir without re-running the agent. Always writes to a shadow directory; refuses to write to the live `fewshot-examples/` dir. See `Decisions/fewshot-generator-enrichment.md` for rationale.

### Agent Integration

The KQAProAgent automatically loads and injects generated material:

**Loading Methods:**

| Method | Source File | Limit | Format |
|--------|-------------|-------|--------|
| `_load_general_guidance()` | `_general.json` | Top 5 | `[applies_to] title: guidance` |
| `_load_tool_tips()` | `_tool_tips.json` | Top 10 | `tool_name | When: pattern | Do: guidance` |

**Injection Point:**

Both are injected during `_build_analysis_context()` after few-shot examples and before the analysis context suffix:

```
[Pre-Analysis Context]
├── Question Type
├── Extracted Entities
├── Type-Specific Strategy
├── Few-Shot Examples (if enabled)
├── General Guidance (if available)  <-- NEW
├── Tool Tips (if available)         <-- NEW
└── Context Suffix
```

**Templates:**

```python
GENERAL_GUIDANCE_TEMPLATE = """
General Guidance (learned from previous runs):
{general_guidance}
"""

TOOL_TIPS_TEMPLATE = """
Tool Tips:
{tool_tips}
"""
```

### Configuration

**Enable in CLI:**

```bash
python -m ama_kbqa.benchmark_agents --agents kqapro --postprocessing llm_judge --generate-fewshot
```

**Enable in config.toml:**

```toml
[postprocessing]
generate_fewshot = true
```

**Priority:** CLI flag > config.toml > `false` (default)

### Example Outputs

**Per-QType Example (_general.json):**

```json
{
  "question": "How many films did Christopher Nolan direct?",
  "answer": "11",
  "qtype": "Count",
  "trace": [
    {"tool": "FindNode", "args": "Christopher Nolan", "result": "Q123456"},
    {"tool": "RunSPARQL", "args": "SELECT COUNT(?film) WHERE...", "result": "11"}
  ],
  "lesson": "For count queries, use RunSPARQL with COUNT() aggregation instead of iterating with GetRelationDetails.",
  "pitfall": "Avoid calling GetRelationDetails repeatedly - it's inefficient for counting.",
  "tool_count": 2,
  "was_correct": true
}
```

**General Guidance (_general.json):**

```json
{
  "title": "SPARQL for Large Result Sets",
  "guidance": "When counting or filtering many entities, prefer RunSPARQL over iterative tool calls. It's faster and avoids loop patterns.",
  "applies_to": ["Count", "SelectAmong", "QueryName"],
  "derived_from_qtype": "Count",
  "was_correct": true
}
```

**Tool Tip (_tool_tips.json):**

```json
{
  "tool_name": "FindNode",
  "problem_pattern": "Semantic search returns irrelevant entities",
  "guidance": "Try exact ID match with GetNodeLabel or use FindByAttribute for attribute-based search",
  "example_args": "entity_name='Boston' for city disambiguation",
  "derived_from_question": "What is the population of Boston?"
}
```

---

## Related Documentation

- [Project Architecture](project_architecture.md) - System overview
- [Database Schema](database_schema.md) - Qdrant/Virtuoso schemas
- [../SOP/adding_new_tools.md](../SOP/adding_new_tools.md) - How to add tools
- [../SOP/running_batch_processing.md](../SOP/running_batch_processing.md) - Batch processing with fewshot generation
- [../../TOOLS_REFERENCE.md](../../TOOLS_REFERENCE.md) - Complete tool reference

# KQAProAgent

KBQA agent for Wikidata-derived general-knowledge factoid Q&A (~47K entities).
Inherits from `BaseKBQAAgent`. Split out of the former monolithic
`agent_system.md` (2026-07-18) — see [Agent Framework](agent_framework.md) for
shared mechanics (synthesis, trace, reset, tool gating) not repeated here.

## Related Docs
- [Agent System (index)](agent_system.md) — map of all agent-system docs
- [Agent Framework](agent_framework.md) — `BaseKBQAAgent` shared mechanics
- [SciQAAgent](sciqa_agent.md) — the sibling agent for ORKG scientific QA
- [Project Architecture](project_architecture.md) — full tool reference and directory tree
- [SOP/adding_new_kqapro_tools.md](../SOP/adding_new_kqapro_tools.md) — checklist for adding tools

---

## File Organization

- `ama_kbqa/agents/kqapro_agent/agent.py` (~290 lines) — `KQAProAgent` class, overrides for classification/extraction
- `ama_kbqa/agents/kqapro_agent/prompts.py` (~665 lines) — all prompt strings and templates

**`prompts.py` exports:** `QTYPE_STRATEGIES` (10 question-type strategies), `SYSTEM_PROMPT` (8 critical rules including anti-premature-termination, prepositional-phrase disambiguation, DO NOT BACKTRACK journal confidence), `CLASSIFICATION_PROMPT_TEMPLATE`, `ENTITY_EXTRACTION_PROMPT`, `ANALYSIS_CONTEXT_TEMPLATE`, `FEWSHOT_EXAMPLES_TEMPLATE` (all 10 qtypes), `GENERAL_GUIDANCE_TEMPLATE` (top 5 from `_general.json`), `TOOL_TIPS_TEMPLATE` (top 10 from `_tool_tips.json`), `ANALYSIS_CONTEXT_SUFFIX`, `JOURNAL_REFRESH_TEMPLATE`, `NO_PROGRESS_TEMPLATE`, `SYNTHESIS_PROMPT_TEMPLATE`, `JOURNAL_SUMMARY_ANSWER_PROMPT`, `TOOL_LOOP_GUIDANCE`, `GENERIC_LOOP_GUIDANCE`, `LOOP_INTERVENTION_TEMPLATE`.

## Agent Lifecycle

```
┌────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
│    - Load chat + synthesis LLM clients from config.toml         │
│    - Initialize MCP server connection                           │
│    - Load system prompt; reset recorder + journal_snapshots     │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│ 2. PRE-AGENT HOOK  [classify span]                              │
│    - Classify question type (10 types); extract entities        │
│    - Load qtype strategy + fewshots (if enabled)                │
│    - Load general guidance (top 5) + tool tips (top 10)         │
│    - Inject pre-analysis context into message history            │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│ 3. MAIN AGENT LOOP (Scratchpad-Enforced), max iterations 50     │
│    - tool_loop_iter event → LLM call [llm_call span] → track    │
│      tokens → execute tool calls via MCP (concurrently via      │
│      asyncio.gather) [tool_call span per tool]                  │
│      • JOURNAL_MUTATING_TOOLS → GetJournalStateJSON snapshot     │
│    - TRUNCATE tool responses (last 2 full, rest 2000 chars)      │
│    - FORCED REFLECTION after each non-journal tool               │
│    - LOOP DETECTION; every 5 iterations: journal refresh          │
│      (append-only; superseded refreshes stubbed on compaction)   │
│    - If GetJournalSummary called: force answer next turn         │
└────────────────────────────────────────────────────────────────┘
                              │
┌────────────────────────────────────────────────────────────────┐
│ 4. POST-AGENT HOOK  [synthesis span]                            │
│    - Always exits through synthesis, even on max-iterations       │
│      (see Synthesis Funnel in agent_framework.md)                │
│    - Strip think blocks; normalize Verify answers to yes/no      │
└────────────────────────────────────────────────────────────────┘
```

## Question Type Classification (10 types)

| Type | Description | Strategy | Completion Gate |
|------|-------------|----------|-----------------|
| **Count** | "How many..." | `FilterEntities`/`CountEntities` for concept/attribute filtering; `CountUnion` for heterogeneous OR branches (including branch-local relation filters); `RunSPARQL` COUNT() only when no deterministic tool fits | ✅ Property discovery + COUNT execution + numeric result |
| **Verify** | "Is...", "Does..." | `VerifyNumericCondition` for numeric/date TRUE/FALSE; `VerifyString` for text (never guess string equality) | - |
| **Select** | General selection | Entity identification and attribute lookup | - |
| **SelectBetween** | Compare 2 entities | `CompareEntities`, verify constraints | ✅ Both values retrieved + comparison made + answer identified |
| **SelectAmong** | Superlative (most, least) | `RunSPARQL` with `ORDER BY LIMIT 1` | - |
| **QueryAttr** | Direct attribute lookup | `GetAttributeDetails` (prepositional-phrase disambiguation applies) | - |
| **QueryAttrQualifier** | Attribute with context | `GetQualifierValue` once qualifier name known (preferred); `GetEdgeQualifiers` for discovery; `QualifierFilter` to narrow by qualifier condition (When→point_in_time, Where→location). Award/work wording routes to relation qualifier `for_work`; fast path blocked for this pattern. | - |
| **QueryRelation** | Relationship identification | `GetRelationBetween(A, B)` when both endpoints known; otherwise `GetRelationDetails` | - |
| **QueryRelationQualifier** | Relation with context | `GetQualifierValue` (preferred); `GetQualifiersByPredicate` for discovery; `QualifierFilter` (When→point_in_time/start_time, Where→location, What role→object_has_role/ceremony, For what→AMBIGUOUS: check both). Slash qualifier names (`number_of_matches_played/races/starts`) resolve via aliases. Subscriber/follower wording maps to `number_of_subscribers`/`number_of_followers`. | - |
| **QueryName** | Reverse lookup | `FindByAttribute`, `FilterEntities` for type+attribute, or `RunSPARQL` | - |
| **Query** | General query | Multi-step reasoning (prepositional-phrase cross-reference) | - |

**Completion Gates** prevent premature termination — the agent must complete all checkpoints before concluding "I cannot answer."

## Loop Detection (KQAPro exercises layers 1-4 and 6-7; layer 5 is SciQA-specific)

1. **Identical Repeated Calls** — same tool+params 3x in a row.
2. **Oscillating Pattern** — A-B-A-B or A-B-C-A-B-C.
3. **Tool Spam** — 5/6 calls to the same tool, different params.
4. **FindResource/FindNode Cap** — 8 calls (`find_resource_cap` in domain_settings).
6. **No Progress** — journal unchanged for 5 iterations; triggers `NO_PROGRESS_TEMPLATE`.
7. **Max Tool Calls** — domain-configured hard cap; on hit, emits `max_tool_calls_reached` and synthesizes from the journal.

(Layer 5, `RunORKGSPARQL` Cap, applies only to agents with that tool — see [sciqa_agent.md](sciqa_agent.md).)

## Journal (Scratchpad) System

**File:** `ama_kbqa/framework/state.py` — single source of truth (Pydantic `BaseModel`). Both `kqapro_server.py` and `sciqa_server.py` import `JournalState` from here rather than defining their own.

```python
class JournalState(BaseModel):
    question_text: str
    question_type: str
    target_entities: list[str]
    visited_nodes: dict[str, str]       # {node_id: node_name}
    verified_facts: list[dict]
    failed_attempts: list[str]          # capped at MAX_FAILED_ATTEMPTS = 10
    found_values: dict[str, dict]       # {entity_id: {attr: value}} + sparql_result_N
    current_plan: list[str]
    completed_steps: list[str]          # capped at MAX_COMPLETED_STEPS = 20
    partial_answer: str
```

### Three Distinct Buckets

| Field | Role | Rendered as |
|-------|------|-------------|
| `found_values` | **Answer-bearing values.** Synthesis receives *only* the rendered journal summary (system + journal + query, no conversation history). If an answer isn't here, synthesis cannot see it. | `📊 DISCOVERED VALUES` |
| `verified_facts` | Relation triples confirmed true (subject → relation → target); provenance tracking. | `🔗 VERIFIED FACTS` (up to 15 triples) |
| `visited_nodes` | Label resolution map `{id: human_label}`, used at render time to resolve opaque IDs. | Referenced inline |

**Architectural invariant:** every answer-producing tool MUST write to `found_values`. Before April 2026, `GetRelationDetails` only wrote to `verified_facts`, making relation-based answers invisible to synthesis (root cause of a class of "⚠️ NO VALUES DISCOVERED YET" bugs). If you add a tool that produces an answer value, it must write to `found_values`.

**Auto-updates:** `visited_nodes` ← `FindNode` (dedup: skips Qdrant search if label cached), `GetNodeLabel`, `BatchGetNodeLabels`. `found_values` ← `GetAttributeDetails`, `GetNodeSummary`, `RunSPARQL` (`sparql_result_N`), `GetRelationDetails` (dual-write, see below), `FilterEntities`. `verified_facts` ← `GetRelationDetails` and other relation tools.

**`GetRelationDetails` dual write:**
```python
# verified_facts (provenance)
{"subject": base_node_id, "relation": relation_name, "related_id": related_id, "direction": direction, "source": "GetRelationDetails"}
# found_values (synthesis visibility)
found_values[base_node_id][relation_name] = [{"value": related_label_or_id, "related_id": related_id, "direction": direction}, ...]
```

**`GetNodeLabel` / `BatchGetNodeLabels` backfill:** after resolving a label, both scan `found_values` for matching `related_id` entries and upgrade `value` from opaque ID to human label.

**`GetJournalSummary` render order:** `📊 DISCOVERED VALUES` (up to 3/attribute) → `🔗 VERIFIED FACTS` (up to 15) → `📈 PROGRESS` → `🔍 EXPLORED NODES` (first 5) → `💭 PARTIAL ANSWER`. Empty-state marker `⚠️ NO VALUES DISCOVERED YET` appears only when `found_values` is empty; `_run_synthesis` keys off these literal renderer strings (not field names) to detect data presence.

**`ManageJournal` actions:** `"set_question"` sets `question_text`; `"update_plan"` splits `content` on newlines into a list; `"update"` general reflection update.

**Scratchpad-enforced reflection:** after each non-journal tool call, a forced text-only LLM call injects `REFLECTION_PROMPT`, extracts LEARNED/PLAN/NEXT, and stores it via `ManageJournal(action="update", ...)`.

**Tool response truncation:** responses over 2000 chars truncated (except journal tools); last 2 preserved in full.

**Concurrent tool execution:** `_execute_tool_calls` validates all calls and runs loop detection sequentially, but executes the validated batch via `asyncio.gather` — independent calls in the same turn run in parallel, each with a correctly-parented child trace span (see [agent_framework.md](agent_framework.md)). Both system prompts instruct models to emit independent lookups as multiple tool calls per turn.

**Journal refresh placement:** appended at the tail (append-only); superseded refreshes stubbed to placeholders during compaction, preserving the prefix-cache key for all earlier messages so the LLM provider can reuse KV-cache entries on every turn instead of invalidating from the replaced index onward. `_next_trim_trigger` uses hysteresis to avoid compaction/refresh alternating every iteration.

### Exact-Constraint Injection (KQAPro-specific)

`KQAProAgent._extract_exact_attribute_constraints()` detects reusable exact constraints — `official name`, `date of birth`, `IAB code`, `ICD-10-CM`, `UMLS CUI`, `ISWC/ISNI`, `known under <identifier>` — and injects an `EXACT ATTRIBUTE CONSTRAINTS DETECTED` block into pre-analysis, telling the agent to use `FindByAttribute` or explicit verification before semantic entity search. The fast path also uses the first exact constraint for reverse lookup.

## Message History Format

```python
[
    {"role": "system", "content": "[System prompt with KBQA rules]"},
    {"role": "user", "content": "[JOURNAL REFRESH]"},
    {"role": "user", "content": "What is the population of Boston?"},
    {"role": "user", "content": "[PRE-ANALYSIS: Question Type, Strategy, etc.]"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "FindNode", "content": "[TRUNCATED if >2000 chars]"},
    {"role": "assistant", "content": "[REFLECTION: LEARNED/PLAN/NEXT]"},
    {"role": "tool", "tool_call_id": "...", "name": "ManageJournal", "content": "Journal updated"},
    {"role": "user", "content": "[SYNTHESIS PROMPT with journal data]"},
    {"role": "assistant", "content": "The population of Boston is..."}
]
```

## KQAPro MCP Tools

See [Project Architecture — §2 MCP Server](project_architecture.md) for the full 29-tool reference (tiers T1 Discovery, T1.5 Filtering, T2 Retrieval, T3 Qualifiers, T4 Verify, Complex, Utility, plus 1 LLM-hidden `GetJournalStateJSON`) — not duplicated here to avoid drift between two copies of the same tool list.

## Batch Processing

See [SOP/running_batch_processing.md](../SOP/running_batch_processing.md) for the unified batch CLI, LLM judge, and fewshot generation.

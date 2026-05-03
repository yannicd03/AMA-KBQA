# SOP: Adding a New KQAPro Tool That Produces Answer Values

This checklist applies when you add a tool to `ama_kbqa/server/kqapro_server.py` that discovers or confirms a fact the agent might use as its final answer.

**Why this matters:** Synthesis runs with minimal context — it sees only the rendered `GetJournalSummary` output (system prompt + journal + query, no conversation history). Any answer value that is not written to `found_values` is invisible to synthesis. This bug manifested in April 2026 when `GetRelationDetails` wrote only to `verified_facts` and synthesis produced "cannot be determined" for relation-based questions.

## Checklist

### 1. Write to `found_values` (required for answer visibility)

```python
# Ensure the entity has a bucket
if entity_id not in session_journal.found_values:
    session_journal.found_values[entity_id] = {}

# Store discovered value(s) as a list
session_journal.found_values[entity_id][attribute_or_relation_name] = [
    {"value": human_label_or_raw_value, "unit": "optional"},
    ...
]
```

If the value is a related entity (another node), also include `"related_id"` so `GetJournalSummary` can resolve it via `visited_nodes` at render time:

```python
session_journal.found_values[entity_id][relation_name] = [
    {"value": label_if_known_else_id, "related_id": related_entity_id, "direction": "forward"},
]
```

### 2. Append to `verified_facts` if it's a relation (provenance tracking)

```python
session_journal.verified_facts.append({
    "subject": entity_id,
    "relation": relation_name,
    "related_id": related_entity_id,
    "direction": "forward",      # or "reverse"
    "source": "YourToolName"
})
```

This populates the `🔗 VERIFIED FACTS` block in the journal summary and provides audit trail. Skip for pure attribute values (which are better represented directly in `found_values`).

### 3. Update `visited_nodes` for any entity referenced

```python
session_journal.visited_nodes[entity_id] = entity_label
```

If you have only an opaque ID and not yet the label, defer to `GetNodeLabel` / `BatchGetNodeLabels`. Those tools will also backfill any `found_values` entries whose `related_id` matches, so the rendered summary shows human labels.

### 4. Use `add_completed_step()` / `add_failed_attempt()` helpers

Do not append to `completed_steps` or `failed_attempts` directly — use the helpers to respect the caps:

```python
session_journal.add_completed_step(f"Found {n} results for {entity_label}")
session_journal.add_failed_attempt(f"YourTool failed: {error_msg}")
```

### 5. Register the tool in the agent

- **Always available:** add to `CORE_TOOLS` in `ama_kbqa/agents/kqapro_agent/agent.py`
- **Question-type specific:** add to the relevant qtype tool map in `agent.py`
- **Update prompts:** add the tool to the appropriate tier in `SYSTEM_PROMPT` and reference it in any relevant `QTYPE_STRATEGIES` entries in `prompts.py`

### 6. Verify synthesis can see the answer

After implementing the tool, run a question where the answer comes solely from your new tool. Confirm the `GetJournalSummary` output contains the answer in the `📊 DISCOVERED VALUES` section. If it only appears in `🔗 VERIFIED FACTS` but not in `DISCOVERED VALUES`, synthesis will still see it via the `verified facts` marker — but including it in `found_values` as well is safer and more explicit.

---

## Common Anti-Patterns

| Anti-pattern | Consequence | Fix |
|---|---|---|
| Tool returns data in its response JSON but never writes to `found_values` | Synthesis sees "NO VALUES DISCOVERED YET" and returns "cannot be determined" even when the tool loop succeeded | Write to `found_values` in the tool handler |
| Tool writes to `verified_facts` only | Before April 2026 this was the `GetRelationDetails` bug. `verified_facts` is visible in summary but less prominent; synthesis may still miss it if it emits a "cannot answer" phrase | Always also write to `found_values` |
| Storing raw opaque IDs in `found_values["value"]` without a `related_id` backfill path | Rendered summary shows `Q3012` instead of `Ulm` | Include `"related_id"` in the entry so label resolution tools can upgrade it |

---

## Aggregating Tools: Always Use Deterministic SPARQL

When a new tool's primary job is **counting, ranking, or filtering** a potentially large entity set, implement it as a single deterministic SPARQL query (COUNT, ORDER BY LIMIT 1, or UNION) rather than fetching results in Python and counting the list.

**Why:** `FilterEntities` has `limit=50`. Any Python-side count of its output is silently wrong for entity sets > 50. This was the root cause of ~42% of `val.json` questions producing incorrect counts before the April 2026 tool expansion (see `Decisions/kqapro-tool-surface-expansion.md`).

**Pattern:** Issue a single `SELECT (COUNT(DISTINCT ?entity) AS ?count)` query, or `SELECT ?entity ORDER BY ?attr LIMIT 1` for superlatives. Validate with both synthetic rdflib unit tests and real-data tests against Hetzner Virtuoso before merging.

---

## Related Docs

- [System/agent_system.md](../System/agent_system.md) — Journal data model: `found_values`, `verified_facts`, `visited_nodes`, and the synthesis context contract
- [System/project_architecture.md](../System/project_architecture.md) — KQAPro server overview and CORE_TOOLS

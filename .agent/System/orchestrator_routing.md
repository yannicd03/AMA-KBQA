# Orchestrator Routing

Multi-agent router that classifies each question and delegates to `KQAProAgent`
or `SciQAAgent`. Moved out of the former monolithic `agent_system.md` (2026-07-18
split); content unchanged in substance, folded further per commit `f0827ae`
("Fold orchestrator routing into one LLM round-trip").

## Related Docs
- [Agent System (index)](agent_system.md) — map of all agent-system docs
- [Agent Framework](agent_framework.md) — `BaseKBQAAgent` mechanics shared with sub-agents
- [Decisions/orchestrator-evidence-based-routing.md](../Decisions/orchestrator-evidence-based-routing.md) — full ADR, rejected alternatives
- [Decisions/multiturn-direct-agent-conversation.md](../Decisions/multiturn-direct-agent-conversation.md) — multiturn scoping for directly-selected agents

---

**Files:**
- `ama_kbqa/agents/orchestrator_agent/agent.py` — `OrchestratorAgent` class, `_route_autonomously`, `_fallback_kqapro`, `_delegate`
- `ama_kbqa/server/orchestrator_server.py` — MCP server with `analyze_query_recommend_db` tool

## Routing Flow

One LLM round-trip via `_route_autonomously`:

```
Step 1 — Probe (direct MCP call, no LLM round-trip)
  analyze_query_recommend_db(query) called directly via MCP client.
  Tool returns: raw evidence JSON (see below).
  If probe raises: degraded flag set, kg_evidence is empty.

Step 2 — Judge (single LLM call)
  LLM receives: routing system prompt + evidence as tool-result message.
  Forced tool call (tool_choice=required): select_agent(agent: enum, reason: str).
  enum values constructed from _agent_config keys at runtime.
  reason recorded on the classify span as route_reason attribute.
```

If `_route_autonomously` raises or returns an unrecognised agent key, `_fallback_kqapro` delegates to KQAPro. If the probe itself fails, the LLM routes from question domain alone (no unconditional KQAPro fallback).

The earlier two-round-trip version (step 1 forced the LLM to call the probe tool verbatim, carrying zero information and costing a full extra round-trip) was folded into the above single-round-trip flow.

## `analyze_query_recommend_db` Evidence Contract

The tool returns structured JSON — never a verdict string:

```json
{
  "semantics": {
    "named_entities": ["entity name", ...],
    "question_domain_hint": "scholarly publication | general knowledge | ambiguous"
  },
  "kg_evidence": {
    "kqapro": {
      "terms_probed": 2, "terms_matched": 1, "avg_score": 0.84,
      "matches": [{"id": "Q12345", "label": "Ada Lovelace", "score": 0.84}]
    },
    "sciqa": {
      "terms_probed": 2, "terms_matched": 2, "avg_score": 0.79,
      "matches": [{"id": "R123456", "label": "Comparison of NLP benchmarks", "score": 0.82}]
    }
  },
  "degraded": false,
  "note": null
}
```

When Qdrant is unreachable or NER fails, `"degraded": true` is set with an explanatory `note`; `kg_evidence` is empty. The LLM still calls `select_agent` in step 2, routing from question domain alone.

## Agent Domain Descriptions (`_agent_config`)

Injected verbatim into the routing system prompt:

| Agent key | Domain description |
|-----------|-------------------|
| `kqapro` | Wikidata-style general knowledge (people, places, events, facts) |
| `sciqa` | ORKG scholarly knowledge (papers, authors, comparisons, research fields) |

## Trace Span: `route_reason`

The `reason` string from `select_agent` is stored on the `classify` span as `route_reason`. Every routing decision is auditable in the Trace Inspector without re-running the question.

## Key Design Invariants

- Substring parsing of verdict text is gone; routing is entirely determined by the `select_agent` structured call.
- The probe (`analyze_query_recommend_db`) is called directly via MCP — the only LLM round-trip in `_route_autonomously` is the judge call that receives probe evidence and forces `select_agent`.
- If the probe fails, routing degrades to a domain-only decision; it no longer unconditionally falls back to KQAPro.
- The Orchestrator stays stateless across turns — no KG-specific session state; each turn creates a fresh routing context.
- Multiturn conversation is scoped to directly-selected sub-agents only. When the user selects "Orchestrator" directly, each turn creates a fresh agent per the Orchestrator path.
- **MCP client instances are not deduped across the probe call and the delegate call** — an A1 dedup change was drafted on a superseded branch and did not land (2026-07-18 reconciliation). Do not assume a shared/cached `MCPClient` exists here without checking `agent.py` directly.

See `Decisions/orchestrator-evidence-based-routing.md` for the full rationale, rejected alternatives, and trade-offs.

# Orchestrator Routing

Multi-agent router that classifies each question and delegates to `KQAProAgent`
or `SciQAAgent` — or, in Federated mode, to both concurrently with answer
fusion. Moved out of the former monolithic `agent_system.md` (2026-07-18
split); content unchanged in substance, folded further per commit `f0827ae`
("Fold orchestrator routing into one LLM round-trip"). Federated dispatch
(Router/Federated per-instance mode switch) ported 2026-09-14.

## Related Docs
- [Agent System (index)](agent_system.md) — map of all agent-system docs
- [Agent Framework](agent_framework.md) — `BaseKBQAAgent` mechanics shared with sub-agents
- [Decisions/orchestrator-evidence-based-routing.md](../Decisions/orchestrator-evidence-based-routing.md) — full ADR for the one-round-trip evidence-based routing flow, rejected alternatives
- [Decisions/federated-dispatch-and-fusion.md](../Decisions/federated-dispatch-and-fusion.md) — full ADR for federated dispatch, fusion, MCP-close-in-own-task, and the Router/Federated mode switch
- [Decisions/multiturn-direct-agent-conversation.md](../Decisions/multiturn-direct-agent-conversation.md) — multiturn scoping for directly-selected agents

---

**Files:**
- `ama_kbqa/agents/orchestrator_agent/agent.py` — `Orchestrator` class, `_route_autonomously`, `_delegate`, `_run_specialist`, `_federate`, `_fuse_answers`, `_fallback_kqapro`
- `ama_kbqa/server/orchestrator_server.py` — MCP server with `analyze_query_recommend_db` tool

## Router vs. Federated mode

`Orchestrator(session_id="default", federation: Optional[bool] = None)`. `federation=None` (the default) follows `get_federation_enabled()` / `[federation]` in `config.toml`; passing `True`/`False` overrides the config per instance. The demo frontend uses this to offer two orchestrators side by side — a "Router" instance (`federation=False`) and a "Federated" instance (`federation=True`) — from one shared config.

**Router mode** (federation disabled, the default) sends the LLM the exact pre-federation single-dispatch contract: `select_agent(agent: enum, reason: str)` and the routing prompt without the federation addendum, byte-for-byte unchanged. This is deliberate, not incidental — the paper's routing-accuracy benchmark numbers were measured against this exact contract, and `feat/langgraph-rewrite`'s probe/`select_agent` graph nodes mirror it too. A snapshot test (`tests/agents/test_orchestrator_federation_mode.py::TestRouterVsFederatedToolGating`) pins the tool schema and prompt so an accidental edit fails loudly.

**Federated mode** sends `select_agents(agents: array[enum], reason: str)` (`minItems=1`, `maxItems=max(1, max_specialists)`) plus a prompt addendum instructing the LLM to select more than one agent only when the probe evidence shows strong matches in more than one graph, or the question genuinely spans domains.

`_route_autonomously` always returns `Optional[List[str]]` regardless of mode (a single-element list in Router mode) — see Routing Flow below.

## Routing Flow

One LLM round-trip via `_route_autonomously`, in both modes:

```
Step 1 — Probe (direct MCP call, no LLM round-trip)
  analyze_query_recommend_db(query) called directly via MCP client.
  Tool returns: raw evidence JSON (see below).
  self.last_routing_evidence is set to the (possibly degraded) evidence JSON
  right after the probe — reused as a fusion prior in Federated mode.
  If probe raises: degraded flag set, kg_evidence is empty.

Step 2 — Judge (single LLM call, mode-gated tool)
  LLM receives: routing system prompt (+ federation addendum if enabled)
    + evidence as a user-message block.
  Forced tool call (tool_choice=required):
    Router mode:    select_agent(agent: enum, reason: str)
    Federated mode: select_agents(agents: array[enum] (maxItems capped), reason: str)
  enum values constructed from _agent_config keys at runtime.
  reason recorded on the classify span as route_reason attribute.
  Unknown agent name(s), an empty selection, or no tool call at all -> None.
  Duplicate names are deduped in router order; a selection beyond
  max_specialists is truncated defensively (schema already caps it).
```

If `_route_autonomously` returns `None`, `ask()` falls back to KQAPro (`_fallback_kqapro`). If the probe itself fails, the LLM routes from question domain alone (no unconditional KQAPro fallback) — evidence is still forwarded as a degraded-JSON stub so `last_routing_evidence` is never left unset.

The earlier two-round-trip version (step 1 forced the LLM to call the probe tool verbatim, carrying zero information and costing a full extra round-trip) was folded into the above single-round-trip flow.

## Dispatch: single vs. federated

`ask()` branches on the length of `_route_autonomously`'s result:

- **Zero agents (`None`)** → `_fallback_kqapro(query)`.
- **One agent** → `_delegate(agent_name, query)`: runs the specialist via `_run_specialist`, falls back to KQAPro on any exception.
- **Two or more agents (Federated mode only)** → `_federate(agent_names, query)`: runs every specialist concurrently via `asyncio.gather(..., return_exceptions=True)`, then:
  - all failed → `_fallback_kqapro`;
  - exactly one survived → that answer is returned directly (fusion of one answer is a no-op);
  - two or more survived → `_fuse_answers(query, answers)`.

`_run_specialist(agent_name, query, fallback=False)` is the shared primitive behind all three paths (including the KQAPro fallback itself, via `fallback=True`). It opens a `delegate` span, shares the recorder with the sub-agent (so the sub-agent's spans nest under the delegate span — `asyncio.gather` task-local `ContextVar` copies keep concurrent specialists as siblings, not cross-parented), tags journal snapshots with `source_agent`, and — critically for the MCP teardown risk below — closes the specialist's MCP connection in a `finally` block immediately after `agent.ask()` returns, before returning the handoff record `{"agent", "answer", "scratchpad"}` (or raising, on failure).

`_fuse_answers` runs under a `synthesis` span, injects each specialist's answer plus its scratchpad (rendered by `_extract_scratchpad` from the sub-agent's final `JournalState`) and `last_routing_evidence` as a conflict-resolution prior, and goes through `self._create_with_retry` like every other Orchestrator LLM call. An empty or failed fusion degrades to the first answer in router order rather than discarding successful specialist runs.

## MCP teardown: close in the specialist's own task

`BaseKBQAAgent.ask()` never closes its own MCP connection after answering (comment: `"Question complete (MCP preserved)"`) so a follow-up turn can reuse it. Closing (or garbage-collecting) that connection from a **different** `asyncio` task than the one that opened it trips `anyio`'s `stdio_client` teardown ("Attempted to exit cancel scope in a different task, this is a bug!") — the frontend's `lifecycle_runner.py` already works around this for its own multiturn continuation path by orphaning the stale connection (`agent.mcp = None`) instead of closing it, since a follow-up turn resumes on a new event loop.

Federated dispatch runs each specialist in its own `asyncio.gather` task, so `_run_specialist` closes that specialist's MCP connection itself (`try/finally: await agent.close()`, tolerant of a sub-agent surface without `close()`), still inside the same task that opened it via `agent.ask() -> _init_mcp()`. This is safe in both single dispatch (awaited directly, one task) and federated dispatch (one task per specialist), and bounds a cached specialist's MCP connection to one `_run_specialist` call rather than leaking it — a later call on the same cached instance simply reconnects via `_init_mcp`'s existing fast path. See `Decisions/federated-dispatch-and-fusion.md`'s 2026-09-14 addendum for the full investigation.

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

## Trace Spans: `route_reason`, `route_mode`

The `reason` string from `select_agent`/`select_agents` is stored on the `classify` span as `route_reason`. A `route_mode` attribute (`"single"` | `"federated"`) is also set on every `classify` span, derived from `len(selected_agent_names) > 1` — this is `"single"` in Router mode always, and in Federated mode whenever the LLM itself chose one agent. The `delegate` span for a federated run's fusion step is a `synthesis` span (kind `synthesis`, name `fuse`); the frontend's lifecycle mapping (`ama_kbqa/frontend/utils/lifecycle_mapping.py`) reads `kind`/`name`/`phase`/`attributes["sub_agent"]` only, so these additions don't require frontend changes. Every routing decision is auditable in the Trace Inspector without re-running the question.

## Key Design Invariants

- Substring parsing of verdict text is gone; routing is entirely determined by the `select_agent`/`select_agents` structured call.
- The probe (`analyze_query_recommend_db`) is called directly via MCP — the only LLM round-trip in `_route_autonomously` is the judge call that receives probe evidence and forces the mode-gated decision tool.
- If the probe fails, routing degrades to a domain-only decision; it no longer unconditionally falls back to KQAPro.
- The Orchestrator stays stateless across turns — no KG-specific session state; each turn creates a fresh routing context.
- Multiturn conversation is scoped to directly-selected sub-agents only. When the user selects "Orchestrator" directly, each turn creates a fresh agent per the Orchestrator path.
- **MCP client instances are not deduped across the probe call and the delegate call** — an A1 dedup change was drafted on a superseded branch and did not land (2026-07-18 reconciliation). Do not assume a shared/cached `MCPClient` exists here without checking `agent.py` directly.
- Router mode's `select_agent` tool schema and system prompt are byte-for-byte the pre-federation contract (see Router vs. Federated mode above) — do not "simplify" it into a `maxItems=1` `select_agents` call; a snapshot test guards this.
- The graph engine (`[agent].engine = "graph"`, `ama_kbqa/graph/orchestrator.py`) has no fan-out/fusion nodes yet, so it models Router mode only. As implemented (2026-09-20), `Orchestrator.ask()` does not degrade a Federated instance to single dispatch on the graph engine — it checks `self._federation_enabled` before graph dispatch and falls back to the full legacy body (routing + fan-out + fusion) with a traced warning, so Federated mode stays genuinely federated regardless of the configured engine. See `System/graph_engine.md`'s "What the graph engine does not cover yet" section and `Decisions/federated-dispatch-and-fusion.md`'s 2026-09-14 addendum point 6.

See `Decisions/orchestrator-evidence-based-routing.md` for the one-round-trip evidence-based routing rationale, and `Decisions/federated-dispatch-and-fusion.md` for federated dispatch, fusion, and the Router/Federated mode switch.

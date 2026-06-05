# ADR: Orchestrator Evidence-Based Two-Step Routing

**Date:** 2026-06-05  
**Commits:** `7248aaf` (demo-bwcloud), `ab79f62` (yannic-dev)  
**Status:** Shipped — deployed to Hetzner and bwcloud on 2026-06-05  
**Files:** `ama_kbqa/agents/orchestrator_agent/agent.py`, `ama_kbqa/server/orchestrator_server.py`, `scripts/benchmark_routing.py`, `tests/server/test_orchestrator_server.py`, `tests/agents/test_orchestrator_routing.py`

## Related Docs
- [System/agent_system.md](../System/agent_system.md) — Orchestrator routing section; span placement table
- [System/project_architecture.md](../System/project_architecture.md) — MCP server overview; `orchestrator_server.py` entry

---

## Problem

The previous routing path had five compounding issues that caused SciQA questions to be frequently misrouted to KQAPro.

**1. Collapsed verdict hid evidence.**  
`analyze_query_recommend_db` (in `orchestrator_server.py`) collapsed Qdrant entity-linking results into a text string: `"Use KQAPro"` or `"Use SciQA"`. The decision gate was `avg_confidence > 0.7`. The LLM received only the verdict, not the underlying match data, so the routing prompt could not reason about *why* a knowledge graph was recommended.

**2. `score_threshold` / verdict gate mismatch (commit `ba23088` collateral damage).**  
A prior commit lowered the shared Qdrant `score_threshold` from 0.7 to 0.6 to improve entity-linking recall. The verdict gate remained at `> 0.7`. Result: matches in the 0.6–0.7 score band counted as found entities but dragged the average below the gate, flipping the verdict to KQAPro. This was silent — nothing in the logs revealed the mismatch.

**3. Scoring ignored term coverage.**  
The average was computed over all matched entities regardless of how many terms from the query were actually matched. One match at score 0.92 could beat five matches at score 0.85 even if the five-match case meant all five query terms were grounded in a knowledge graph.

**4. Spurious matches were invisible.**  
The matched entity labels were not surfaced to the routing LLM. A Qdrant vector match might superficially resemble a KG term without being relevant (e.g., a paper title fragment matching an ORKG resource). Without labels, the LLM could not detect this.

**5. Ties and failure paths silently defaulted to KQAPro.**  
Any error in the tool — NER failure, Qdrant unavailable, equal-confidence case — collapsed to a KQAPro recommendation. The degraded path was invisible in logs and indistinguishable from a confident decision.

**The LLM call was decorative.** The tool was the only one available (`tool_choice=required`), so the LLM was forced to call it regardless of the question, and the returned verdict left nothing to reason about. Agent descriptions were present in `_agent_config` but never reached the routing prompt.

---

## Decision: Return raw evidence; route in two LLM steps

### Tool contract change (`orchestrator_server.py`)

`analyze_query_recommend_db` now returns a structured JSON evidence object instead of a verdict string:

```json
{
  "semantics": {
    "named_entities": ["author name", "comparison title"],
    "question_domain_hint": "scholarly publication / general knowledge / ambiguous"
  },
  "kg_evidence": {
    "kqapro": {
      "terms_probed": 2,
      "terms_matched": 1,
      "avg_score": 0.84,
      "matches": [{"id": "Q12345", "label": "Ada Lovelace", "score": 0.84}]
    },
    "sciqa": {
      "terms_probed": 2,
      "terms_matched": 2,
      "avg_score": 0.79,
      "matches": [
        {"id": "R123456", "label": "Comparison of NLP benchmarks", "score": 0.82},
        {"id": "R654321", "label": "benchmark dataset", "score": 0.76}
      ]
    }
  },
  "degraded": false,
  "note": null
}
```

When the evidence path is degraded (Qdrant unreachable, NER failure), the payload sets `"degraded": true` with an explanatory `note` and omits or empties `kg_evidence`. The router then decides from question domain alone.

### Two-step routing in `Orchestrator._route_autonomously` (`agent.py`)

**Step 1 — probe:** The orchestrator's LLM is given the routing system prompt (which includes agent domain descriptions from `_agent_config`) and is forced via `tool_choice=required` to call `analyze_query_recommend_db`. The tool returns the evidence JSON as a tool message.

**Step 2 — judge:** The evidence is appended as a tool-result message and the LLM is forced via `tool_choice=required` to call `select_agent(agent: enum, reason: str)`. The enum is constructed from `_agent_config` keys at runtime. The `reason` string is recorded on the `classify` trace span as `route_reason` so every routing decision is auditable.

**Agent domain descriptions in `_agent_config`:**
- KQA Pro: "Wikidata-style general knowledge (people, places, events, facts)"
- SciQA: "ORKG scholarly knowledge (papers, authors, comparisons, research fields)"

These descriptions are injected into the routing system prompt verbatim so the LLM knows what each agent covers when inspecting the evidence labels.

**Degraded path:** If the evidence payload is `degraded: true`, the LLM still calls `select_agent` in step 2 — it must decide from the question text and domain hint alone. Previously all degraded paths returned KQAPro unconditionally.

**Substring matching removed.** The old `_route_autonomously` parsed the verdict string with substring tests (`"kqa" in verdict_text`). This is gone; routing is entirely determined by the `select_agent` call.

**Fallback unchanged.** If `_route_autonomously` returns `None` (exception, or `select_agent` returns an unrecognised agent key), `_fallback_kqapro` is invoked as before.

---

## Rejected alternatives

### Single-call: orchestrator calls MCP tool directly and decides in one LLM turn

The orchestrator could call `analyze_query_recommend_db` itself (without making the LLM issue the tool call) and then pass the evidence directly to a single LLM call that returns a routing decision. This would save one LLM round-trip.

**Rejected because:**
- It requires the orchestrator code to hard-code the tool call and manage the evidence-injection logic in Python rather than in the message conversation. This is less flexible: adding a new probe (e.g., a second entity-linking pass or a query-rewrite step) requires Python changes rather than prompt changes.
- The two-call shape is more agentic: the routing LLM participates in deciding *what to probe*, not just in interpreting pre-fetched results. Future routing improvements (multi-turn clarification, confidence thresholds expressed as LLM judgment) can be added without restructuring the Python control flow.
- The two-call form keeps the Orchestrator's decision logic in LLM message space, where it is visible in traces and testable via prompt injection.

### Pure threshold realignment (repair the 0.6/0.7 mismatch)

Align `score_threshold` and the verdict gate back to the same value, or compute a better scoring metric (e.g., weighted by term coverage).

**Rejected because:**
- The LLM's role would remain decorative: one tool, forced call, verdict string. Any threshold fix is fragile against future config changes.
- This does not solve the spurious-match invisibility problem: the LLM still cannot see entity labels.
- Domain-hint routing (question semantics when KG evidence is ambiguous) is not expressible via thresholds.
- A threshold calibration would need a routing benchmark to validate; the evidence-based approach is self-validates through the `route_reason` trace attribute.

---

## New artifacts

### `scripts/benchmark_routing.py`

Route-only benchmark script. Samples labeled questions from both datasets via `benchmark_agents.load_raw_dataset`, runs the orchestrator's `_route_autonomously` path (no sub-agent execution), and reports confusion matrix + per-class accuracy. Requires Qdrant and a provider key. Intended for before/after comparison on the servers, not in CI.

### Tests

- `tests/server/test_orchestrator_server.py` — modernised to the evidence contract (no `avg_confidence > 0.7` assertions); new NER-failure test asserts `degraded: true` payload.
- `tests/agents/test_orchestrator_routing.py` — 8 hermetic tests for the two-step flow: mock tool return, verify `select_agent` call shape, `route_reason` span attribute, fallback on unknown agent key, degraded-payload branch.

---

## Trade-offs accepted

| Aspect | Trade-off |
|--------|-----------|
| Determinism | Routing is now LLM-judged. Two identical questions may route differently across temperature restarts. The heuristic gate was fully deterministic. |
| Latency | Two LLM round-trips instead of one. Both are small (system prompt + evidence JSON, no tool-loop context). Acceptable at interactive latency. |
| Auditability | Gained: `route_reason` is stored on the `classify` span and is visible in the Trace Inspector. Lost: the 0.7 threshold was self-documenting in the code. |
| Degraded-path quality | Improved: degraded cases now reach the LLM as `degraded:true` evidence rather than silently producing a KQAPro verdict. |

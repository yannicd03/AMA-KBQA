# ADR: Federated Multi-Specialist Dispatch and Answer Fusion

**Date:** 2026-06-07
**Status:** Implemented — committed on `feature/federated-retrieval` (commit `092f66d`), rebased onto `main`
**Files:** `ama_kbqa/agents/orchestrator_agent/agent.py`, `ama_kbqa/config.py`, `config.toml`, `config.docker.toml`, `scripts/benchmark_routing.py`, `tests/agents/test_orchestrator_routing.py`

## Related Docs
- [Decisions/orchestrator-evidence-based-routing.md](./orchestrator-evidence-based-routing.md) — prior ADR; two-step evidence-based routing this ADR extends. Rationale there remains intact.
- [System/agent_system.md](../System/agent_system.md) — Orchestrator Deep Dive; routing flow, span table, `_delegate` / `_fallback_kqapro`
- [System/project_architecture.md](../System/project_architecture.md) — MCP server overview; federation config section

---

## Problem

The orchestrator implemented in `orchestrator-evidence-based-routing.md` performs a two-step evidence-based routing that concludes with a `select_agent(agent: enum, reason: str)` call — a single-valued enum. This is sufficient when evidence clearly favours one knowledge graph, but two failure modes remained unaddressed:

1. **Cross-domain questions.** A question such as "Which papers cite Ada Lovelace?" spans both the ORKG scholarly graph (SciQA) and Wikidata-style entities (KQAPro). Routing to one specialist discards relevant evidence from the other.
2. **Ambiguous probe signals.** When both `kg_evidence.kqapro` and `kg_evidence.sciqa` show high `terms_matched` / `avg_score`, the router LLM must arbitrarily pick one. The paper's "agent swarm" figure had always described this as a fan-out scenario.

The single-dispatch constraint also meant that the `classify` span's route selection was binary even when the probe evidence was genuinely multi-valued.

---

## Decision: Extend step 2 to `select_agents` (array) with conditional fan-out

### Tool contract change — step 2 forced call

The step-2 forced tool is renamed from `select_agent` (scalar enum) to `select_agents`:

```json
{
  "name": "select_agents",
  "schema": {
    "agents": {
      "type": "array",
      "items": { "type": "string", "enum": ["kqapro", "sciqa", ...] },
      "minItems": 1,
      "maxItems": <federation_cap>
    },
    "reason": { "type": "string" }
  }
}
```

`maxItems` is set to `get_federation_max_specialists()` at routing time. When federation is **disabled** (the default — see Config section below), `maxItems = 1`, which degenerates the schema to the original single-dispatch contract. The router LLM physically cannot return more than one agent; existing benchmark numbers are protected.

The routing system prompt instructs the LLM to federate only when the probe evidence shows strong matches in more than one graph **or** the question genuinely spans domains. It must not federate by default.

The `reason` string is stored on the `classify` span as `route_reason` (unchanged) and a new `route_mode` attribute records `"single"` or `"federated"`.

### `_run_specialist(agent_name, query, fallback=False)` — new shared primitive

A single coroutine that encapsulates the full delegate-and-trace lifecycle for one specialist:

- Opens a `delegate` span (kind `delegate`, name = agent key).
- Initialises the sub-agent recorder sharing as before.
- Tags journal snapshots with `source_agent = agent_name` so fused answers carry per-source attribution.
- Returns `(agent_name, answer_text)` on success, `(agent_name, None)` on exception.
- Accepts `fallback=True` so `_fallback_kqapro` can route through the same path (side fix — see below).

### `_federate(agent_names, query)` — concurrent fan-out

```python
results = await asyncio.gather(
    *[self._run_specialist(name, query) for name in agent_names],
    return_exceptions=False
)
```

Each `_run_specialist` call opens its own `delegate` span. Because `TraceRecorder` uses a `ContextVar` and `asyncio.gather` runs coroutines as independent tasks with task-local contextvars, the spans nest as siblings under the root `agent_run` span — not as grandchildren of one another.

**Failure isolation:**
- One specialist raises or returns `None` → the surviving answer is used directly (no fusion call needed; fusion of one answer is a no-op).
- All specialists fail → fall back to KQAPro via the existing `_fallback_kqapro` path (mirrors `_delegate` policy).

### `_fuse_answers(answers, query)` — single LLM synthesis call

Called only when two or more specialists return non-empty answers.

**Inputs injected into the fusion prompt:**
1. Per-agent answer blocks tagged with agent key and domain description.
2. `self.last_routing_evidence` — the raw evidence JSON from step 1 of `_route_autonomously`, newly stored as an instance attribute. Provides domain match scores as a fusion prior so the LLM can weigh conflicting answers by evidence strength rather than surface similarity.

**Conflict policy (system prompt):**
| Situation | Instruction |
|-----------|-------------|
| Answers agree | Return a single synthesised answer |
| Answers are complementary | Merge with per-source attribution (e.g., "According to KQAPro … According to SciQA …") |
| Answers conflict | Present both, name their sources, and use routing evidence scores to judge which is more likely correct |
| Neither answer contains the relevant fact | Do not invent; acknowledge the gap |

**Span:** `_fuse_answers` runs under a `synthesis` span (kind `synthesis`). The existing frontend lifecycle mapping already lights the `post_synthesis` phase on `synthesis` spans, so no frontend changes are needed.

**Degrade path:** If fusion returns an empty string or raises, the first answer in router order is returned rather than discarding successful specialist results.

### Side fix A — `_fallback_kqapro` now traces under a `delegate` span

Previously `_fallback_kqapro` ran its sub-agent call outside any `delegate` span. Its child spans landed in the same trace but had `agent_run` (the root) as their effective parent, making the fallback invisible in the delegate-centric lifecycle view. `_fallback_kqapro` now calls `_run_specialist("kqapro", query, fallback=True)`, gaining proper span nesting with no behaviour change.

### Side fix B — journal snapshots tagged with `source_agent`

`_run_specialist` writes `source_agent = agent_name` onto every journal snapshot it collects from the sub-agent. After a federated merge the Chat UI can display which KG contributed each snapshot step.

---

## Configuration

New `[federation]` section in `config.toml` and `config.docker.toml`:

```toml
[federation]
enabled = false
max_specialists = 2
```

Getters in `ama_kbqa/config.py`:
- `get_federation_enabled() -> bool`
- `get_federation_max_specialists() -> int`

**Why a dedicated section:** The prior lesson from `score_threshold` (documented in `orchestrator-evidence-based-routing.md`) is that shared config knobs silently couple unrelated consumers when a value is changed. Federation is an orchestrator-level policy; it has no meaning to sub-agents or MCP servers. A dedicated section makes the scope explicit and prevents accidental coupling.

---

## Benchmark and test updates

**`scripts/benchmark_routing.py`:** normalises the new list return. A `"+"`-joined multi-label (e.g., `"kqapro+sciqa"`) is counted as a miss for the single-dispatch routing accuracy metric. This keeps the existing benchmark comparable to pre-federation runs.

**`tests/agents/test_orchestrator_routing.py`:** updated for the `select_agents` schema + 12 new tests covering:
- Federated selection (two agents returned, `asyncio.gather` fanout)
- Cap / dedupe defence (cap enforced by schema, duplicate agent keys deduplicated before dispatch)
- Sibling `delegate` spans under `asyncio.gather` (ContextVar task-locality)
- `source_agent` tag on journal snapshots after a federated merge
- Failure degradation (one specialist fails → surviving answer used; all fail → KQAPro fallback)
- Fusion prompt construction (evidence injected as `last_routing_evidence`)
- Fusion degrade path (empty fusion → first answer in router order)

Total test suite: **301 passed**.

---

## Rejected alternatives

### Explicit `mode` enum parameter (`"single"` / `"federated"`)

Adding a separate `mode` field alongside the `agents` array is redundant: mode is already fully determined by `len(agents)`. A two-field schema can also be contradictory (`mode="single"`, `agents=["kqapro","sciqa"]`). Rejected.

### Rank-fusion (RRF) of answers

Reciprocal rank fusion aggregates ranked lists, not free-text answers. Two synthesised answers with attached evidence trails is a semantic reconciliation problem — the evidence scores are the ranking signal — not a ranking problem. An RRF pass would lose the reasoning content. Rejected: LLM-judge synthesis with evidence prior is the right layer.

### Always-federate (remove the `enabled` flag)

Always fanning out to all available specialists doubles the token cost on every question and invalidates benchmark comparability by mixing architectures mid-run. The default-disabled flag preserves single-dispatch semantics for benchmarking while making federation available for interactive demo use. Rejected as the default.

---

## Future work (out of scope for this ADR)

True cross-KG join queries (e.g., "Which ORKG papers cite entities found in KQAPro?") require a decomposition planner that generates sub-queries against each KG's abstract-operations contract and then joins the result sets. This ADR covers fan-out over independent specialists and single-pass fusion only; structured cross-KG joins are a separate, more invasive problem.

---

## Trade-offs accepted

| Aspect | Trade-off |
|--------|-----------|
| Default behaviour | `enabled = false` means federation is opt-in. Interactive demo users get it; batch benchmarks stay comparable. |
| Token cost | Federated mode: two specialist runs + one fusion LLM call per question. Acceptable at demo latency; unsuitable for bulk benchmarks without explicit opt-in. |
| ContextVar task-locality | `asyncio.gather` isolates `TraceRecorder` contextvar per task. Sibling spans are correct; sub-agent spans do not accidentally cross-parent. Tested. |
| Fusion quality | Conflict resolution is LLM-judged with evidence scores as a prior. The LLM may still misjudge conflicts; the fusion prompt encodes best-effort policy, not a deterministic rule. |
| Fallback span nesting | Side fix A is a behaviour-neutral trace improvement. It cannot regress answers; it only changes how fallback spans appear in the inspector. |

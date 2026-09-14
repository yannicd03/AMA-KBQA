# ADR: Federated Multi-Specialist Dispatch and Answer Fusion

**Date:** 2026-06-07
**Status:** Implemented — committed on `feature/federated-retrieval` (commit `092f66d`), rebased onto `main`. Ported onto `demo-v2` 2026-09-14 (per-instance mode switch) — see the dated addendum below.
**Files:** `ama_kbqa/agents/orchestrator_agent/agent.py`, `ama_kbqa/config.py`, `config.toml`, `config.docker.toml`, `scripts/benchmark_routing.py`, `tests/agents/test_orchestrator_routing.py`, `tests/agents/test_orchestrator_federation_mode.py`

## Related Docs
- [Decisions/orchestrator-evidence-based-routing.md](./orchestrator-evidence-based-routing.md) — prior ADR; two-step evidence-based routing this ADR extends. Rationale there remains intact.
- [System/orchestrator_routing.md](../System/orchestrator_routing.md) — Orchestrator routing flow, span table, `_delegate` / `_run_specialist` / `_federate` / `_fuse_answers` (content moved here from `agent_system.md` in the 2026-07-18 doc split, before this ADR's port)
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
- Returns a handoff record `{"agent", "answer", "scratchpad"}` on success; raises on failure (see the port addendum for the exact record shape, added in a later commit).
- Accepts `fallback=True` so `_fallback_kqapro` can route through the same path (side fix — see below).

### `_federate(agent_names, query)` — concurrent fan-out

```python
results = await asyncio.gather(
    *[self._run_specialist(name, query) for name in agent_names],
    return_exceptions=True,
)
```

Each `_run_specialist` call opens its own `delegate` span. Because `TraceRecorder` uses a `ContextVar` and `asyncio.gather` runs coroutines as independent tasks with task-local contextvars, the spans nest as siblings under the root `agent_run` span — not as grandchildren of one another.

**Failure isolation:**
- One specialist raises → the surviving answer is used directly (no fusion call needed; fusion of one answer is a no-op).
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

**`tests/agents/test_orchestrator_routing.py`:** updated for the `select_agents` schema + new tests covering:
- Federated selection (two agents returned, `asyncio.gather` fanout)
- Cap / dedupe defence (cap enforced by schema, duplicate agent keys deduplicated before dispatch)
- Sibling `delegate` spans under `asyncio.gather` (ContextVar task-locality)
- `source_agent` tag on journal snapshots after a federated merge
- Failure degradation (one specialist fails → surviving answer used; all fail → KQAPro fallback)
- Fusion prompt construction (evidence injected as `last_routing_evidence`)
- Fusion degrade path (empty fusion → first answer in router order)

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

---

## Addendum 2026-09-14: ported onto `demo-v2` with a per-instance mode switch

This branch (`feature/federated-retrieval`, commits `034dddc`/`7195f60`/`67eb6b7`/`98882ac`) forked from an old `main` (`5b4d7f8`). Independently, `dev` reworked the orchestrator's routing mechanics before this ADR's commits could be fast-forwarded: the two-LLM-call router (step 1 forces the probe tool call, step 2 judges) was folded into a **one-round-trip** flow (the probe is called directly via MCP, no LLM involved; a single forced decision call follows), `TransientRetry`/`_create_with_retry` was wired into every LLM call, and `demo-v2` additionally removed the LLM-only last-resort fallback (`_fallback_kqapro` now raises if the KQAPro agent can't load, instead of degrading to a knowledge-base-free LLM answer). This ADR's `_route_autonomously` rewrite conflicted directly with that rework.

**Porting approach:** `git cherry-pick -n 034dddc 98882ac` onto `demo-v2` (bc9d6e2), then hand-resolved the conflicts in `ama_kbqa/agents/orchestrator_agent/agent.py`, `tests/agents/test_orchestrator_routing.py`, `.agent/README.md`, and `.agent/System/agent_system.md` (superseded by `.agent/System/orchestrator_routing.md` after the 2026-07-18 doc split; that file is what actually changed) by rebuilding this ADR's federation logic on top of `dev`'s one-round-trip structure rather than reintroducing the two-call flow.

**What changed from the original design:**

1. **Per-instance override.** `Orchestrator.__init__` gained a `federation: Optional[bool] = None` parameter. `self._federation_enabled = get_federation_enabled() if federation is None else bool(federation)`. This lets the demo frontend construct two orchestrator instances from one config — `Orchestrator(federation=False)` ("Router") and `Orchestrator(federation=True)` ("Federated") — without touching `config.toml`, which still supplies the default for any caller that doesn't pass the kwarg (e.g. the benchmark script, the MCP-hosted server path).

2. **Router mode is byte-for-byte `dev`'s single-dispatch router — the deliberate departure from this ADR's original design.** The original design always sent `select_agents` (an array capped at `maxItems=1` when disabled) so that "disabled" and "enabled with one selection" shared one code path. On `demo-v2` this is split explicitly: with federation off, `_route_autonomously` sends `dev`'s exact pre-federation `select_agent` tool (string enum, unchanged schema/description down to the words) and the pre-federation routing system prompt (no federation addendum). This is not cosmetic — the paper's routing-accuracy benchmark numbers were measured against that exact single-agent contract, and the in-progress `feat/langgraph-rewrite` branch's probe/`select_agent` graph nodes were built to match it too. A `maxItems=1` array would have been behaviorally equivalent for a compliant provider but is a different wire contract, and this port chooses not to risk that divergence. A snapshot test (`tests/agents/test_orchestrator_federation_mode.py::TestRouterVsFederatedToolGating`) pins the exact tool schema and prompt captured from `demo-v2` immediately before this port, so a future edit that touches Router mode's contract fails loudly.

   Federated mode is unchanged from the original design: `select_agents` (array, `maxItems=max(1, max_specialists)`), the federation prompt addendum, and the dedupe/unknown-name/cap-enforcement parsing described above.

   Both modes now use `dev`'s one-round-trip structure underneath: the probe (`analyze_query_recommend_db`) is always called directly via MCP (never as a forced LLM tool call), and `self.last_routing_evidence` is set immediately after the probe (success or the constructed degraded-JSON fallback) so it is always available as a fusion prior — this ADR's design only set it after a successful two-call flow.

3. **`_route_autonomously` always returns `Optional[List[str]]`**, even in Router mode (a single-element list rather than a bare string). `ask()`, `_delegate`/`_federate` dispatch, `scripts/benchmark_routing.py` (joins a multi-agent selection with `"+"` for the routing-accuracy metric, as before), and the LangGraph rewrite's future probe/select_agent nodes all consume the list form.

4. **`_fuse_answers`'s LLM call now goes through `self._create_with_retry`** (the `chatkit.TransientRetry`-backed helper `dev` wired into every other Orchestrator LLM call), instead of a bare `client.chat.completions.create(...)`. This ADR's original fusion call predates that retry wiring; the port closes the gap rather than leaving fusion as the one uninsured LLM call in the dispatch path.

5. **MCP teardown risk, investigated and closed.** `BaseKBQAAgent.ask()` never closes its own MCP connection after answering (`self._trace("Question complete (MCP preserved)")` in `base_agent.py`) — connections are left open so a follow-up turn in the same conversation can reuse them. The frontend's `lifecycle_runner.py` already works around the consequence of this for its own continuation path: because a follow-up turn resumes on a **new** event loop/thread, it cannot safely call `agent.close()` on the stale connection (closing an `anyio`-backed `stdio_client`'s `AsyncExitStack` from a different task than the one that opened it raises "Attempted to exit cancel scope in a different task, this is a bug!"), so it instead orphans it (`agent.mcp = None`) and lets `_init_mcp` rebuild a fresh connection.

   Federated dispatch reintroduces the same hazard in a new shape: `_federate` runs each specialist under `asyncio.gather`, i.e. one `asyncio.Task` per specialist. Before this port, nothing ever closed a cached specialist's MCP connection at all (it lived until the `Orchestrator`/sub-agent instances were garbage-collected) — meaning a specialist's connection could be torn down by the garbage collector from whatever task happened to be running when the last reference dropped, which is exactly the cross-task-close failure mode, just deferred to GC time instead of triggered explicitly.

   Fix: `_run_specialist` now wraps `agent.ask(query)` in a `try/finally` that calls `await agent.close()` (via `getattr(agent, "close", None)`, tolerant of a sub-agent surface without one) immediately afterward, still inside the same coroutine. Because `_run_specialist`'s entire body — from the sub-agent's own `_init_mcp()` inside `ask()` to this `close()` — executes start-to-finish in one task in *both* single dispatch (awaited directly) and federated dispatch (one `asyncio.gather` task per specialist), closing here is always same-task-safe. This also bounds a specialist's MCP connection to one `_run_specialist` call instead of leaking it for the cached agent's lifetime; a later `_run_specialist` call on the same cached instance (e.g. `kqapro_agent` reused as both a direct route and, on a later turn, the fallback) simply reconnects via `_init_mcp`'s existing `if self.mcp: return` fast path finding `self.mcp is None` and rebuilding. `tests/agents/test_orchestrator_federation_mode.py::TestMcpCloseInOwnTask` asserts `close()` is awaited from the same `asyncio.Task` that ran `_run_specialist`, for both single and concurrent federated dispatch, and that a close failure doesn't mask a successful answer.

6. **`feat/langgraph-rewrite` has no federated path.** That branch's engine (still not the demo-v2 default) copies `dev`'s probe/`select_agent` graph nodes but has no fan-out or fusion nodes. Until it gains them, an `Orchestrator`-equivalent built on that engine must reject `federation=True` rather than silently falling back to single dispatch — silent fallback would let a demo operator believe Federated mode is active when it is not. This is a note for whoever wires federation into that engine, not a behavior implemented on `demo-v2`'s (non-LangGraph) `Orchestrator` in this port.

**Test suite:** `demo-v2` baseline 551 passed / ruff clean before this port. After: `tests/agents/test_orchestrator_routing.py` (rewritten fed's `TestFederate`, `TestFuseAnswers`, `TestExtractScratchpad`, `TestRouteAutonomouslyFederated` against `dev`'s one-call `_FakeMcp.call_tool` pattern instead of the old two-call `list_tools`/`call_tool` pattern) plus a new `tests/agents/test_orchestrator_federation_mode.py` (constructor override, Router/Federated tool-gating snapshot, `route_mode` span attribute, fusion-via-retry, MCP-close-in-own-task).

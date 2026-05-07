# ADR: `GetQualifierValue` Tool — Direct Qualifier Projection

**Date:** 2026-05-05  
**Status:** Accepted  
**Code:**
- `ama_kbqa/server/kqapro_server.py` — `async def GetQualifierValue` (inserted between `GetQualifiersByPredicate` and `ManageJournal`, ~165 lines)
- `ama_kbqa/agents/kqapro_agent/agent.py:53-58` — `QTYPE_TOOL_MAP` entries for `QueryAttrQualifier` and `QueryRelationQualifier`
- `ama_kbqa/agents/kqapro_agent/prompts.py` — `QTYPE_STRATEGIES["QueryAttrQualifier"]` and `QTYPE_STRATEGIES["QueryRelationQualifier"]` elevate `GetQualifierValue` as preferred once qualifier name is known

## Related Docs

- [Decisions/qualifier-and-format-prompt-hardening.md](./qualifier-and-format-prompt-hardening.md) — post-merge audit that identified qualifier extraction as the largest remaining gap; `GetQualifierValue` was recommended next step #3
- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — prior tool surface expansion (CountEntities / SelectExtreme / VerifyFact)
- [System/agent_system.md](../System/agent_system.md) — QTYPE_STRATEGIES and QTYPE_TOOL_MAP context

## Context

After the qualifier-and-format-prompt-hardening bundle landed (commit `6c80356`) and the judge swap + code-level RULE 0 retry shipped (commits `182237c`, `d9ae8de`), n=100 accuracy stabilized at:

- `minimax-m2.7-kit`: **0.810** (+2 pp vs 0.79 pre-hardening baseline)
- `gemma-4-31b-kit`: **0.790** (-2 pp from 0.81 — within noise; 0 timeouts vs 36/100 previously)

Per the post-merge audit in the hardening ADR, 13/39 wrongs across the n=100 sample touched qualifier extraction. QueryAttrQualifier and QueryRelationQualifier had the deepest accuracy gap:

| Type | gemma (pre-fix) | minimax (pre-fix) |
|---|---|---|
| QueryAttrQualifier | 0.50–0.70 | 0.50–0.70 |
| QueryRelationQualifier | 0.56–0.78 | 0.56–0.78 |

**Root cause:** agents knew which qualifier they wanted but called `GetEdgeQualifiers` or `GetQualifiersByPredicate`, which return the *full* qualifier dict for a statement. The LLM then had to pick the right key out of 5–15 qualifier slots — and frequently mis-picked, especially when multiple plausible keys were present (e.g., `point_in_time` vs `start_time` vs `end_time`). The full-dict response also inflated context, crowding out later reasoning.

The three existing qualifier tools each solve a different sub-problem:

| Tool | What it does | Limitation |
|---|---|---|
| `GetEdgeQualifiers` | Returns full qualifier dict for an **attribute** statement | Full dict; agent must pick the right key |
| `GetQualifiersByPredicate` | Returns full qualifier dict for a **relation** statement | Full dict; agent must pick the right key |
| `QualifierFilter` | Filters a list of entities by a qualifier condition | Input is entity list, not a single answer extraction |

None of these tools project a *single* qualifier value directly — forcing extra LLM reasoning over a noisy multi-key response.

## Decision

Add `GetQualifierValue` as the 26th KQAPro MCP tool. The tool projects exactly one qualifier value from a statement, eliminating the pick-the-key step entirely.

### Signature

```python
GetQualifierValue(
    subject_id: str,       # KG entity Q-id of the subject
    predicate: str,        # Relation or attribute predicate label
    target: str,           # Q-id for relations; literal string for attributes
    qualifier_name: str,   # The specific qualifier key to project
    predicate_type: str = 'auto',  # 'relation' | 'attribute' | 'auto'
)
```

### Auto-detection and direction logic

- **`predicate_type='auto'`**: inspects `target` shape — a Q-id (starts with `Q`) implies a relation statement; a literal implies an attribute statement.
- **Direction**: tries `subject_id → target` first; if not found, auto-retries `target → subject_id` (backward direction). This subsumes the direction rule added in the hardening ADR's Change #3, so agents no longer need to reason about directionality manually.
- **Return**: the projected qualifier value only — not the full qualifier dict.
- **Journal write**: logs to `session_journal.verified_facts` with `type="qualifier_value"` (consistent with other fact-retrieval tools).
- **Entity resolution**: auto-resolves entity URIs to labels via `BatchGetNodeLabels` before returning.

### Prompt integration

`QTYPE_STRATEGIES["QueryAttrQualifier"]` and `["QueryRelationQualifier"]` now recommend:
1. Use `GetEdgeQualifiers` / `GetQualifiersByPredicate` for **discovery** (unknown qualifier name).
2. Once the qualifier name is known (from the question or a prior discovery call), use `GetQualifierValue` — not another full-dict call.

`QTYPE_TOOL_MAP` maps both types to `GetQualifierValue` so it appears in the agent's tool priority list.

## Why not extend the existing tools

- `GetEdgeQualifiers` and `GetQualifiersByPredicate` are used for qualifier *discovery* — callers that don't yet know which qualifier key to use. Changing their return shape to project a single key would break the discovery use case.
- A new tool keeps both use cases explicit in the agent's strategy, which is more parseable for the LLM than an overloaded parameter on an existing tool.

## Validation Plan

1. Re-run n=100 benchmark (seed=42, KIT endpoint, `minimax-m2.7-kit` then `gemma-4-31b-kit`, v4-pro judge, Fix 1 retry active). Output dirs: `benchmark_results/{minimax,gemma}-getqualval-2026-05-05-1100/`.
2. Required signal: QueryAttrQualifier and QueryRelationQualifier accuracy bands move upward relative to the 0.810 / 0.790 baseline. Count and Verify should be stable (tool is not in those strategies).
3. If qualifier bands do not improve, audit traces for whether the agent is *calling* `GetQualifierValue` or falling back to `GetEdgeQualifiers` — the prompt elevation may need stronger FORBIDDEN language.

## In-flight Status

Sequential benchmark launched on hetzner (PID 3195921): minimax first, then gemma. Results pending.

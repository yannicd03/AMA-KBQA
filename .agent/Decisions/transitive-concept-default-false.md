# ADR: `transitive_concept` Default = False (with Retry-on-Empty Guidance)

**Date:** 2026-05-02  
**Commits:**  
- `1383a1d` — originally flipped default to True (regression, reverted)  
- `5a5090f` "Revert transitive_concept default to False after n=100 regression" — current state  

**Code:**  
- `ama_kbqa/server/kqapro_server.py` — `CountEntities` (~line 3825) and `SelectExtreme` (~line 3944): `transitive_concept: bool = False`  
- `ama_kbqa/agents/kqapro_agent/prompts.py` — `TOOL_LOOP_GUIDANCE` retry-on-empty instruction

## Related Docs
- [Decisions/kqapro-tool-surface-expansion.md](./kqapro-tool-surface-expansion.md) — original ADR for CountEntities / SelectExtreme addition
- [SOP/adding_new_kqapro_tools.md](../SOP/adding_new_kqapro_tools.md) — includes "always use deterministic SPARQL" rule; the lesson here extends it

---

## Context

`transitive_concept=True` switches `_concept_clause` from `rdf:type` to `rdf:type*` (subclass-aware SPARQL path). The hypothesis when flipping the default was: subclass expansion would recover cases like "woodwind instrument" where the gold count includes subclasses.

The n=100 v1 benchmark falsified this. minimax Count accuracy dropped 0.73 → 0.45. Two concrete regressions:
- **Woodwind question:** returned 68 entities instead of 3 gold (subclass pollution from the `rdf:type*` path expanding far past direct instances).
- **Netherlands provinces:** returned 0 instead of 11 gold (the `rdf:type*` join changed the WHERE clause shape so OR-conditions dropped rows).

Root cause: `rdf:type*` interacts badly with flat concepts (`country`, `province`, `manifestation`) — it either over-counts via deep subclass traversal, or changes the join selectivity enough to zero-out results. Both failure modes are concept-shape-dependent and not visible in code review.

## Decision

**Default stays False** (flat concept matching via `rdf:type`). Transitive expansion is explicit opt-in only.

To keep genuine hierarchy cases (saxophone → woodwind → instrument) recoverable without a default-on regression:
- `TOOL_LOOP_GUIDANCE` instructs the agent to retry `CountEntities` / `SelectExtreme` with `transitive_concept=True` IF a 0-result comes back AND the concept could plausibly have subclasses.
- This makes the saxophone/woodwind case a one-call fallback, not a default, so flat counts are never polluted.

## Trade-off Rejected

Default-on was rejected: the -28 pp Count regression on minimax from flat-concept breakage dwarfed any gain from auto-subclass expansion. A flag that silently changes the SPARQL WHERE clause shape is unsafe as a default.

## Lesson for Future Work

Default-flag flips that interact with SPARQL join shape need a sanity check on at least one flat-concept question before shipping. "Looks safe in code review" doesn't transfer when the flag rewrites the triple pattern. Add it to the pre-merge checklist in `SOP/adding_new_kqapro_tools.md` if similar flags appear on future aggregating tools.

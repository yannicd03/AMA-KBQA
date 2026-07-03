# ADR: WikiKGQA Synthesis Context Default Reverted to Minimal

**Date:** 2026-07-02
**Commit:** `dadfdbe` "WikiKGQA: default synthesis back to minimal (full-context A/B was a wash)"
**Status:** Settled

## Related Docs
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference for `AgentSparqlGenerator` / `WikidataAgent`, including the synthesis config knobs table
- [Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md](./wikikgqa-tool-budget-and-resilience-2026-06-30.md) — the force-synthesis-at-`max_tool_calls` mechanism that this decision governs the context of
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md) — follow-on decision the next day, also touching the generator's commit-time behavior

---

## Background

Commit `90326fa` (2026-07-02, earlier the same day) had introduced an
optional **full-context synthesis** mode for `WikidataAgent`: instead of
giving the synthesis step only the current journal (validated queries +
findings), it could be given the entire exploration transcript (every tool
call and result) on the theory that more information would let the model
assemble a better final SPARQL query. It defaulted **on**
(`WIKIKGQA_FULL_SYNTHESIS` default `"1"`, i.e. full unless explicitly
disabled via `WIKIKGQA_FULL_SYNTHESIS=0` / `benchmark.py --minimal-synthesis`).

## Decision

Flip the default to **minimal** (journal-only) synthesis context. Full
transcript context becomes opt-in via `WIKIKGQA_FULL_SYNTHESIS=1` /
`benchmark.py --full-synthesis` (the CLI flag itself was inverted to match —
`--minimal-synthesis` no longer exists, replaced by `--full-synthesis`).

**Evidence:** A same-day A/B on seed-42, `rand-50`, model
`kit.qwen3.5-397b-A17b` (Qwen3.5-397B):

| Synthesis context | Macro F1 |
|---|---|
| Full transcript | 0.781 |
| Minimal (journal-only) | **0.807** |

The gap (0.026) is within run-to-run noise for this sample size, and minimal
context is strictly cheaper (fewer tokens fed to the synthesis call, no
transcript-truncation risk on long explorations). With no measurable accuracy
benefit, the cheaper option becomes the default.

**Rejected:** Keeping full-context as the default because it was "probably
not worse." The A/B specifically existed to test that assumption and found
no benefit to justify the extra cost.

## Implementation

- `ama_kbqa/agents/wikidata_agent/agent.py` — `WikidataAgent._synthesis_full_context()`:
  now returns `os.environ.get("WIKIKGQA_FULL_SYNTHESIS", "0") == "1"` (was
  `!= "0"`, i.e. the env var's absence now means minimal, not full).
- `ama_kbqa/wikikgqa/benchmark.py` — CLI flag renamed `--minimal-synthesis` →
  `--full-synthesis`; setting it now sets `WIKIKGQA_FULL_SYNTHESIS=1` (was:
  unsetting the flag's absence meant full by default, presence meant `=0`).

## Consequence

Any run invoked without an explicit synthesis-context flag from this commit
onward uses minimal (journal-only) context. Comparing F1 numbers across runs
predating `dadfdbe` (2026-07-02) requires checking which context mode was
active — `run_manifest.json` (added the next day, see
`Decisions/wikikgqa-commit-time-recovery-2026-07-03.md`) records the
resolved `WIKIKGQA_FULL_SYNTHESIS` state per run going forward; earlier runs
predate the manifest and must be checked via console logs or commit date.

## Open Questions

| Question | Notes |
|---|---|
| Does full-context ever help on harder/longer explorations not well-represented in the 50-question sample? | Not tested; full-context remains available via the opt-in flag for future targeted experiments |

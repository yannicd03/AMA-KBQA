# 0003: Every conversational answer ends with a reproducible SPARQL query

**Status:** Accepted, shipped `5f05e1d` on `demo-v2-sparql` → merged into
`demo-v2-int` (`1e4d470`) → `demo-booth` (`c333c1d`), 2026-09-15.

## Context

The demo's conversational mode (`synthesis_mode = "conversational"`) bypasses
the benchmark's synthesis step and shows the agent's own final loop message
directly to the visitor. That answer previously ended at prose, with no way
for a booth visitor to check it against the underlying knowledge graph
themselves — a gap for a demo whose whole point is "ask a real KG a real
question." The fix needed to hold for direct sub-agents (KQAPro, SciQA), the
Orchestrator's Router mode, and Federated mode (two specialists, one fused
answer, two different graphs), and it had to leave benchmark mode's
byte-for-byte behaviour untouched, since the paper's numbers depend on it.

## Decision

Every conversational final answer now has three sections: the answer, a
short "How I found this:" summary, and a "Reproduce with SPARQL:" section
holding exactly one fenced query that retrieves the answer from the graph the
specialist actually used.

**Mechanism:**
- `_CONVERSATIONAL_ANSWER_DIRECTIVE` (`ama_kbqa/framework/base_agent.py:75`)
  states the contract in the shared system prompt, appended by
  `_get_effective_system_prompt` only when `get_synthesis_mode() ==
  "conversational"`.
- `_get_sparql_reproduction_hint()` (empty hook on `BaseKBQAAgent`) is
  overridden per KG to supply the one thing only the subclass knows — the
  real URI scheme, the named-graph/`FROM` clause, and which raw SPARQL tool
  verifies it: KQAPro's `ex:`/`prop:`/`attr:`/`qual:` scheme against
  `FROM <http://kqapro.org/kb>` via `RunSPARQL`; SciQA's
  `orkgr:`/`orkgp:`/`orkgc:` scheme inside `GRAPH <http://sciqa.org/kg>` via
  `RunORKGSPARQL`, with a no-raw-SPARQL variant when
  `AMA_SCIQA_DISABLE_RAW_SPARQL` has gated that tool off.
- **Verification is self-administered and bounded to one attempt.** While
  tools are still callable, the agent runs its own query once and prints
  "Verified against the knowledge graph." if the result contains the answer;
  otherwise (or once the final-answer step has forbidden further tool calls)
  it prints "(not executed)" and does not retry. No code re-runs or checks
  the query server-side — the contract is enforced by prompt instruction
  plus the loop-message inspection described below, not by a validator.
- **Fast path.** KQAPro's 1-hop fast path has no final LLM turn to write the
  contract, so `_finish_fast_path` (`base_agent.py:1384`) assembles it
  deterministically: `_build_fast_path_reproduction_query`
  (`kqapro_agent/agent.py:244`) rebuilds the lookup as a pasteable query and
  verifies it with one extra `RunSPARQL` call; an unbuildable or
  empty-result query returns `None`, falling back to the full agent loop
  rather than emitting a contract-violating answer.
- **Fusion.** The orchestrator's `_fusion_system_prompt`
  (`orchestrator_agent/agent.py:833`) requires every specialist's SPARQL
  block reproduced verbatim in one combined section, each labelled by the
  new module-level `AGENT_CONFIG[...]["graph_label"]` ("KQAPro (Wikidata
  subset)" / "ORKG") — merging, rewriting, or dropping a block is explicitly
  forbidden, since the two specialists query different endpoints with
  different URI schemes.
- **Benchmark mode is untouched.** The directive/hint are gated on
  `synthesis_mode`; pinned by `tests/framework/test_sparql_block_preservation.py`
  and `tests/test_sparql_reproduction_prompt.py`.

## Two loop bugs found and fixed (conversational mode only)

Both were latent before this change — the longer answer just made them
visible for the first time.

1. **`tool_choice` forced to `"required"` past the answer prompt.**
   `base_agent.py`'s tool loop forces `tool_choice="required"` for
   iterations ≤3 to make the model call a tool instead of "thinking". That
   directly contradicted the "give your final answer, do NOT call tools"
   prompt injected after `GetJournalSummary`: the model wrote the real
   answer as message content *and* attached a throwaway tool call to satisfy
   the API, and the loop's next turn — filler content such as "Task
   completed." — silently became `final_agent_content`, overwriting the real
   answer. **Fix:** once the answer prompt has been injected,
   `tool_choice` drops to `"auto"` (conversational mode only); a remembered
   contract-satisfying message (`_satisfies_conversational_contract`, checks
   for both "How I found this:" and "Reproduce with SPARQL:") is preferred
   over a final message that doesn't satisfy the contract.
   **Benchmark mode keeps the old schedule on purpose** — it's the loop
   behaviour the paper's numbers were measured with, and benchmark answers
   are terse and re-derived by synthesis anyway, so touching it there would
   be an unmeasured change to a published result.
2. **Verify-answer normalisation collapsed the whole answer to yes/no.**
   `_normalize_verify_answer` is the benchmark's exact-match contract for
   `Verify`-type questions. Applied unconditionally, it would throw away the
   "How I found this:"/"Reproduce with SPARQL:" sections a conversational
   `Verify` answer still owes the user. **Fix:** `_finalize_answer_text`
   (`base_agent.py:632`) now applies the yes/no collapse only when
   `synthesis_mode != "conversational"`.

## Rejected / not attempted

- **Server-side re-validation of the printed query** (re-run it after the
  agent finishes, outside the tool-call budget) — would add a guaranteed
  extra round trip to every conversational answer and doesn't fit the
  "verify once while tools are still allowed" framing the directive uses;
  not attempted.
- **A single shared hint string instead of a per-KG hook** — rejected
  because the URI scheme, named-graph clause, and verification tool are
  genuinely different per graph; a hook keeps that KG-specific knowledge in
  the subclass that already owns it, matching every other
  `_get_*`-override pattern in `BaseKBQAAgent`.

## Known limits (not fixed here)

- **ASK queries cannot be verified** through either raw SPARQL tool
  (`RunSPARQL`/`RunORKGSPARQL`). Prompts instruct the agent to verify the
  pattern as `SELECT ... LIMIT 1` instead of running the printed `ASK`
  query. Untested end to end.
- **Query quality is model-dependent** — SciQA in particular tends to emit
  a `VALUES`-list of the resources it already found rather than a general
  query a visitor could adapt.
- **Fusion has been observed dropping a `FROM` clause** on at least one run
  despite the "reproduce verbatim" rule in `_fusion_system_prompt`. Informal
  acceptance observation, not test-covered — worth a fusion-specific
  regression test if it recurs.

## Acceptance (2026-09-15, KIT Mistral Small 4)

KQAPro director query executes and returns Christopher Nolan; Router pill
returns Ulm; SciQA COVID-19-papers query returns the four papers against the
live ORKG endpoint; a Federated run preserved both specialists' blocks.
Latency deltas were within noise except SciQA (~+12s median).

## Related Docs

- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) "Answer Contract (Conversational Mode)" — user-facing mechanics, per-KG hint reference, fast path, fusion labels
- [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md) — the fusion mechanism this ADR's fusion rule builds on
- [Decisions/multiturn-direct-agent-conversation.md](multiturn-direct-agent-conversation.md) — conversational-mode loop semantics this ADR's two fixes touch
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) §9-10 — smoke-test procedure that exercises this contract on staging

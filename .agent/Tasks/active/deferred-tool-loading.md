# Deferred / Two-Tier Tool Loading

**Status:** 📋 Planned
**Priority:** Medium (cost + latency, not correctness)
**Identified:** 2026-07-05 (architecture & implementation audit, finding C3)

## Related Docs
- [../../Decisions/architecture-audit-2026-07-05.md](../../Decisions/architecture-audit-2026-07-05.md) — full audit; C3 is the source finding, split into "cheap filtering" (already scheduled as an immediate FIX) and this deferred task
- [../../System/agent_system.md](../../System/agent_system.md) — agent loop, `_llm_call`, tool catalog injection
- [../../System/project_architecture.md](../../System/project_architecture.md) — MCP server / tool catalog overview

## Problem

`framework/mcp_client.py:convert_tools_to_openai_format` resends the **full** tool schema catalog on every `_llm_call`. Per a prior feasibility study (2026-04-05), tool schemas account for ~39.7% of per-call token overhead. Most of any given agent run only ever exercises a small "core" subset of tools; the long tail of domain-specific tools (KQAPro has 26+, SciQA has 20+) is rarely touched in a given question but its full JSON schema is paid for on every single turn regardless.

A companion, cheaper fix — auditing `_get_allowed_tools_for_qtype` to shrink each question type's allowed tool set to what it actually needs — is being done immediately as part of the 2026-07-05 audit (FIX, not this task). That captures some of the token win with zero new machinery. This task is the larger, structural follow-up: don't just narrow the set per qtype, make the set itself elastic within a single agent run.

## Goal

Send full schemas only for a small always-on **core** tool set. Long-tail tool schemas are fetched on demand, mid-run, only when the model actually needs them. Target ~33% token reduction (per the 2026-04-05 feasibility study), lowering both cost and latency.

## Design

1. **Partition tools into CORE vs. DEFERRED.**
   CORE = a small, always-loaded set covering the common path: `FindNode`, `GetJournalSummary`, `GetNodeSummary`, `GetAttributeDetails`, `GetRelationDetails`, `RunSPARQL`, and similarly high-frequency tools per KG. DEFERRED = the long tail of domain-specific / low-frequency tools.

2. **Add a meta-tool for on-demand loading.**
   Introduce `load_tools(names[])` (or `search_tools(query)`) that returns full schemas for the requested tool names and marks them "active" for the rest of the conversation turn onward.

3. **Track an active-tool set per agent run.**
   `base_agent.py` maintains a set of currently-active tool names. `_llm_call` sends `CORE ∪ active` schemas only — never the full catalog — on every turn.

4. **Escalation path when the model reaches for an unloaded tool.**
   If the model emits a tool call naming a tool not yet in the active set, auto-load it (add to active set) and retry the call rather than erroring. This keeps the model's mental model of "the tools exist" intact even though it hasn't seen the schema yet — it should behave like a natural miss-then-recover, not a hard failure.

## Risks

- **First-use latency cost.** Loading a deferred tool the first time a run needs it adds a round-trip. Expected net win overall because most iterations don't touch the tail, but any single question that needs 3+ deferred tools could see more round-trips than today, not fewer. Worth measuring in the A/B, not just assuming.
- **Tool visibility changes agent behavior.** This is not latency-neutral for correctness — hiding a tool's schema can change whether the model even considers using it (same caveat as the cheap-filtering companion fix). A full benchmark A/B (not just a token-count comparison) is required before landing.
- **Interacts with text-mode tool-call catalog injection.** Models using the client-side `<tool_call>` text parser (`framework/text_tool_calls.py`, see [text-mode-tool-calls.md](../../Decisions/text-mode-tool-calls.md)) get the tool catalog injected into the prompt text rather than via the native `tools=` API param. That injection path must also respect CORE/deferred partitioning — an oversight here would silently defeat the token savings for those models, or leave them unable to discover deferred tools at all.

## Effort estimate

3-5 days, per the prior feasibility study. Low risk if the CORE set is kept generous (better to slightly under-defer at first than to force excessive `load_tools` round-trips).

## Prerequisites / synergies

- Pairs with the qtype-level tool filtering already landing as part of the 2026-07-05 audit's cheap-filtering FIX (C3) — that work narrows the *static* per-qtype allowed set; this task adds a *dynamic* on-demand layer on top of whatever that set becomes.
- Synergizes with A2 (server-base extraction, see [architecture-audit-2026-07-05.md](../../Decisions/architecture-audit-2026-07-05.md#a2--duplicated-server-infrastructure-across-kqaprosciqa-servers)) — if/when server infra is unified, tool metadata (which tools are "core" per KG) should live in one place rather than being duplicated per server.

## Out of scope for this task

- Any change to tool *return shapes* or answer-path text — that's explicitly the abstract-operations-contract stance (see A2) and would move benchmark numbers.
- Cross-KG tool sharing/dedup — this task is about visibility/loading, not about consolidating the KQAPro and SciQA tool catalogs themselves.

## Validation plan

- Token-count before/after on a fixed question sample (confirm ~33% reduction materializes as measured, not just theoretical).
- Full benchmark A/B (not just latency/cost) on both KQAPro and SciQA, since tool visibility can shift which tools the model reaches for.
- Confirm text-mode tool-call models (minimax-m2.7 style) retain full tool discoverability under the new catalog injection.

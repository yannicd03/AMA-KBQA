# Rewrite the Agent Loop on Plain LangGraph `StateGraph`

**Date:** 2026-08-22
**Branch:** `feat/langgraph-rewrite` (separate worktree). Pre-rewrite reference commit tagged `pre-langgraph-rewrite`.
**Status:** Approved, in implementation.

## Related Docs
- [Tasks/active/langgraph-rewrite.md](../Tasks/active/langgraph-rewrite.md) — full PRD: phased plan, gates, in-flight-PRD disposition, follow-up TODOs
- [Tasks/active/langgraph-rewrite-phase0/request_diff.md](../Tasks/active/langgraph-rewrite-phase0/request_diff.md), [.../cache_gate.md](../Tasks/active/langgraph-rewrite-phase0/cache_gate.md) — Phase 0 go/no-go evidence
- [Decisions/transient-retry-and-chatkit-extraction.md](transient-retry-and-chatkit-extraction.md) — superseded in part by this ADR (see note added at its top)
- [System/agent_framework.md](../System/agent_framework.md) — current `BaseKBQAAgent` mechanics this rewrite replaces
- [Tasks/active/benchmark_persistence_refactor.md](../Tasks/active/benchmark_persistence_refactor.md), [Tasks/active/deferred-tool-loading.md](../Tasks/active/deferred-tool-loading.md) — absorbed into this rewrite's scope (see Decision)

## Context

`ama_kbqa/framework/base_agent.py` is a 2,702-LOC god class implementing an already-graph-shaped
control flow (classify → fast-path → tool loop → synthesis → finalize) as nested `if`/`while`
inside one class. Several independent PRDs were separately queued against pieces of this same
class: `benchmark_persistence_refactor.md` (checkpointing), `deferred-tool-loading.md` (per-request
tool overrides). The wiki previously evaluated LangGraph on 2026-04-05
(`0_Claude/AMA-KBQA/Feasibility - LangGraph Migration Token and Latency Impact.md`) and **rejected**
it — but strictly on token/latency grounds, for a migration undertaken *for those reasons alone*.
That note explicitly flagged its own limit: reconsider "if we're refactoring the agent architecture
for other reasons and want to adopt a standard framework." This PRD is that reconsideration; it
claims no token or latency win, only structural ones.

The user's initial ask covered `langchain.agents.middleware` (`create_agent`) and `deepagents`. A
2026-08-22 scope amendment restricted this to **plain LangGraph only**: an explicit `StateGraph`
(model node, tools node, guard nodes, conditional edges), no `deepagents`, no `create_agent`, no
prebuilt `middleware` stack. The middleware-hook mapping worked out in the PRD (its §2) is kept as
a *behaviour inventory* — each row becomes a graph node or a pure helper called from one, not an
`AgentMiddleware`.

## Decision

**Rewrite the agent loop on an explicit LangGraph `StateGraph`**, conditional on the Phase 0
prefix-cache gate passing (it did — see below). Concretely:

- One `KGAgentGraph` per adapter (state = messages + journal + question meta + budgets):
  `START → classify → fast_path | tool_loop → synthesis → finalize → END`, with a follow-up edge
  bypassing classify/fast-path. `OrchestratorGraph`: `START → probe → select_agent → conditional
  edge → {kqapro|sciqa} subgraph → fallback_llm → END`. Routing is a graph edge, not a tool
  returning `Command`, so the `Command.PARENT` subgraph-routing trap (`ORCA/langgraph-subgraph-routing.md`)
  does not apply by construction.
- Every current loop behaviour (qtype tool filter + schema compression, `tool_choice` schedule,
  text-mode tool calls, context compaction with hysteresis, 5-layer loop detection, zero-tool/
  truncated-call retry, iteration/tool-call caps, concurrent tool execution + truncation + spans,
  `TransientRetry`, token accounting) is ported as an explicit node or a pure helper called from
  one — see PRD §2 for the full behaviour-to-node table.
- **No `deepagents`.** Its `FilesystemMiddleware`/`SubAgentMiddleware` cannot be disabled (only
  hidden), it hard-pulls `langchain-anthropic`/`langchain-google-genai` we don't use, and its
  always-on `SummarizationMiddleware` conflicts with the deliberately different journal-based
  compaction. Only its *pattern* (one state key + one tool + one prompt hook per concern) is
  borrowed, for a hand-built `JournalMiddleware`-shaped node.
- **Journal stays server-side inside the MCP tool servers** for the first cut (option A). Moving
  `JournalState` into graph state with a reducer (option B) is explicitly **postponed until after
  the rewrite reaches parity** (PRD §8.2, §9) — not dropped, tracked as a follow-up TODO.
- **LangSmith is opt-in only** (`LANGSMITH_TRACING=true` + API key); `TraceRecorder`
  (`framework/trace.py`) remains the primary, renderer-agnostic trace source and nothing in the
  code path requires LangSmith.
- Dependencies added: `langgraph>=1.0`, `langchain-core`, `langchain-openai` (the spike-only `langchain-mcp-adapters` is removed again in Phase 1; tool execution reuses `MCPClient`),
  `chatkit[langchain]>=1.1.0` (bump from the pinned `v1.0.0` core-only extra). Removed:
  `openai-agents`, `fastapi`, `uvicorn` (declared, never imported).
- Absorbs two previously-separate PRDs into this rewrite's design rather than implementing them
  against the old class: `deferred-tool-loading.md` (per-request tool override is now a first-class
  graph-node concern) and `benchmark_persistence_refactor.md` (a `SqliteSaver` checkpointer gives
  `--resume` for free). Both are archived as superseded-by-absorption, not shipped independently.
- Implementation delegated to Sonnet subagents in phases (Phase 0 spike → foundation → behaviour
  port → graph topology → harness/frontend → parity benchmark → cleanup), orchestrator
  reviews/verifies/commits, on a dedicated branch/worktree (per PRD §5, §8.4).

**Supersedes, in part:** `Decisions/transient-retry-and-chatkit-extraction.md`'s "AMA-KBQA has no
LangChain dependency and should not gain one via a loose chatkit version bump" invariant and its
import-surface regression test (`tests/llm/test_kit_library.py::test_langchain_is_not_installed`,
`test_langchain_only_names_raise_helpful_error`). That ADR's retry-wiring and package-extraction
history is otherwise unaffected and its body is not rewritten — see the dated note added at its
top pointing here. Both tests are deleted in Phase 1 as designed failures.

## Phase 0 evidence (2026-08-22 result: GO)

A minimal explicit `StateGraph` spike (tools loaded via `langchain-mcp-adapters` for the session only) over the KQAPro server (KIT model,
`ToolGating`-only, so tool schemas stay byte-identical to `convert_tools_to_openai_format`) was
diffed against the current loop's raw request bodies (`scripts/langgraph_spike.py`,
`scripts/prefix_cache_gate.py`):

- **Request shape**: `tools` (15, incl. 300-char compressed descriptions) and `messages` (4
  messages) are byte-identical between old and new paths. Only differences: LangChain
  unconditionally renames `max_tokens` → `max_completion_tokens` (KIT honours both identically,
  verified via `finish_reason=length` at 5 tokens) and adds an explicit `stream: false`.
- **Cache gate** (KIT `kit.gemma4-31b-it`, 18 requests, 3 trials × {cold, warm, grown}): prompt
  tokens 5213-5214 (raw) vs 5215 (LangGraph) — noise-level. Median TTFT raw 18.1/17.1/19.3s vs
  LangGraph 13.5/11.0/7.4s (cold/warm/grown) — no path-specific regression; endpoint noise
  dominated the run.
- Resolved dependency versions: `langgraph` 1.2.11, `langchain-core` 1.6.0, `langchain-openai`
  1.6.0, `langchain-mcp-adapters` 0.3.2, `chatkit` 1.1.0. Side effect: `openai` 2.8 → 3.3.1, which
  vendors `httpx2` — any future transport mocking must target `httpx2`, not `httpx`.
- Phase 1 requirements learned from the spike: never let `langchain-mcp-adapters` auto-schemas
  reach the model directly (route through `convert_tools_to_openai_format`); `tool_choice="required"`
  passes through verbatim; read "messages as sent" from the outgoing request, not from a live
  mutable list.

Full detail in `Tasks/active/langgraph-rewrite-phase0/request_diff.md` and `cache_gate.md`.

## Consequences

- `BaseKBQAAgent`'s 2,702-LOC god class is decomposed into ~10 independently testable graph
  nodes/helpers, each parity-tested against recorded traces (`tests/**/tool_traces/question_NNN.json`)
  before the graph topology phase begins.
- Native LangGraph streaming deletes the thread+queue bridge in `frontend/utils/lifecycle_runner.py`.
- A checkpointer (`MemorySaver` default, `SqliteSaver` for benchmark runs) gives incremental,
  resumable benchmark runs, closing `benchmark_persistence_refactor.md` without separate work.
- Per-request tool override is now a first-class extension point, closing `deferred-tool-loading.md`
  the same way.
- AMA-KBQA gains a LangChain dependency for the first time, reversing the explicit "langchain-free"
  invariant from `transient-retry-and-chatkit-extraction.md`. Any future dependency work must treat
  that invariant as retired, not silently re-enforce it.
- Reproducibility risk: the SEMANTiCS paper's numbers must stay pinned to the `pre-langgraph-rewrite`
  tag on `main`; nothing in this rewrite is claimed to reproduce or improve those numbers on its own.
- Ongoing risk carried into Phases 1-5: prefix-cache regression from any change in serialized
  message/tool shape (a 45-token tools-list change was measured at ~3x TTFT,
  `Tasks/active/prompt-cache-utilization.md` §4) and token-accounting gaps from LangChain callbacks
  not propagating into `create_agent` subgraphs (not applicable here since `create_agent` itself is
  out of scope, but the underlying callback-propagation gap is why token accounting is read from
  `AIMessage.usage_metadata` in an explicit node rather than via callbacks).
- Acceptance gate (Phase 5): full KQAPro + SciQA parity runs at one commit/model/config; accept if
  accuracy is within judge run-to-run variance and prompt tokens/TTFT are within 10% of the
  reference run resolved per `SOP/refreshing_paper_numbers.md`. Otherwise bisect by node.

## Alternatives considered

- **`langchain.agents.middleware` / `create_agent`** (the pre-amendment plan). Rejected by explicit
  user scope amendment on 2026-08-22, not on technical grounds — the middleware-hook mapping in the
  PRD remains a valid behaviour inventory, just realized as graph nodes instead of `AgentMiddleware`
  instances.
- **`deepagents`**: rejected — see Decision above. Verified against `deepagents` 0.7.x
  (wiki `deepagents-langgraph.md`, 2026-08-13/18/22 entries).
- **Do nothing (keep the god class)**: rejected — three independent PRDs
  (`benchmark_persistence_refactor.md`, `deferred-tool-loading.md`, and the general audit finding
  that the class had become hard to extend) were converging on the same file with overlapping
  surgery; a structural rewrite subsumes all three more cheaply than three separate patches.
- **`TodoListMiddleware`-as-scratchpad-replacement**: rejected. The journal is structured
  KG-exploration state auto-populated by tools and consumed by synthesis; `write_todos` is a
  free-text plan list. At most it could replace the `current_plan`/`completed_steps` fields — left
  as a measure-don't-assume follow-up TODO (PRD §9), not adopted here.

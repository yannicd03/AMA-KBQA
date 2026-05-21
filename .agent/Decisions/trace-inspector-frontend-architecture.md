# ADR: Trace Inspector + Graph View — Frontend and Instrumentation Architecture

**Status:** Accepted (merged 2026-05-09, commit `880e865`)

## Related Docs
- [Project Architecture](../System/project_architecture.md) — overall system overview
- [Agent System](../System/agent_system.md) — agent lifecycle and journal data model

---

## Context

The Streamlit frontend previously surfaced agent reasoning via a single captured-stdout ANSI block. This was sufficient for casual debugging but poor for non-trivial questions: no span hierarchy, no per-span latency or token attribution, no structured view of discovered knowledge graph data.

Two capabilities were added, modelled after Langfuse:

1. **Trace Inspector** — hierarchical two-pane view of an agent run as timed spans (LLM calls, tool calls, classification, synthesis, delegation, interventions) with per-span detail tabs.
2. **Graph View** — vis-network visualisation of the subgraph built into the journal (`visited_nodes` + `verified_facts`) with a timeline scrubber across journal snapshots.

---

## Decision 1: Stay on Streamlit — reject React + FastAPI migration

**Chosen:** Add two pages to the existing Streamlit app.

**Rejected:** Migrate frontend to React + FastAPI.

**Rationale:**
- Scope is two pages plus instrumentation. A React + FastAPI migration is estimated 10–20× the work.
- The tool is internal and SSH-tunnelled; there are no multi-user auth, real-time collaboration, or URL-routable deep-linking requirements that would justify migration now.
- The `TraceRecorder` is designed with an OTel-compatible shape — it is renderer-agnostic. A future React frontend would consume `recorder.to_dicts()` / `recorder.to_jsonl()` verbatim, so the instrumentation layer is not throwaway work.

**Threshold to revisit:** multi-user auth requirement, real-time collaboration need, or URL-routable deep-linking. Until then, Streamlit is sufficient.

---

## Decision 2: ContextVar-based span nesting, not manual parent_id threading

**Chosen:** `TraceRecorder` uses a `ContextVar[Optional[str]]` to propagate the current span id across `await` boundaries.

**Rejected:** Threading `parent_span_id` through every function signature.

**Rationale:**
- The actual call graph is: `ask` → `_run_tool_loop` → `_execute_tool_calls` → `_execute_single_tool` → `mcp.call_tool`. Awaits occur at every level. Manual parent_id threading would require changing every signature and is brittle at depth 4.
- A `ContextVar` propagates correctly across `await` boundaries in a single-coroutine async chain, making nesting work without signature changes.
- The `ContextVar` is per-`TraceRecorder` instance, so two concurrent recorders (e.g. orchestrator + sub-agent) remain isolated until the orchestrator deliberately shares its recorder.

---

## Decision 3: Journal snapshots on mutating tools only — not per-iteration

**Chosen:** `GetJournalStateJSON` is called only (a) after tool calls listed in `JOURNAL_MUTATING_TOOLS` frozenset, and (b) after journal refreshes that show progress.

**Rejected:** Fetching a snapshot on every tool-loop iteration.

**Rationale:**
- A typical question runs ~15–25 tool-loop iterations. Per-iteration snapshots would add ~50 stdio MCP RPCs per question (~0.5–1.5 s overhead, worse on Hetzner).
- The `JOURNAL_MUTATING_TOOLS` set cross-checks which tools actually call `session_journal.*` write operations in `kqapro_server.py` and `sciqa_server.py`. Non-mutating tools (e.g. `FindNode`, `GetNodeSummary` when they are pure reads) do not trigger snapshots.
- `GetJournalStateJSON` is not exposed to the LLM — it is filtered from the tool list returned to the agent. Only the agent's snapshot path calls it.
- This bounds the extra MCP calls to approximately the count of journal-mutating tool calls, not iterations.

---

## Decision 4: Orchestrator sub-agent nesting via shared recorder, not event-copying

**Chosen:** Before calling a sub-agent, the Orchestrator sets `agent.recorder = self.recorder` and `agent._parent_span_id_override = current_span_id`. The sub-agent appends its spans directly to the shared recorder.

**Rejected:** Copying sub-agent events into the orchestrator recorder after the sub-agent returns (would require ID rewriting).

**Rationale:**
- Span IDs are generated at event creation time. Copying after the fact would require rewriting parent_span_id chains, which is fragile and loses temporal ordering.
- Shared recorder preserves exact timing interleaving and the full nesting hierarchy in one flat ordered list.
- Sub-agents accept `parent_recorder` + `parent_span_id` kwargs. The `delegate` span that wraps the sub-agent call is opened by the orchestrator before assignment, so the sub-agent's root span becomes a child of the delegate span.
- After the sub-agent returns, the orchestrator hoists `agent.journal_snapshots` and token totals into its own state so the Chat page sees a uniform interface regardless of whether the orchestrator or a direct agent was used.

---

## Decision 5: vis-network via CDN, not a Python graph-viz dependency

**Chosen:** `st.components.v1.html` embeds a complete HTML document with vis-network loaded from CDN. No new Python dependencies.

**Rejected:** Python-side graph rendering (e.g. pyvis, networkx + matplotlib, plotly graph objects).

**Rationale:**
- Zero Python deps added. The project already has `streamlit`; vis-network is JS-only.
- vis-network provides interactive pan/zoom, node dragging, and physics layout that Python-side libraries cannot match without a JS bridge anyway.
- Iframe re-render mitigations: `@st.fragment` scoping so unrelated reruns don't remount the iframe; viewport pan/zoom persisted in iframe `localStorage` keyed per `trace_id` so the view survives Streamlit reruns within the same trace.

---

## Decision 6: Tool-loop iteration is a point-in-time event, not an interval span

**Chosen:** `recorder.event("tool_loop_iter", ...)` — instantaneous marker at the start of each iteration.

**Rejected:** Wrapping the loop body in `async with recorder.span("tool_loop_iter", ...)`.

**Rationale:**
- The loop body contains many `continue`/`break` paths (zero-tool-call retry, truncated-tool-call retry, max-iterations abort, loop detection intervention). Wrapping it in an async context manager would require restructuring all these paths.
- The LLM-call and tool-call spans inside each iteration already provide the structural timing. The iteration event serves only as an iteration-boundary marker.

---

## Known Gap: Live Updates Deferred to v2

v1 ships post-hoc only: `agent.ask()` runs to completion, then `recorder.to_dicts()` and `agent.journal_snapshots` are read. There is no live streaming of spans as the agent runs.

**v2 direction:** Mirror the existing `2_Batch_Processing.py` background-thread pattern. Use `@st.fragment(run_every=N)` to poll a shared queue into which the recorder pushes completed spans. This is the closest Streamlit analog to a live feed. Tracked as a follow-up; no ticket yet.

---

## Status Update — v2 Shipped (commit `81eefd9`, 2026-05-21)

The deferred live-updates gap above is now resolved. The full design is recorded in **[live-trace-and-chat-unification.md](./live-trace-and-chat-unification.md)**.

Summary of what closed the gap:
- `TraceRecorder` gained `add_listener` / `remove_listener` observer hooks; listeners receive `("open"|"close"|"event", info_dict)` tuples as spans fire.
- The Chat page spawns a daemon thread with its own asyncio loop that runs `agent.ask()` and pushes recorder notifications onto a `queue.Queue`; the Streamlit main thread drains the queue inside an `@st.fragment(run_every=0.4)` block — exactly the pattern projected above.
- A live inline-SVG agent-lifecycle figure (ported from the paper's `fig:agent_flow`) lights up nodes as the run progresses.
- The Chat page was unified: Trace Inspector and Graph View now appear as tabs in-page (Lifecycle | Trace | Graph) rather than requiring navigation to pages 5/6.
- Pages 5 and 6 were refactored into thin wrappers delegating to `utils/trace_panel.py` and `utils/graph_panel.py` panel helpers shared with the Chat page.

---

## Implementation

**New files (commit `880e865`):**

| File | Purpose |
|------|---------|
| `ama_kbqa/framework/trace.py` | `TraceEvent` dataclass + `TraceRecorder` (async/sync context managers, ContextVar nesting, `to_dicts`/`to_jsonl`) |
| `ama_kbqa/frontend/pages/5_Trace_Inspector.py` | Trace selector + two-pane span tree + detail tabs + JSONL export |
| `ama_kbqa/frontend/pages/6_Graph_View.py` | Trace selector + snapshot scrubber + vis-network iframe + toggle/stats |
| `ama_kbqa/frontend/utils/trace_render.py` | Pure-Python helpers: `build_tree`, `summarise`, `render_tree_html`, `format_duration` |
| `ama_kbqa/frontend/utils/graph_html.py` | `journal_to_graph`, `build_graph_html` — complete HTML doc with vis-network template |

**Modified files (commit `880e865`):**

| File | Change |
|------|--------|
| `ama_kbqa/framework/base_agent.py` | Owns `self.recorder`, wraps all span boundaries, emits point-in-time events |
| `ama_kbqa/agents/orchestrator_agent/agent.py` | Owns recorder + journal_snapshots, `_delegate()` helper, hoists sub-agent data |
| `ama_kbqa/frontend/pages/1_Chat.py` | Reads `recorder.to_dicts()` + snapshots after `ask()`, adds Trace/Graph buttons |
| `ama_kbqa/server/kqapro_server.py` | Adds `GetJournalStateJSON` (LLM-hidden snapshot tool) |
| `ama_kbqa/server/sciqa_server.py` | Adds `GetJournalStateJSON` (LLM-hidden snapshot tool) |
| `ama_kbqa/frontend/utils/styling.py` | Adds kind-coloured pill CSS for span kinds |

**Tests:** 122 total (was 111). New: `tests/framework/test_trace.py` (14 tests), `tests/framework/test_trace_render.py` (11 tests).

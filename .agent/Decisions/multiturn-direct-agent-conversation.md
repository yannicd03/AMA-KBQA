# ADR: Multiturn Conversation for Directly-Selected Sub-Agents

**Date:** 2026-05-29  
**Commit:** `81fbdf9`  
**Status:** Shipped  
**Files:** `ama_kbqa/framework/base_agent.py`, `ama_kbqa/frontend/pages/1_Chat.py`, `ama_kbqa/frontend/utils/lifecycle_runner.py`

## Related Docs
- [System/project_architecture.md](../System/project_architecture.md) — overall system overview; BaseKBQAAgent provides list
- [System/agent_system.md](../System/agent_system.md) — agent lifecycle, pre-agent hook, loop detection
- [Decisions/live-trace-and-chat-unification.md](./live-trace-and-chat-unification.md) — lifecycle_runner and trace propagation design

---

## Problem

The system was single-turn and stateless. Each question created a fresh agent; `reset()` wiped `self._messages`; the chat UI displayed prior exchanges but never fed them back to the agent. Follow-up questions like "where was he born?" could not work because the agent had no memory of the prior exchange.

---

## Decision 1: Persist the full message stack on the sub-agent instance

**Chosen:** Keep one sub-agent instance alive across the conversation (`st.session_state["persistent_agent"]`) and accumulate its `_messages` across turns via `reset(keep_history=True)`.

**Alternatives considered:**

| Alternative | Rejected because |
|-------------|-----------------|
| Collapse history to (Q, A) pairs and inject them as a fresh system message | Loses tool results and intermediate reasoning; coreference in the model depends on tool call/result pairs being present in the exact position the model produced them. |
| Thread prior (Q, A) pairs through classify → route → tool loop as extra context | Would require touching the classifier, analysis-context builder, and router; significant blast radius for what is fundamentally a message-stack continuity problem. |

**Why the full stack works:** The main tool loop is already structured as a rolling `_messages` list with system + user + assistant + tool-call/result turns. The LLM resolves coreferences from that same context. Preserving the stack intact means the model sees prior tool results and answers in their native positions; no reconstruction is needed.

---

## Decision 2: Scope to directly-selected sub-agents only (Orchestrator stays stateless)

The Orchestrator dispatches to sub-agents dynamically and does not itself maintain KG-specific state. Routing a follow-up through the Orchestrator would require the router to know which prior sub-agent was active, adding router complexity for unclear gain. The Orchestrator path is left stateless and creates a fresh agent per turn.

In `1_Chat.py`, `multiturn_enabled = selected_agent != "Orchestrator"`.

---

## Decision 3: Skip the entire pre-agent hook on follow-up turns

**Follow-up detection:** `is_followup = any(m.get("role") == "user" for m in self._messages)` — evaluated before appending the new query. A fresh turn's stack has only system/catalog messages; a continuation has at least one prior user turn.

**On a follow-up turn, the entire pre-agent hook is bypassed:**
- No classification LLM call
- No analysis-context injection
- No fast path
- No tool filtering

**Why:** A bare elliptical question ("where was he born?") classified in isolation yields low-confidence output that would mislead the fast path (wrong entity extraction, wrong qtype) and tool filtering (may exclude the tool that resolves the reference). The preserved stack already carries the prior entity, relationships, and reasoning. The full tool loop with all tools lets the model self-resolve coreference from context with high reliability.

**Trade-offs accepted:**
- Follow-ups do not benefit from Verify yes/no normalization (postprocessing).
- Follow-ups pay the full tool-set token cost per iteration (no `_get_allowed_tools_for_qtype` filtering). This is acceptable: follow-ups are typically drill-downs and involve fewer total iterations.

---

## Implementation gotchas

### `reset(keep_history=False)` new parameter
`base_agent.reset()` now accepts `keep_history: bool = False`. When `True`, `self._messages` is preserved; all other per-turn state resets (token counts, `tool_call_counts`, `tool_call_durations`, loop-detection counters, `recorder`, `journal`). This gives each turn a fresh trace and fresh per-turn token totals while keeping the conversation stack.

### `_catalog_injected` flag
The text-mode tool catalog (injected in `_ask_impl` for minimax-m2.7 compatibility) must not be re-injected on a reused stack. A `_catalog_injected: bool` flag on the instance gates the injection. When `keep_history=False` the flag is reset to `False` alongside `_messages`.

### Fast-path answer must be recorded in `_messages`
The fast path (`QueryAttr`/`QueryRelation`/`QueryName` single-hop shortcut) builds its answer from tool results without a final LLM turn. In single-turn mode the stack is discarded so this was harmless. With a persistent stack, a follow-up replaying the accumulated context would see the tool results but not the answer, breaking coreference. Fix: after a successful fast-path, append `{"role": "assistant", "content": final}` before returning.

### MCP / asyncio event-loop safety
Each turn runs in a new `asyncio` event loop (spawned by `lifecycle_runner.start_run()`). The previous turn's MCP connection is bound to a now-closed loop. Attempting to `close()` it on the new loop can raise. Fix: orphan the previous connection (`agent.mcp = None`) before calling `reset(keep_history=True, keep_mcp_open=True)`; `agent.ask()` → `_init_mcp` rebuilds a fresh connection on the current loop.

### Recorder capture ordering in `lifecycle_runner`
`start_run()` previously captured `recorder = getattr(agent, "recorder", None)` before spawning the worker thread. On a continuation, `reset()` replaces the recorder, so the listener would bind to the *old* recorder. Fix: capture the recorder inside the worker, after `reset()` completes, and push the new `trace_id` to the main thread via a `("__trace_id__", {"trace_id": tid})` queue message. `drain_into()` handles this message by updating `state.trace_id` so the live panel tracks the correct trace.

---

## Regression guarantee

First-turn and single-turn behavior is unchanged:
- `is_followup` is `False` on the first turn (no prior user messages in the stack), so the full pre-agent hook runs normally.
- The Orchestrator path creates a fresh agent every turn; `persistent_agent` is popped from session state for Orchestrator.
- Restart button and agent-switch both pop `persistent_agent`, forcing a fresh start.

Tests: `tests/framework/test_base_agent_multiturn.py` (221 lines — reset semantics, first-turn vs follow-up hook skipping) and additions to `tests/frontend/test_lifecycle_runner.py` (82 lines — continuation through the real worker). Full suite: 206 tests passing.

# ADR: Live Trace Pipeline + Chat-Page Unification (v2)

**Status:** Accepted (merged 2026-05-21, commit `81eefd9`)

## Related Docs
- [trace-inspector-frontend-architecture.md](./trace-inspector-frontend-architecture.md) — v1 ADR; the "Known Gap: Live Updates Deferred to v2" section there records the pre-conditions this ADR resolves
- [Project Architecture](../System/project_architecture.md) — overall system overview, data flow diagram
- [Agent System](../System/agent_system.md) — TraceRecorder API and span placement

---

## Context

The v1 trace architecture (commit `880e865`) ran `agent.ask()` synchronously on the Streamlit main thread, then read `recorder.to_dicts()` and `agent.journal_snapshots` only after the call returned. Users saw no indication of agent progress during a question — no live lifecycle state, no streaming spans. The v1 ADR explicitly deferred this to a "v2" follow-up, projecting the `2_Batch_Processing.py` background-thread pattern as the reference.

This ADR records the decisions made in v2.

---

## Decision 1: Listener-based observer on TraceRecorder — not polling `to_dicts()` diff

**Chosen:** `TraceRecorder.add_listener(fn)` / `remove_listener(fn)`. Listeners are called from inside `_open_span`, `_close_span`, and `event` with a `(kind, info_dict)` tuple where `kind` is `"open"`, `"close"`, or `"event"`. Exceptions are swallowed and logged (never propagate into agent logic).

**Rejected:** Polling `recorder.to_dicts()` on a timer and diffing against last-seen length.

**Rationale:**
- Push is cheaper than pull: the observer fires exactly once per span boundary, with no repeated list serialisation.
- Diff-based polling is fragile when spans close out of order or are emitted in bursts.
- Swallowing listener exceptions isolates frontend failures from agent correctness — a broken UI callback must not abort a benchmark run.
- The listener interface is the minimal surface needed; `to_dicts()` remains the canonical export path for post-hoc rendering and JSONL export.

---

## Decision 2: ContextVar-aware single asyncio loop in the worker thread

**Chosen:** `lifecycle_runner.py::start_run(...)` spawns a daemon thread that creates its own `asyncio` event loop (`asyncio.new_event_loop()`), runs `agent.ask(query)` on it, and pushes recorder notifications onto a `queue.Queue`. The main Streamlit thread only reads from the queue.

**Rejected:** Running `agent.ask()` via `asyncio.run()` on the main thread (blocks UI), or using `asyncio.run_coroutine_threadsafe` into a shared loop (requires a persistent background loop singleton, complicates cleanup).

**Rationale:**
- The agent's `ContextVar`-based span nesting (Decision 2 in the v1 ADR) is per-coroutine-context. A dedicated loop per question keeps `ContextVar` propagation correct without cross-contamination between concurrent questions.
- Daemon thread teardown is automatic when the Streamlit process exits; no explicit cleanup is needed.
- This mirrors `2_Batch_Processing.py` — the pattern already proven stable for long-running agent work in this codebase.

---

## Decision 3: `_SilentPlaceholder` to suppress worker-thread Streamlit calls

**Chosen:** Objects returned from inside the worker thread that would ordinarily trigger `st.write` / `st.status` calls are replaced with a `_SilentPlaceholder` no-op shim. Agent code that tries to update Streamlit state from the worker thread silently discards the call.

**Rejected:** Propagating `st.empty()` / `st.status()` handles across thread boundaries.

**Rationale:**
- Streamlit's widget API is not thread-safe. Calling `st.write()` from a non-main thread raises a `StreamlitAPIException`.
- The placeholder pattern requires zero changes to agent code; the agent does not know it is running in a worker thread.
- Actual UI updates happen only on the main thread via the queue drain fragment.

---

## Decision 4: Chat-page panel unification — tabs in-page, not separate Streamlit pages

**Chosen:** The Lifecycle, Trace, and Graph views are rendered as tabs in a panel beneath the chat conversation on `1_Chat.py`. Pages 5 and 6 remain as standalone pages (for deep inspection of historical traces) but now delegate to shared panel helpers: `utils/trace_panel.py::render_trace_panel(...)` and `utils/graph_panel.py::render_graph_panel(...)`.

**Rejected:** Keeping Trace/Graph as page-5/page-6 navigation only (original v1 design); or moving everything into a single page and removing 5/6.

**Rationale:**
- Users want to see trace context alongside the conversation, not navigate away. A mid-page tab panel is the minimal friction path.
- Pages 5/6 retain value for post-hoc, multi-trace comparison (they have trace selectors and are bookmarkable). Deleting them would regress that workflow.
- Extracting panel helpers (DRY) means the identical rendering logic is not duplicated: Chat page and standalone pages call the same code.

---

## Decision 5: "Simplified view" toggle as escape hatch

**Chosen:** A sidebar "Simplified view" toggle hides the Lifecycle/Trace/Graph panel, expanders, footers, and the ❉ decorative glyph. Only chat bubbles and the input widget remain. A silent `@st.fragment(run_every=0.4)` still drains the queue so the answer renders when the run finishes.

**Rejected:** Making simplified mode the default; or removing the toggle and always showing the panel.

**Rationale:**
- The lifecycle panel adds ~200px of vertical space. Users focused on iterating over many questions find it distracting.
- Hiding the panel must not stall the answer: the queue drain must continue even when the SVG and tabs are hidden.
- A sidebar toggle is immediately reversible and requires no page navigation.

---

## Decision 6: Inline-SVG lifecycle figure — ported from paper TikZ

**Chosen:** `utils/lifecycle_svg.py` renders a self-contained SVG with coordinates ported from the `fig:agent_flow` TikZ figure in `paper/main.tex`. Each node carries `data-state="idle|visited|active"` attributes; CSS in `utils/styling.py` (new `.lifecycle-*` block) applies colour via attribute selectors. Span-kind → node-id mapping lives in `utils/lifecycle_mapping.py` with `TOOLS_A` / `TOOLS_B` frozensets matching the figure caption's KQAPro / SciQA tool set distinction.

**Rejected:** Using a third-party diagram library (Mermaid, Graphviz, vis-network) for the lifecycle figure; or re-using the existing vis-network graph (which shows the KG subgraph, not the agent flow).

**Rationale:**
- The paper's TikZ figure is the canonical representation of the agent lifecycle. Porting its coordinates produces a figure that is consistent with the published paper with no extra dependencies.
- CSS attribute selectors on `data-state` are simpler and faster than JavaScript DOM manipulation for a small fixed graph.
- The `TOOLS_A` / `TOOLS_B` distinction propagates the figure's tool-set split into code without requiring a string match on every span.

---

## User-Visible Behaviour

1. When a question is submitted on the Chat page, the lifecycle SVG immediately highlights the **Classification** node.
2. As the agent progresses, nodes light up in real time (pre-hook → tool loop → synthesis) at ~0.4 s polling granularity.
3. After the answer arrives, the Trace and Graph tabs are populated; the SVG settles into a fully-visited state.
4. In Simplified view, none of the above is shown — the answer still appears when ready.

---

## Implementation

**New files (commit `81eefd9`):**

| File | Purpose |
|------|---------|
| `ama_kbqa/frontend/utils/lifecycle_runner.py` | `LiveLifecycleState`, `start_run(...)`, `drain_into(...)` — background thread + queue |
| `ama_kbqa/frontend/utils/lifecycle_svg.py` | Self-contained SVG renderer; `data-state` node attributes |
| `ama_kbqa/frontend/utils/lifecycle_mapping.py` | `SPAN_KIND_TO_NODE`, `TOOLS_A`, `TOOLS_B` mapping frozensets |
| `ama_kbqa/frontend/utils/trace_panel.py` | `render_trace_panel(trace, key_prefix=...)` extracted panel helper |
| `ama_kbqa/frontend/utils/graph_panel.py` | `render_graph_panel(trace, key_prefix=...)` extracted panel helper |

**Modified files (commit `81eefd9`):**

| File | Change |
|------|--------|
| `ama_kbqa/framework/trace.py` | Added `add_listener` / `remove_listener` / `_notify_listeners`; listener dispatch inside `_open_span`, `_close_span`, `event` |
| `ama_kbqa/frontend/pages/1_Chat.py` | Background-thread run; `@st.fragment(run_every=0.4)` drain; Lifecycle/Trace/Graph tab panel; Simplified view toggle |
| `ama_kbqa/frontend/pages/5_Trace_Inspector.py` | Thin wrapper: trace selector + sidebar + `render_trace_panel`; `st.slider` single-snapshot guard |
| `ama_kbqa/frontend/pages/6_Graph_View.py` | Thin wrapper: trace selector + sidebar + `render_graph_panel`; `KeyError: '_depth'` span-click fix |
| `ama_kbqa/frontend/utils/styling.py` | New `.lifecycle-*` CSS block for `data-state` node colouring |

**Tests:** 122 → 176. New: `tests/framework/test_trace.py::TestRecorderListeners` (4 tests), `tests/frontend/test_lifecycle_svg.py` (12), `tests/frontend/test_lifecycle_mapping.py` (19), `tests/frontend/test_lifecycle_runner.py` (3, stub agent + `threading.Event` for determinism).

---

## Status Update — v3 Shipped (commit `a1ca605`, 2026-05-26)

Three changes landed together and were deployed to the Hetzner frontend container (healthy).

### 1. Agent-lifecycle SVG redesigned to match `fig:agent_flow`

`utils/lifecycle_svg.py` was restructured to align precisely with the paper's `fig:agent_flow` layout. The previous flat-ish node list was replaced with three explicit phase bands:

| Band | Nodes |
|------|-------|
| Pre-Agent Hook | Question Type Classification → Entity Extraction → Strategy Injection |
| Main-Agent Loop | LLM Reasoning → Tool Call → Scratchpad → Done? (diamond); "no" loops back to LLM Reasoning, "yes" exits to Post hook |
| Post-Agent Hook | Answer Synthesis → Trace Evaluation → Lessons Learned |

A dashed feedback edge runs from Lessons Learned back to Strategy Injection. The band formerly titled "Post-Agent Synthesis" is now "Post-Agent Hook".

Old nodes that were collapsed: Classifier/Extractor text-label outputs, Loop Detect, Journal State, Tools A / Tools B, More?, Answer, Response.

New node IDs (canonical, used in mapping and tests):
`agent_invocation`, `pre_classifier`, `pre_extractor`, `pre_strategy_inject`, `main_llm_reason`, `main_tool_call`, `main_scratchpad`, `main_done`, `post_synthesis`, `post_evaluate`, `post_lessons`.

`utils/lifecycle_mapping.py` was updated to the new IDs:
- TOOLS_A (KQAPro traversal) and TOOLS_B (summary/verify) both light `main_tool_call`.
- `GetJournalStateJSON` and journal refresh light `main_scratchpad`.
- `loop_detected` / `intervention` / `context_trim` light `main_done`.
- `agent_run open` lights `agent_invocation`.
- `synthesis` lights `post_synthesis`.

### 2. Trace Inspector span tree is now click-to-select (native buttons)

The `st.radio` span selector was removed; the dark span tree itself is the selector.

Implementation: `utils/trace_panel.py::render_trace_panel` renders the depth-ordered spans as full-width `st.button`s inside a keyed `st.container(key="tracetree-<key_prefix>")`. `styling.py` scopes CSS to `[class*="st-key-tracetree-"]` to make the container a dark scroll box and strip the buttons down to dark tree rows. `utils/trace_render.py::span_button_label(event, depth)` builds each label as `:colour-background[kind] name  duration` with em-space indentation per tree depth (the kind→badge-colour map mirrors the kind-pill palette). Selecting sets `st.session_state[selected_key]` and calls `st.rerun()`.

> **History:** an earlier iteration used query-param anchor links (`render_tree_html(..., link_param=...)` → `<a href="?sp_…">`). That worked but every click was a full-page navigation, which reset the embedded Chat panel's active tab and felt broken. The button approach reruns over the websocket — **no page reload, no tab reset, URL unchanged** — verified with Playwright (click updates the detail pane, a `window` marker survives, URL stays clean). `render_tree_html` reverted to its original read-only signature (still used for the live lifecycle view).

### 3. Graph View removed (broken/non-functional)

`pages/6_Graph_View.py` was deleted. The Graph tab was removed from `pages/1_Chat.py`; the Chat panel now has two tabs: **Lifecycle** and **Trace**.

`utils/graph_panel.py` and `utils/graph_html.py` were **intentionally kept** in the tree. The page can be restored by recreating `6_Graph_View.py` and adding a Graph tab entry in `1_Chat.py` — no utility code changes needed.

The description of `6_Graph_View.py` in Decision 4 above now describes a deleted file. The `utils/graph_panel.py` / `utils/graph_html.py` helpers remain available for a future restoration.

**Tests:** 176 → 182. Updated: `test_lifecycle_svg`, `test_lifecycle_mapping`, `test_lifecycle_runner`, `test_trace_render` to match the new node IDs, click-link rendering, and removed graph tab.

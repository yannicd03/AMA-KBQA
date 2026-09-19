# Graph Engine (`ama_kbqa/graph/`)

The LangGraph `StateGraph` implementation of the agent loop, selectable
alongside the original hand-written loop in `BaseKBQAAgent` via an engine
switch. This doc covers the graph engine as implemented through Phase 5 of
`Tasks/active/langgraph-rewrite.md` (commit `6adae54`, 2026-08-23). It
describes **current state**; phase-by-phase history and rationale live in
`Decisions/langgraph-adoption.md` and the PRD.

## Related Docs
- [Decisions/langgraph-adoption.md](../Decisions/langgraph-adoption.md) — ADR: why LangGraph, why not `deepagents`/`create_agent`, Phase 0 evidence
- [Tasks/active/langgraph-rewrite.md](../Tasks/active/langgraph-rewrite.md) — full PRD, phase-by-phase results log (§10-§15)
- [System/agent_framework.md](agent_framework.md) — `BaseKBQAAgent` mechanics the graph engine ports (loop detection, trace spans, retry, token tracking, journal, synthesis funnel) — this doc assumes that one as background
- [System/orchestrator_routing.md](orchestrator_routing.md) — legacy `Orchestrator._route_autonomously` that `graph/orchestrator.py` mirrors
- [System/project_architecture.md](project_architecture.md) — full repo structure, config system

---

## Package layout (`ama_kbqa/graph/`, 1934 LOC across 10 files)

| File | Role |
|------|------|
| `__init__.py` | Package docstring: topology ASCII diagram, Phase 2 behaviour-inventory summary, the text-mode gap. No code. |
| `state.py` | `GraphState` — the `TypedDict` threaded through the core tool-loop graph. |
| `messages.py` | OpenAI-dict ⇄ LangChain `BaseMessage` conversion, with stable message ids. |
| `model.py` | `build_chat_model()` — LangChain chat-model factory per provider. |
| `builder.py` | `build_graph()` — compiles the `call_model` ↔ `execute_tools` tool-loop graph. |
| `guards.py` | Pure-function Phase 2 guards: loop detection wrapper, zero-tool-call retry/hard-stop, `max_tool_calls` cap, wrap-up nudge. |
| `context.py` | Phase 2 "swap-and-diff" hooks: context compaction, journal refresh, `GetJournalSummary` answer prompt, raw-SPARQL distress. |
| `runner.py` | `run_tool_loop_graph()` — bridges `BaseKBQAAgent._run_tool_loop` into the compiled graph and back. |
| `pipeline.py` | Phase 3: `build_pipeline_graph()`/`run_pipeline_graph()` — the outer per-question pipeline (`prepare → classify → assemble_prompt → fast_path → tool_loop → finalize`), plus the opt-in checkpointer. |
| `orchestrator.py` | Phase 3b: `build_orchestrator_graph()`/`run_orchestrator_graph()` — the Orchestrator's routing/delegation topology. |

---

## Engine switch and dispatch points

`ama_kbqa/config.py:575 get_agent_engine()` reads `AMA_AGENT_ENGINE` (env,
checked first) or `[agent].engine` in `config.toml` (`config.toml:14`,
default `"legacy"`); an unrecognized value also falls back to `"legacy"`.
Three call sites dispatch on it, each importing the graph module lazily
(inside the function, not at module load):

- **`BaseKBQAAgent._ask_impl`** (`ama_kbqa/framework/base_agent.py:974-986`) —
  if `engine == "graph"` and the agent is not in text-mode
  (`self._text_tool_call_mode`), calls
  `ama_kbqa.graph.pipeline.run_pipeline_graph(self, query, _root_span)` and
  returns its result directly; the entire legacy `_ask_impl` body below is
  skipped.
- **`BaseKBQAAgent._run_tool_loop`** (`base_agent.py:1384-1421`) — same check,
  additionally falling back to the legacy loop (with a warning trace) if the
  agent is in text-mode even though the graph engine was requested. Calls
  `ama_kbqa.graph.runner.run_tool_loop_graph(...)`. This is also the seam the
  pipeline graph's own `tool_loop` node calls through (see below), so the
  text-mode fallback rule applies uniformly regardless of which graph invoked
  it.
- **`Orchestrator.ask`** (`ama_kbqa/agents/orchestrator_agent/agent.py:387-405`)
  — if `engine == "graph"`, calls
  `ama_kbqa.graph.orchestrator.run_orchestrator_graph(self, query)` inside the
  already-open `agent_run` span, skipping the legacy routing/delegation body.

All three checks are independent (each reads `get_agent_engine()` itself),
so in principle the pipeline graph and the tool-loop graph could disagree —
in practice both read the same config/env at the same instant, so they are
consistent for the lifetime of one process.

---

## Two (three) graphs

### 1. Core tool loop (`builder.py:build_graph`, Phase 1/2)

```
START -> call_model -(tool_calls?)-> execute_tools -(cap reached?)-> END
            |    \-(no tool_calls, evidence exists)-> END               |
            |    \-(no tool_calls, zero-tool retry)-> call_model         |
            \-(max_iterations reached)-> END                            v
                                                                   call_model
                                                                   (loop back)
```

State: `GraphState` (`state.py`) — `messages` (`Annotated[List[AnyMessage],
add_messages]`, append-only), `iteration`, `total_tool_calls`, `exit_reason`,
`tool_call_history`, `zero_tool_call_retries`, `final_content`.

- **`call_model`** (`builder.py:153-279`): increments `iteration`; checks the
  iteration cap **before** calling the LLM (`iteration > max_iterations` →
  `exit_reason="max_iterations"`, no LLM call — mirrors
  `base_agent.py:_run_tool_loop` ~:1405); emits the `tool_loop_iter` event
  with the pre-compaction message count; runs
  `context.run_before_model_mutations` (compaction + journal refresh); binds
  tools with `tool_choice="required"` for iteration ≤3 else `"auto"`
  (`base_agent.py` ~:1446); calls the model inside an `llm_call` span,
  routing the `ainvoke` through `agent._retry` itself for every provider
  except `"kit"` (whose `ChatKIT` model already wraps retry — see Chat-model
  factory below); tracks token usage; then branches on whether the response
  has tool calls:
  - has calls → append the assistant message, `exit_reason=None`.
  - no calls → `guards.zero_tool_call_decision` picks `"retry"` (discard the
    turn, append `guards.ZERO_TOOL_RETRY_NUDGE`, `exit_reason="zero_tool_retry"`,
    which `route_after_model` routes back to `call_model` itself — the one
    self-loop in this graph), `"hard_stop"` (`exit_reason="zero_tool_hard_stop"`,
    nothing appended), or `"accept"` (`exit_reason="final_answer"`, message
    appended only if it has content, `final_content` captured either way).
- **`execute_tools`** (`builder.py:281-401`): merges `tool_calls` +
  `invalid_tool_calls` into emission order via `_ordered_tool_calls`
  (LangChain's OpenAI parser splits them into two lists, losing interleave —
  `_sequence_key` recovers order from numeric `call_<n>` id suffixes);
  validates each call (malformed-JSON args → the same error text as legacy;
  unknown tool name → the same error text as legacy); runs
  `guards.apply_loop_detection` per call (which calls
  `agent._detect_loops`/`agent._handle_loop_detected` **unchanged**); executes
  the surviving calls via `agent._execute_single_tool` — one call directly if
  there's exactly one, `asyncio.gather` if more than one, matching legacy's
  concurrent-execution behaviour; builds `ToolMessage`s; runs
  `context.run_after_tools_mutations` (`GetJournalSummary` answer prompt +
  raw-SPARQL distress); checks `guards.max_tool_calls_reached` (exits
  straight to `END` with `exit_reason="max_tool_calls"`, skipping the nudge);
  otherwise appends `guards.wrap_up_nudge_message` if due, and
  `route_after_tools` sends control back to `call_model`.

`execute_tools` deliberately reuses `agent._execute_single_tool` rather than
reimplementing tool execution, so the `tool_call` trace span,
`tool_call_counts`/`tool_call_durations` bookkeeping, and journal snapshotting
after `JOURNAL_MUTATING_TOOLS` are identical between engines "for free"
(`builder.py:20-28`).

**Bridge back:** `runner.py:run_tool_loop_graph` (called by
`BaseKBQAAgent._run_tool_loop`) builds the chat model, compiles the graph,
seeds `initial_state` from `agent._messages`/`agent.tool_call_counts`, runs
`graph.ainvoke` with a generous `recursion_limit` (`2*max_iterations+10`,
floor 50 — each loop pass is two graph steps), resyncs `agent._messages` from
the final state, then replicates legacy's exit-path logic exactly on
`exit_reason`:
- `"max_iterations"` / `"max_tool_calls"` → unconditional `agent._run_synthesis(...)`.
- `"zero_tool_hard_stop"` → returns `guards.ZERO_TOOL_HARD_STOP_ANSWER` literally, **no** synthesis.
- `"final_answer"` → the `get_synthesis_enabled()` bypass (return `final_content` via `agent._finalize_answer_text` if synthesis is disabled and content is non-empty) else `agent._run_synthesis(...)`.

### 2. Outer pipeline (`pipeline.py:build_pipeline_graph`, Phase 3)

```
prepare -> {no_mcp_fallback | classify | assemble_prompt(followup)}
classify -> assemble_prompt -> {tool_loop(followup) | fast_path}
fast_path -> {finalize(success) | tool_loop}
tool_loop -> finalize -> END
no_mcp_fallback -> finalize -> END
```

`PipelineState` (`pipeline.py:126-144`) deliberately carries **no**
`messages` key — the message history lives only on `agent._messages`
throughout, exactly as in legacy `_ask_impl`; state carries only the
decisions made between nodes (qtype, entities, relations, filtered tool
list, the final answer).

- **`prepare`**: `agent._init_mcp()`; if no MCP, routes to `no_mcp_fallback`
  (`agent._llm_call_text_only()`); else lists/converts tools, sets
  `agent._known_tool_names`, `_find_resource_cap`/`_sparql_cap`/`_context_limit`/`_max_tool_calls`
  from `config.domain_settings`, detects follow-up (`any user role message
  already in agent._messages`), reads `max_iterations`/`refresh_interval`.
  Text-mode catalog injection is present but dead code in practice (text-mode
  agents never reach this graph — see below).
- **`classify`**: `agent._classify_question(query)` → qtype, entities,
  relations, fewshot examples (skipped for follow-ups by the conditional edge
  out of `prepare`).
- **`assemble_prompt`**: for a fresh turn, appends the static qtype block
  (`agent._build_static_qtype_context`), the raw query, and the per-question
  context (`agent._build_question_context`) — the prefix-cache-ordered
  assembly; for a follow-up, just appends the query and defaults
  `qtype="Query"`.
- **`fast_path`**: eligibility = qtype in `{QueryAttr, QueryRelation,
  QueryName}` AND exactly 1 entity AND ≤1 relation AND
  `not agent._should_skip_fast_path(...)` AND `enable_fast_path` config. On
  success, records the `fast_path` span, finalizes the answer via
  `agent._finalize_answer_text`, routes to `finalize`. On failure or
  ineligibility, computes the qtype-filtered + denylisted tool list
  (`_filter_tools_for_qtype`, mirroring `_ask_impl` ~:1110-1130) and routes to
  `tool_loop`.
- **`tool_loop`**: calls `agent._run_tool_loop(...)` — **not**
  `runner.run_tool_loop_graph` directly — so the engine dispatch inside
  `_run_tool_loop` (which itself calls `run_tool_loop_graph`) fires exactly
  once, and existing test doubles that override `_run_tool_loop` keep working
  unchanged for the pipeline graph too.
- **`finalize`**: a no-op terminal node; the answer was already set by
  whichever upstream node produced it.

`run_pipeline_graph` (`pipeline.py:458-486`) wraps the whole graph invocation
in the same try/except/finally shape as `_ask_impl`: trace + re-raise on
exception, and **always** call `agent._finalize_question()` in `finally` —
this is why `_finalize_question` (MCP journal read-back + tool-summary trace)
is not itself a graph node: LangGraph does not run downstream nodes after one
raises, so the guarantee has to live in the wrapper.

### 3. Orchestrator graph (`orchestrator.py:build_orchestrator_graph`, Phase 3b)

```
START -> route (probe -> select_agent, in a "classify" span)
       -> conditional edge -> {delegate_kqapro | delegate_sciqa | fallback_kqapro}
       -> END
```

`route` invokes a small routing subgraph (`probe -> select_agent`) inside a
`recorder.span("classify", "route", ...)` — the same span boundary legacy
opens around the whole `_route_autonomously` call — so `probe`/`select_agent`
get genuine node identity while the outer span's `selected_agent`/`route_reason`
attributes still land in the same place.

- **`probe`**: the deterministic MCP `analyze_query_recommend_db` call
  (`agent.PROBE_TOOL_NAME`); no MCP client → `no_mcp` → routes straight to
  `END` of the subgraph (outer graph then falls to `fallback_kqapro`); a
  probe exception degrades to a `{"degraded": true, "note": ...}` evidence
  payload rather than aborting.
- **`select_agent`**: one forced `select_agent` tool call
  (`tool_choice={"type": "function", "function": {"name": "select_agent"}}`)
  via `agent._create_with_retry`; any failure (no tool call, unknown agent
  name, exception) → `selected_agent=None`.
- **`delegate_kqapro`/`delegate_sciqa`**: call `agent._delegate("kqapro_agent"
  | "sciqa_agent", query)` **unchanged** — shares `agent.recorder`, nests the
  sub-agent's `agent_run` span under this run's `delegate` span, falls back to
  `fallback_kqapro` itself on load/ask failure.
- **`fallback_kqapro`**: calls `agent._fallback_kqapro(query)` unchanged
  (which itself falls back further to `agent._fallback_llm` on failure —
  there is no separate top-level `fallback_llm` graph node, because legacy
  never distinguishes that case at the routing layer either).

Routing is a **graph conditional edge**, not a tool returning `Command` — so
the `Command.PARENT` subgraph-routing trap that motivated part of the ADR
cannot arise by construction (nothing in `probe`/`select_agent` is itself a
tool call that could route).

`probe`/`select_agent` are deliberately **copies** of
`Orchestrator._route_autonomously`'s two steps (not one node calling that
method whole) — the PRD wanted them independently observable with a real
edge between them, and `_route_autonomously` has no natural split point.
`orchestrator.py:52-68` flags this explicitly: if `_route_autonomously`
changes, both copies need the same edit.

---

## Behaviour inventory: where each Phase-2 behaviour lives

| Behaviour | Graph location | Reuse strategy |
|---|---|---|
| Loop detection (5 layers) + intervention | `guards.apply_loop_detection`, called from `execute_tools` per proposed call, before execution | Calls `agent._detect_loops`/`agent._handle_loop_detected` **unchanged** — no `self._messages` touched by those methods, so nothing to swap |
| Periodic journal refresh + no-progress template | `context.run_before_model_mutations`, called from `call_model` before the LLM call | Swap-and-diff: calls `agent._inject_journal_refresh(iteration)` unchanged |
| Context-window compaction (hysteresis) | `context.run_before_model_mutations`, same call site, runs **before** refresh | Swap-and-diff: calls `agent._manage_context_window()` unchanged |
| `GetJournalSummary` answer prompt | `context.run_after_tools_mutations`, called from `execute_tools` after tool execution | Swap-and-diff: appends `agent._get_journal_summary_answer_prompt()` |
| Raw-SPARQL distress intervention | `context.run_after_tools_mutations`, same call site | Swap-and-diff: calls `agent._maybe_inject_raw_sparql_distress(iteration)` unchanged |
| Zero-tool-call retry + hard stop | `guards.zero_tool_call_decision`, wired inline into `call_model`'s self-loop (`route_after_model`'s `"retry"` branch) | Reimplemented as pure logic — legacy has no extracted method (inline in the `while True` body) |
| `max_tool_calls` hard cap | `guards.max_tool_calls_reached`, checked inline in `execute_tools` | Reimplemented pure logic; exits straight to synthesis via `exit_reason="max_tool_calls"` |
| Iteration-15+ wrap-up nudge | `guards.wrap_up_nudge_message`, checked inline in `execute_tools` | Reimplemented pure logic, verbatim copy of the legacy nudge text |
| `tool_choice` schedule (`required` ≤3, then `auto`) | `call_model`, `builder.py:181` | Reimplemented (one-liner) |
| `max_iterations` cap | `call_model`, checked before the LLM call | Reimplemented (one-liner) |
| Concurrent tool execution + tool spans/journal snapshots | `execute_tools`, via `agent._execute_single_tool` (single call directly, else `asyncio.gather`) | Fully reused — the method itself is unchanged |

**Not ported to the graph engine at all:** text-mode tool calls
(`framework/text_tool_calls.py`) and the text-mode-only truncated-tool-call
retry. Both `_ask_impl` (`base_agent.py:984`) and `_run_tool_loop`
(`base_agent.py:1412`) check `self._text_tool_call_mode` and fall back to the
legacy code path regardless of the configured engine — see "What is NOT on
the graph engine" below.

---

## The "swap-and-diff" pattern (`context.py`)

`BaseKBQAAgent._manage_context_window`, `_inject_journal_refresh`, and
`_maybe_inject_raw_sparql_distress` all read and mutate `self._messages`
directly — in place for compaction's stub/truncate passes, append-only for
refresh/journal-prompt/distress. None can be called against the live agent
mid-graph-run, because during a run the graph's `GraphState["messages"]` —
not `agent._messages` — is the authoritative message list (`agent._messages`
is only resynced once, at the very end, by `runner.py`).

Re-deriving this logic as pure `list[dict] -> list[dict]` functions was
considered and **rejected**: compaction alone encodes non-trivial,
easy-to-drift behaviour (hysteresis trigger schedule, superseded-refresh
stubbing keyed off a marker, a two-tier aggressive/moderate truncation
policy) that would need to be kept in sync with the legacy method by hand.

Instead, every hook in `context.py` does the same trick
(`context.py:57-132`):
1. Temporarily point `agent._messages` at a plain OpenAI-dict mirror of the
   relevant slice of graph state (`from_lc_messages(state["messages"])`).
2. Call the **unchanged** legacy method — it mutates the mirror as if it were
   the real thing, and everything it touches that isn't `self._messages`
   (recorder events, `agent.mcp`, config getters) still runs against the real
   agent.
3. Restore `agent._messages` to its original value (`finally`).
4. Diff the mutated mirror against the original (`_diff_to_lc_updates`) to
   produce a LangGraph message-list update: in-place edits become a
   replacement LangChain message carrying the **same id** as the message it
   replaces (so `add_messages` overwrites in place instead of duplicating —
   this is why `messages.to_lc_messages` guarantees every message an id);
   pure appends become new messages with fresh ids.

This keeps every one of these behaviours byte-for-byte identical to legacy
(same thresholds, same templates, same recorder events) while adding zero
duplicated logic — at the cost of a genuinely transitional pattern flagged in
the module docstring and the PRD (§Phase 2 result): **Phase 6 must move this
logic into graph-owned code before `base_agent.py`'s loop is deleted**, since
the swap trick only works because the legacy methods still exist to be
called.

---

## Message conversion and stable ids (`messages.py`)

`to_lc_messages(messages: List[Dict]) -> List[AnyMessage]` wraps
`langchain_core.messages.convert_to_messages`, then assigns every message a
random `uuid.uuid4()` id if it doesn't already have one. This matters because
`GraphState["messages"]` uses LangGraph's `add_messages` reducer, which
treats a re-added message with the **same** id as an in-place replacement
rather than an append — the mechanism `context.py`'s swap-and-diff pattern
depends on. IDs are unique within one run only, never reproducible across
runs, and never leak into `agent._messages` (the inverse conversion only
reads OpenAI-dict fields, never `.id`).

`from_lc_messages(messages: List[AnyMessage]) -> List[Dict]` wraps
`convert_to_openai_messages` and patches one asymmetry: an assistant message
with only `tool_calls` (no text) round-trips with `content == ""`, but this
repo's dicts (and the raw OpenAI API response they came from) use
`content: None` for that case — restored explicitly so the round trip is
lossless (`tests/graph/test_message_roundtrip.py`).

Not used for text-mode agents: those store tool calls as
`<tool_call>...</tool_call>` text inside `role=user`/`role=assistant`
messages, which `convert_to_messages` has no concept of.

---

## Chat-model factory (`model.py:build_chat_model`)

Mirrors `ama_kbqa.config._create_client` (base_url, API-key env-var mapping,
`httpx.Timeout(connect=20, read=60, write=10, pool=5)`, OpenRouter headers/
provider preferences, `AMA_LLM_SEED`) but returns a LangChain
`BaseChatModel` instead of a raw `openai.OpenAI` client, since `call_model`
drives the model through `bind_tools`/`ainvoke`.

- **`provider == "kit"`**: returns `chatkit.client.get_kit_model(model=...,
  api_key=..., base_url=..., temperature=..., max_tokens=..., max_retries=0,
  timeout=_TIMEOUT, streaming=False, preflight=False, retry=agent._retry,
  ...)`. `ChatKIT` already wraps every `_agenerate` call in whatever
  `TransientRetry` instance is passed as `retry=`, so `call_model` does
  **not** wrap the `ainvoke` call again (that would double the backoff —
  `builder.py:194-199`).
- **Every other provider**: `ChatOpenAI(max_retries=0, ...)`, which has no
  such built-in hook — `call_model` wraps its own `ainvoke` call in
  `agent._retry.arun(...)` instead (`builder.py:200-203`).

Both branches set `max_retries=0` on the underlying SDK client, matching the
legacy double-retry-avoidance pattern documented in `agent_framework.md`'s
Transient LLM Retry section (`get_chat_client(max_retries=0)`).

---

## Token accounting

`builder.py:_track_token_usage_from_message` (`builder.py:77-106`) reads
`AIMessage.usage_metadata` (LangChain-standard `input_tokens`/
`output_tokens`/`total_tokens`) first, falling back to the raw OpenAI-shaped
`response_metadata["token_usage"]` dict for providers that only populate
that. Adds into `agent.token_usage["prompt_tokens"/"completion_tokens"/"total_tokens"]`
— the same dict legacy's `_track_token_usage` updates — so downstream
consumers (`get_tool_call_summary()`, benchmark aggregation) don't need to
know which engine ran. This is why token accounting is read from the message
object in an explicit node rather than via LangChain callbacks: callbacks do
not reliably propagate across the kind of subgraph boundaries this rewrite
uses (noted as a risk in `Decisions/langgraph-adoption.md`'s Consequences).

---

## Trace spans

The graph engine emits the **same span/event names** as the legacy loop, so
the frontend's Trace Inspector and `trace_render.py` need no engine-specific
branching:

| Kind | Where |
|---|---|
| `agent_run` (span) | Opened by `BaseKBQAAgent.ask()`/`Orchestrator.ask()` before either engine dispatches — unchanged for both. |
| `classify` (span) | `orchestrator.py`'s `route` node wraps the routing subgraph in exactly the span boundary legacy opens around `_route_autonomously`. |
| `fast_path` (span) | `pipeline.py`'s `fast_path` node. |
| `llm_call` (span) | `builder.py`'s `call_model`, one per LLM call, same attributes (`model`, `n_messages`, `n_tools`, `tool_choice`) plus token-usage attributes and a `tool_calls`/`assistant_content` payload. |
| `tool_call` (span) | Opened by `agent._execute_single_tool` itself (reused unchanged) — one per tool, concurrent calls parent correctly via the same `ContextVar` mechanism as legacy. |
| `synthesis` (span) | Opened by `agent._run_synthesis`, called unchanged from `runner.py`. |
| `delegate` (span) | Opened by `agent._delegate`, called unchanged from `orchestrator.py`'s `delegate_kqapro`/`delegate_sciqa`. |
| `tool_loop_iter` / `intervention` (events) | Emitted directly by `builder.py` (`call_model`'s iteration marker; `execute_tools`'/`call_model`'s zero-tool-retry, zero-tool-hard-stop, max-tool-calls, loop-detected events) — same event names as `base_agent.py`. |

---

## Opt-in checkpointer and `thread_id` scheme (Phase 4, `pipeline.py`)

Configured via `[agent].checkpointer` / `AMA_AGENT_CHECKPOINTER`
(`config.py:600-620`, `config.toml:15-22`): `"none"` (default), `"memory"`,
or `"sqlite"` (path via `[agent].checkpointer_path` /
`AMA_AGENT_CHECKPOINTER_PATH`, default `benchmark_results/checkpoints.sqlite`
relative to repo root). Only consulted when `engine == "graph"`; the legacy
engine never attaches a checkpointer.

`pipeline._resolve_checkpointer` (`pipeline.py:172-215`) lazily builds and
caches the checkpointer on `agent._graph_checkpointer` (so it persists across
questions answered by the same agent instance, not just within one
`ask()` call): `"memory"` → `langgraph.checkpoint.memory.MemorySaver()`;
`"sqlite"` → `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`, entered once
via `from_conn_string(...).__aenter__()` and kept open for the agent's
lifetime (never `__aexit__`'d); `"none"` → `None`, compiling the pipeline
graph exactly as before this option existed (byte-identical).

`pipeline._next_thread_id` (`pipeline.py:218-232`) returns
`f"{session_id}::{counter}"` — `session_id` (default `"default"`) segments
agent instances/conversations, and a per-agent counter (incremented every
call, **never** reset by `soft_reset()`/`reset()`) segments consecutive
questions from the same instance, so every question gets its own thread even
though the checkpointer instance itself persists across questions within a
benchmark run.

Known gap (documented in the PRD §14, not a defect): mid-question `--resume`
on top of the sqlite checkpointer is a follow-up — it needs a
`(output_dir, question_index)`-derived thread id and an `aget_state` probe in
`is_run_completed()`, neither of which exists yet. What Phase 4 does deliver
is per-question `--resume` (skip already-completed questions), which needed
no graph-engine-specific work at all.

---

## What is NOT on the graph engine

- **Text-mode tool calls** (`framework/text_tool_calls.py`, models like
  `minimax-m2.7` that can't emit native `tool_calls`): both dispatch points
  (`_ask_impl:984`, `_run_tool_loop:1412`) check
  `self._text_tool_call_mode` and unconditionally fall back to the legacy
  code path, regardless of the configured engine — `_run_tool_loop` even logs
  a warning trace when this happens (`base_agent.py:1417-1421`). The
  `messages.py` conversion has no concept of the `<tool_call>` XML text
  format these agents use, so this isn't a missing feature so much as a
  structural incompatibility the graph engine doesn't attempt to bridge.
- **Journal option B** (moving `JournalState` into graph state with a
  reducer): explicitly postponed past Phase 5 per the ADR's Decision §2; the
  journal still lives server-side inside the MCP tool processes, mirrored via
  `journal_snapshots` exactly as in legacy.
- **The `context.py` swap-and-diff mechanism itself**: a deliberate
  transitional device, not a permanent design — see that section above.
- **LangSmith tracing**: opt-in only (`LANGSMITH_TRACING=true` + API key);
  nothing in either graph requires it, and `TraceRecorder` remains the
  primary trace source for both engines.

---

## Test layout (`tests/graph/`, ~68 test functions across 13 files)

| File | Covers |
|---|---|
| `_fakes.py` | Not a test module. `ScriptedChatModel` (a `BaseChatModel` returning one scripted `AIMessage` per call, records every `tool_choice` `bind_tools` was called with) and `GraphAgentDouble` (a minimal `BaseKBQAAgent` subclass with sane defaults for every attribute the Phase 2 reused methods expect), shared by most files below. |
| `test_graph_loop.py` (7) | Phase 1 core-loop contract against `builder.build_graph` directly: two tool rounds then a final answer, the `tool_choice` schedule, malformed-JSON tool args, concurrent execution, `max_iterations` cap. |
| `test_guards.py` (10) | Unit tests for the pure helpers in `guards.py` in isolation (no full loop). |
| `test_context.py` (8) | Unit tests for the swap-and-diff **mechanism** itself (id-preserving in-place replacement vs. append) against a minimal fake agent with scripted mutators — isolated from the legacy methods' own logic, which is covered elsewhere (`tests/framework/test_prefix_cache_history.py`, `tests/framework/test_raw_sparql_distress.py`). |
| `test_message_roundtrip.py` (4) | `to_lc_messages`/`from_lc_messages` round-trip losslessness, including the `content: None` vs `""` fix. |
| `test_model_factory.py` (5) | `build_chat_model` builds the right LangChain class per provider without opening a socket; asserts `preflight=False` for `"kit"`. |
| `test_runner.py` (3) | `run_tool_loop_graph`'s two exit paths (`max_iterations`, final answer) against a real `BaseKBQAAgent` subclass double, `build_chat_model` monkeypatched to a scripted fake. |
| `test_legacy_graph_parity.py` (1) | `test_legacy_and_graph_engines_produce_identical_messages_and_usage` — the core Phase 1 parity gate: both engines driven with the same scripted 2-round transcript via `_ParityAgent`. |
| `test_parity_phase2.py` (9) | One legacy-vs-graph parity test per Phase 2 behaviour: loop detection, periodic journal refresh, context compaction, `GetJournalSummary` answer prompt, raw-SPARQL distress, `max_tool_calls` cap, wrap-up nudge, zero-tool-call retry/hard-stop, empty-final-content handling. |
| `test_pipeline.py` (7) | Phase 3 outer pipeline: fresh-question routing through classify/assemble/tool_loop, message-shape parity with legacy `_ask_impl` at the point the tool loop is entered, follow-up short-circuiting, fast-path success/failure/ineligibility, exception propagation with `_finalize_question` still firing. |
| `test_pipeline_checkpointer.py` (6) | Phase 4: `checkpointer="none"` byte-identical compile, `"memory"` end-to-end run + retrievable checkpoint via `graph.aget_state(...)`, two consecutive `ask()` calls on one agent instance getting two distinct `thread_id`s. |
| `test_orchestrator_graph.py` (8) | Phase 3b: routing to each sub-agent, no-MCP fallback, probe-failure degradation, no-tool-call/unknown-agent/exception fallback paths, and a full legacy-vs-graph `ask()` comparison for recorder-event-kind parity. |

`GraphAgentDouble` and `_ParityAgent` (defined in `test_legacy_graph_parity.py`)
are the two base test-double classes; `PipelineAgentDouble`
(`test_pipeline.py`) extends `GraphAgentDouble` with the MCP/config/
classification surface `pipeline.py`'s nodes need. All doubles bypass a real
MCP connection and a real LLM provider entirely — none of these tests make
network calls.

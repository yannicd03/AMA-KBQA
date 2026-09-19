# PRD: Rewrite the agent pipeline on LangGraph (explicit `StateGraph`)

> **Scope amendment (user, 2026-08-22):** plain LangGraph only. No `deepagents`, no `create_agent`, no prebuilt
> `langchain.agents.middleware` stack. The tool loop is an explicit `StateGraph` (model node, tools node, guard
> nodes, conditional edges). The hook-by-hook mapping in §2 and the stack in §4.4 remain the *behaviour inventory*
> to port; each row becomes a graph node or a pure helper called from one, not an `AgentMiddleware`.
> Dependencies therefore reduce to `langgraph`, `langchain-core`, `langchain-openai`, `langchain-mcp-adapters`,
> `chatkit[langchain]>=1.1.0`.

Status: **approved 2026-08-22, in implementation** on branch `feat/langgraph-rewrite` (separate worktree). Pre-rewrite reference commit tagged `pre-langgraph-rewrite`.
Supersedes the scope of `0_Claude/AMA-KBQA/Feasibility - LangGraph Migration Token and Latency Impact.md` (2026-04-05), which rejected LangGraph *for token/latency reasons* and explicitly said to reconsider "if we're refactoring the agent architecture for other reasons and want to adopt a standard framework". That is this PRD's premise. Nothing here claims a token or latency win.

## 1. Verdict

| Question | Answer |
|---|---|
| Switch to LangGraph + `langchain.agents.middleware`? | **Yes, conditionally.** The structure fits (see §2) and middleware removes most of the cost the April note priced at 3-4 weeks: every custom loop behaviour in `base_agent.py` maps to exactly one middleware hook. Condition: the Phase 0 prefix-cache gate (§5) passes. |
| Use `deepagents`? | **No.** See §3. Borrow its *patterns* (middleware-as-harness, one state key + one tool + one prompt hook per concern), not the package. |
| Replace the scratchpad with `TodoListMiddleware`? | **No.** The journal is structured KG-exploration state, auto-populated by tools and consumed by synthesis; `write_todos` is a free-text plan list. At most it replaces the `current_plan`/`completed_steps` fields. Build a `JournalMiddleware` in the same shape as `TodoListMiddleware` instead (§4.3). |
| What do we actually gain? | Decomposition of the 2,702-LOC `BaseKBQAAgent` god class into ~10 independently testable middleware; native streaming (deletes the thread+queue bridge in `frontend/utils/lifecycle_runner.py`); checkpointing (absorbs `benchmark_persistence_refactor.md`); per-request tool override (absorbs `deferred-tool-loading.md`); standard graph for the routing/fast-path/synthesis topology; optional LangSmith on top of the existing `TraceRecorder`. |
| What do we risk? | Prefix-cache regression from any change in serialized message/tool shape (measured: a 45-token tools-list change ≈ 3x TTFT, `prompt-cache-utilization.md` §4); token-accounting gaps from LangChain callbacks not crossing `create_agent` subgraphs; reproducibility of the SEMANTiCS numbers (pin the paper to the pre-rewrite commit). |

## 2. Why the structure fits

Current control flow (`ama_kbqa/framework/base_agent.py`):

```
ask()  → classify (JSON-mode LLM, :619)
       → follow-up? skip classify/fast-path (:1019)
       → fast path for simple qtypes (:1189)            ─┐ success → answer
       → tool loop (:1371) with interventions            │
       → synthesis (:2198, optional)                   ←─┘ failure
       → finalize (:2316)
Orchestrator (orchestrator_agent/agent.py): probe → forced select_agent → delegate → fallback
```

This is already a graph with conditional edges; it is just written as nested `if`/`while` inside one class. The tool loop itself is structurally `create_agent` (the April note already established this for `create_react_agent`). What was expensive in April (reimplementing loop behaviours as custom nodes/`state_modifier`) is now the designed extension point:

| Current behaviour (`base_agent.py`) | Hook | Notes |
|---|---|---|
| qtype tool filter + denylist (`:1112-1130`), 300-char schema compression (`mcp_client.py:226`) | `wrap_model_call` → `request.override(tools=...)` | Per-request tool override is first-class. Also the natural home for `deferred-tool-loading.md`. |
| `tool_choice="required"` for iterations ≤3 (`:1446`) | `wrap_model_call` | Trivial. Measured to not affect the cache prefix. |
| Text-mode tool calls for non-native models (`framework/text_tool_calls.py`) | `wrap_model_call` | Strip tools, inject catalog, parse text into `AIMessage.tool_calls`. Self-contained. |
| Journal refresh every N iterations, append-only, no-progress template (`:2005-2064`) | `before_model` (+ state key `journal`) | See §4.3. |
| Context compaction with hysteresis 0.5→0.9 (`:2066`, `:2178`) | `wrap_model_call` | Port as a *view* over state (state keeps full history for traces; model sees the compacted prefix). Must stay deterministic across calls or the cache is lost. |
| Loop detection, 5 layers (`:841`), intervention (`:1962`), SPARQL distress (`:1679`) | `after_model` | Inspect proposed `tool_calls` vs history; rewrite/append intervention; `jump_to="end"` on hard caps. |
| `max_iterations`, `max_tool_calls`, exit through synthesis (`:1405`, `:1633`) | prebuilt `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` (`exit_behavior="end"`) | Prebuilt fits here. Wrap-up nudge (every 5 from iter 15) is a 5-line `before_model`. |
| Zero-tool-call retry (`:1539`), truncated tool-call retry (`:1496`) | `after_model` | Same shape as loop detection. |
| Concurrent tool execution, 2000-char truncation, timing, spans (`:1719-1933`) | `wrap_tool_call` + LangGraph's parallel tool node | Concurrency comes free. Truncation + span + journal snapshot in one wrapper. |
| `TransientRetry` around every LLM call (`:2348`) | model object (`chatkit[langchain].ChatKIT`, v1.1.0+) | Not a middleware. Keep SDK `max_retries=0` as today. Do **not** add `ModelRetryMiddleware` on top (double backoff; see `deepagents-langgraph.md` 2026-08-21). |
| Token accounting (`:2652`) | read `AIMessage.usage_metadata` in `after_model` | Not callbacks: callbacks do not propagate into `create_agent` subgraphs (`openrouter-langchain-gotchas.md`). |

## 3. Why not `deepagents`

Verified against `deepagents` 0.7.x (wiki `deepagents-langgraph.md`, 2026-08-13/18 entries, plus docs re-checked 2026-08-22):

- `FilesystemMiddleware` and `SubAgentMiddleware` **cannot be removed** (`excluded_middleware` raises). You can only hide their tools. That is 7+ tool schemas of dead weight on the request path of a project where tool-schema bytes are the single most cache-sensitive thing we measured.
- `SummarizationMiddleware` is always on and compacts the message history by LLM summarization at 85% of context. Our compaction is deliberately *not* that (journal-based, append-only, cache-preserving); we would have to override it by name just to neutralize it.
- Pulls `langchain-anthropic` and `langchain-google-genai` as hard deps; we use OpenAI-compatible endpoints only (`config.py:235`).
- Its system prompt and tool surface assume a coding/file-centric agent. Our tools are MCP KG operations.
- Its value (filesystem offloading, subagent context quarantine, skills, sandboxes) solves long-horizon coding tasks, not a 10-15 call KG exploration loop.

The one deepagents idea worth copying is the *shape* of `TodoListMiddleware`: one state key, one tool, one `before_model` prompt contribution. That is exactly how `JournalMiddleware` should be built.

## 4. Target architecture

### 4.1 Dependencies

Add: `langchain>=1.3.14`, `langgraph>=1.0`, `langchain-openai`, `langchain-mcp-adapters`, `chatkit[langchain]>=1.1.0` (bump from v1.0.0; 1.1.0 is the first release where the LangChain `ChatKIT` surface has `TransientRetry`). Python 3.12 and pydantic 2.12.4 already satisfy the `langchain-core>=1.5` floor (the blocker that hit manga-image-translator does not apply here).
Remove: `openai-agents`, `fastapi`, `uvicorn` (declared, never imported).
Delete `tests/llm/test_kit_library.py::test_langchain_is_not_installed` and `test_langchain_only_names_raise_helpful_error`, with an ADR superseding `Decisions/transient-retry-and-chatkit-extraction.md`'s "langchain-free repo" clause.

### 4.2 Graphs

```
KGAgentGraph (one per adapter; state = AgentState + journal + question meta + budgets)
  START → classify ──(follow-up)──────────────────┐
             │ (simple qtype, 1 entity, ≤1 rel)    │
             ▼                                     ▼
          fast_path ──success──► finalize ◄── tool_loop  (create_agent subgraph w/ middleware)
             │ fail                    ▲              │
             └────────────────────────►│◄── synthesis ┘ (node present only if synthesis_enabled)
                                       ▼
                                      END

OrchestratorGraph
  START → probe (MCP analyze_query_recommend_db) → select_agent (forced enum tool call)
        → conditional edge → kqapro_graph | sciqa_graph (subgraphs) → fallback_llm → END
```

Routing is a graph edge, not a tool returning `Command`, so the `Command.PARENT` trap in `ORCA/langgraph-subgraph-routing.md` is avoided by construction. Any future tool that *does* route must set `graph=Command.PARENT` or it silently routes inside the inner subgraph.

`classify` and `synthesis` are plain LLM nodes (structured output for classify). They keep their own prompts and, per `prompt-cache-utilization.md` option 2, may later share a static leading prefix with the tool-loop system prompt; that is orthogonal to this PRD.

### 4.3 Journal (scratchpad) design

Today the journal lives **inside the MCP server process** (`kqapro_server.py:1716 ManageJournal`, `sciqa_server.py:5810`), auto-mutated by tools, and mirrored client-side via `_snapshot_journal()` calls after every `JOURNAL_MUTATING_TOOLS` member. Two options:

- **A (minimal, recommended for the first cut):** keep the journal server-side. `JournalMiddleware.wrap_tool_call` snapshots after mutating tools (as today) into `state["journal"]`; `before_model` appends the periodic refresh message; synthesis node reads `state["journal"]`. Zero change to the 11k LOC of tool servers.
- **B (follow-up):** move `JournalState` into graph state with a reducer; tools return journal deltas in MCP structured content; `ManageJournal` becomes a client-side tool provided by the middleware (exactly the `write_todos` pattern). Gains: checkpointable journal, no MCP round trip for `read`/`clear`, UI reads state instead of snapshots. Cost: touch every mutating tool in both servers. Do this only after A reaches parity.

`TodoListMiddleware` may be added as an *experiment* on top of A for weak KIT models (LangChain's own claim is that explicit planning helps less capable models), replacing the `current_plan`/`completed_steps` fields. Measure; do not assume.

### 4.4 Middleware stack (ordered, outermost first)

1. `TraceMiddleware` (wrap_model_call + wrap_tool_call): spans/events into the existing `TraceRecorder` (`framework/trace.py`). Preserved unchanged; it is renderer-agnostic and the frontend depends on it.
2. `ToolGatingMiddleware` (wrap_model_call): qtype filter, denylist, compressed schemas, `tool_choice` schedule. Owns the tool list the model sees; later hosts deferred loading.
3. `TextModeToolCallMiddleware` (wrap_model_call): only attached when `needs_text_tool_calls(model)`.
4. `ContextBudgetMiddleware` (wrap_model_call): hysteresis compaction view.
5. `JournalMiddleware` (before_model, wrap_tool_call): §4.3.
6. `GuardrailMiddleware` (after_model): loop detection, zero-tool retry, truncated-call retry, SPARQL distress, wrap-up nudge. One class, because all five need the same call-history view; split later if it grows.
7. Prebuilt `ModelCallLimitMiddleware(run_limit=max_iterations, exit_behavior="end")`, `ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="end")`.
8. Prebuilt `ToolErrorMiddleware` (tool exceptions → error `ToolMessage`, matching current behaviour).
9. `ToolResultMiddleware` (wrap_tool_call): 2000-char truncation, timing, `get_tool_call_summary()` data.

Explicitly **not** used: `SummarizationMiddleware`, `ContextEditingMiddleware` (both mutate the prefix; conflict with journal-based compaction), `LLMToolSelectorMiddleware` (an extra LLM call to do what qtype gating does for free), `ModelRetryMiddleware` (§2).

### 4.5 Provider layer

`config.py:235 _create_client()` becomes a `ChatOpenAI`/`ChatKIT` factory with the same `base_url`, `httpx.Timeout(connect=20, read=60, write=10, pool=5)`, OpenRouter headers and `extra_body={"provider": ...}`, `max_retries=0`. Seed via `AMA_LLM_SEED` as today. Structured output: `with_structured_output(method="json_mode")` for classify/judge/fewshot, keeping `_extract_json_object()` as the fallback parser for `<think>`-prefixed responses (`classifier-think-prefix-fix.md`).

### 4.6 Facade for harness and frontend

Keep `BaseKBQAAgent.ask(question) -> str`, `token_usage`, `get_tool_call_summary()`, `journal_snapshots`, `_messages` (property over `state["messages"]`), `soft_reset()` (new `thread_id`), `reset()`. `benchmark_agents.py:878 process_single_question` and `frontend/utils/agent_factory.py` then need no semantic changes in Phase 1-3. `lifecycle_runner.py` switches from thread+queue to `graph.astream(stream_mode=["updates","custom"])` in Phase 4.

Add `MemorySaver` checkpointer by default; `SqliteSaver` for benchmark runs so `--resume` can resume mid-question and runs survive a Streamlit session (absorbs `benchmark_persistence_refactor.md`).

## 5. Phases and gates

**Phase 0, spike (≈3 days). Go/no-go gate.**
Minimal `create_agent` + `langchain-mcp-adapters` over the KQAPro server, KIT model, no middleware except ToolGating (so tool schemas are byte-identical to today's `convert_tools_to_openai_format`). Log the raw request bodies from both the old loop and the spike via an httpx event hook and diff them. Then run the `prefix_cache_test.py` protocol from `prompt-cache-utilization.md` on 20 questions. **Gate:** warm/cold TTFT ratio and prompt-token counts within noise of the current loop. If LangChain's serialization changes tool/message shape in a way we cannot override, stop here and record why.

**Phase 1, foundation (≈1 week).** Deps, provider factory, MCP tool loading, `TraceMiddleware`, `ToolGating`, facade `ask()`. Tests: request-shape golden tests from Phase 0; unit tests per middleware using `tests/**/tool_traces/question_NNN.json` as fixtures (replay a recorded trace through a middleware and assert its decision).

**Phase 2, behaviour port (≈1.5 weeks).** Remaining middleware, one at a time, each with a parity test against the old implementation's decision on recorded traces (loop detection, compaction thresholds, journal refresh cadence, zero-tool retry are all pure functions of message history and are testable without an LLM).

**Phase 3, graph topology (≈1 week).** classify / fast_path / synthesis / finalize nodes; orchestrator graph; multiturn follow-up edge (`multiturn-direct-agent-conversation.md` semantics preserved).

**Phase 4, harness and frontend (≈1 week).** `benchmark_agents.py` on the graph, checkpointer, streaming in `lifecycle_runner.py`, trace panel fed from the same `TraceRecorder`.

**Phase 5, parity benchmark. Acceptance gate.** Full KQAPro + SciQA runs at one commit, same model and config as the last reference run (resolve from `run_manifest.json`, per `.agent/SOP/refreshing_paper_numbers.md`). Accept if accuracy is within the judge's run-to-run variance and prompt tokens/question and TTFT are within 10%. Otherwise bisect by middleware.

**Phase 6, cleanup.** Delete `base_agent.py` loop code, remove dead deps, ADR, `.agent/System/agent_framework.md` rewrite, `docs-agent` capture.

Total: ≈5-6 weeks of calendar work at the implementation cadence of recent PRDs; Phases 1-4 are well suited to Sonnet subagents with the orchestrator reviewing (per memory `delegate-implementation-to-sonnet`).

## 6. In-flight PRDs

| PRD | Disposition |
|---|---|
| `deferred-tool-loading.md` | **Absorbed** into `ToolGatingMiddleware` design (per-request `tools` override makes two-tier loading a few lines). |
| `benchmark_persistence_refactor.md` | **Absorbed** via checkpointer (Phase 4). |
| `prompt-cache-utilization.md`, `retrieval-fusion-ab.md` | Complete; archive. Option 2 (shared prefix for classify/synthesis) becomes a follow-up on the new graph. |
| `locate-term-tool.md`, `self-diagnosing-empty-results.md`, `selectextreme-relation-filter.md`, `sciqa-grouped-aggregation-tools.md`, `fewshot-bank-audit.md` | **Independent**: all server-side tool or data work, land on `main` as usual; the rewrite consumes tools through MCP unchanged. |

## 7. Invariants to preserve (settled decisions, not up for re-litigation)

- Append-only message growth and deterministic compaction (prefix cache).
- Every failure path exits through synthesis/finalize with a string answer, never an exception to the harness (`agent_framework.md`).
- Abstract-operation bindings per adapter (`abstract-operation-contract.md`); adapters stay the only KG-specific config.
- Evidence-based routing, one LLM round-trip (`orchestrator-evidence-based-routing.md`).
- `TraceRecorder` as the OTel-shaped trace source (`trace-recorder-architecture`, wiki).
- `TransientRetry` owns retries, SDK retries at 0.

## 8. Decisions (user, 2026-08-22)

1. SEMANTiCS camera-ready has been handed in. The paper's reference commit is tagged `pre-langgraph-rewrite` on `main`; numbers are reproducible from that tag regardless of what lands afterwards.
2. Journal option B (journal in graph state) is **postponed until after the rewrite reaches parity**. See the explicit TODO in §9; it must not be silently dropped.
3. LangSmith is **opt-in only** (`LANGSMITH_TRACING=true` + API key in env); `TraceRecorder` stays the primary trace source and nothing in the code path requires LangSmith.
4. Implementation on a new branch in a new worktree, bulk implementation delegated to Sonnet subagents, orchestrator reviews/verifies/commits.

## 9. Follow-up TODOs (after parity)

- [ ] **Journal option B**: move `JournalState` into graph state with a reducer; tools return journal deltas via MCP structured content; `ManageJournal` becomes a client-side middleware tool (`write_todos` pattern). Gains checkpointable journal, no MCP round trip for read/clear, UI reads state. Touches every mutating tool in both servers; do only after Phase 5 acceptance.
- [ ] `TodoListMiddleware` experiment on weak KIT models, replacing `current_plan`/`completed_steps` only. Measure, don't assume.
- [ ] `prompt-cache-utilization.md` option 2 (shared static prefix for classify/synthesis) on the new graph.

## 10. Phase 0 result (2026-08-22): GO

Evidence in `langgraph-rewrite-phase0/` (`request_diff.md`, `cache_gate.md`); scripts `scripts/langgraph_spike.py`, `scripts/prefix_cache_gate.py`.

- Request bodies: `tools` (15, incl. 300-char compressed descriptions) and `messages` deep-equal between `BaseKBQAAgent` and an explicit `StateGraph` + `ChatOpenAI` when raw OpenAI tool dicts are passed to `bind_tools`. Only differences: `max_tokens` → `max_completion_tokens` (LangChain renames unconditionally; verified KIT honours both identically, `finish_reason=length` at 5 tokens) and an explicit `stream: false`.
- Cache gate (KIT, `kit.gemma4-31b-it`, 18 requests): prompt_tokens 5213-5214 raw vs 5215 LangGraph; median TTFT raw 18.1/17.1/19.3s vs LangGraph 13.5/11.0/7.4s (cold/warm/grown). No path-specific regression; endpoint noise dominated this run.
- Resolved: langgraph 1.2.11, langchain-core 1.6.0, langchain-openai 1.6.0, langchain-mcp-adapters 0.3.2, chatkit 1.1.0. Side effect: `openai` 2.8 → 3.3.1, which vendors `httpx2`; any transport mocking must target `httpx2`.
- Phase 1 requirements learned: never let `langchain-mcp-adapters` auto-schemas reach the model (use `convert_tools_to_openai_format`); `tool_choice="required"` passes through verbatim; read "messages as sent" from the request, not from a live list.
- `tests/llm/test_kit_library.py::test_langchain_only_names_raise_helpful_error` now fails by design; removed in Phase 1 with the ADR.

## 11. Phase 1 result (2026-08-23): landed, `21c855d`

- `ama_kbqa/graph/` holds the explicit `StateGraph` loop (`call_model` ↔ `execute_tools`), the OpenAI⇄LangChain message conversion, and a chat-model factory mirroring `config._create_client` (KIT via `chatkit.get_kit_model(streaming=False, retry=agent._retry)`, others via `ChatOpenAI(max_retries=0)` wrapped in the agent's `TransientRetry`).
- Engine switch `[agent].engine` / `AMA_AGENT_ENGINE` (default `legacy`). `BaseKBQAAgent._run_tool_loop` dispatches; everything around the loop (classify, fast path, synthesis, finalize) is unchanged, so the harness and frontend work on both engines.
- `execute_tools` reuses `agent._execute_single_tool`, so tool spans, timing, and journal snapshots are identical by construction. Legacy/graph parity test passes on a scripted 2-round transcript. 560 tests green.
- Live smoke (OpenRouter `openai/gpt-4.1-mini`; KIT was unresponsive and the configured `openrouter/elephant-alpha` model has been retired, config still points at it): both engines answer "Christopher Nolan", 6 tool calls, 22.8k vs 22.9k prompt tokens, same span kinds.
- Dropped `langchain-mcp-adapters` again (spike-only); declared `tqdm` (was transitive). Deleted the langchain-free guard tests.

## 12. Phase 2 result (2026-08-23): landed, `3147263`

- All nine loop interventions run on the graph engine: loop detection (reuses `_detect_loops`/`_handle_loop_detected` unchanged), zero-tool-call retry + hard stop (`call_model` self-loop), `max_tool_calls` cap, wrap-up nudge (`graph/guards.py`); context compaction, periodic journal refresh, `GetJournalSummary` answer prompt, raw-SPARQL distress (`graph/context.py`).
- `context.py` uses a transitional "swap-and-diff" pattern: it points `agent._messages` at a dict mirror of graph state, calls the legacy method verbatim, and diffs the result back into `add_messages` updates (in-place edits keep the message id). **Phase 6 must move that logic into graph-owned code before `base_agent.py`'s loop is deleted.**
- One legacy-vs-graph parity test per behaviour (`tests/graph/test_parity_phase2.py`) plus unit tests; 587 tests green. Remaining gap: text-mode tool calls still use the legacy loop.
- Live check (OpenRouter `gpt-4.1-mini`): correct answer, 6 tool calls, 23.0k prompt tokens; no intervention fired on the simple question, as expected.

## 13. Phase 3 result (2026-08-23): landed, `4cc1600`

- `graph/pipeline.py`: `prepare → classify → assemble_prompt → fast_path → tool_loop → finalize` (+ `no_mcp_fallback`), every node delegating to the existing `BaseKBQAAgent` hook; the message list entering the loop is asserted byte-equal to legacy. Follow-ups bypass classify and tool filtering. Exceptions propagate after `_finalize_question`, exactly as `_ask_impl` does (the PRD §7 wording "never an exception to the harness" was imprecise; the harness catches per question).
- `graph/orchestrator.py`: `probe → select_agent` inside the `classify` span, then a conditional edge to `delegate_kqapro` / `delegate_sciqa` / `fallback_kqapro` (which nests `_fallback_llm` as legacy does). `probe`/`select_agent` duplicate `_route_autonomously`'s body (kept in sync note); Phase 6 should make legacy call the graph helpers instead.
- 602 tests; live KQAPro and CLI-orchestrator runs correct on both engines. Note: both engines show a pre-existing `anyio` MCP stdio-shutdown `RuntimeError` on CLI cleanup.

## 14. Phase 4 result (2026-08-23): landed, `80ffca9`

- No harness/frontend code change needed: with `AMA_AGENT_ENGINE=graph` the benchmark writes `results.json`, `summary.json`, `tool_traces/`, `judgments.json`, `run_manifest.json` with key sets identical to legacy; `--resume` skips completed runs; `--concurrency 2` works; the frontend lifecycle runner streams the same span kinds in the same order (verified through `lifecycle_runner.start_run`, no Streamlit).
- Opt-in checkpointer: `[agent].checkpointer = none|memory|sqlite`, `checkpointer_path`, env `AMA_AGENT_CHECKPOINTER[_PATH]`; per-question `thread_id = session_id::counter`. `none` is byte-identical to before. Mid-question `--resume` on top of the sqlite checkpointer is a follow-up (needs a `(output_dir, question_index)` thread id and an `aget_state` probe in `is_run_completed`).
- 618 tests.
- Environment findings (not engine defects): KIT chat *and* embedding endpoints have been hanging since 2026-08-22 evening; `openrouter/elephant-alpha` in `config.toml` is retired (404); the local Virtuoso has no `http://sciqa.org/kg` graph (ORKG dump lives on the Hetzner host only), so SciQA parity must run there; the worktree venv needed `uv sync --extra rerank` for `reranker_enabled = true`.

## 15. Phase 5 plan as executed (2026-08-23)

KIT unavailable, so the parity run is engine-vs-engine at a fixed substitute model: KQAPro, 100 questions, seed 42, `openai/gpt-4.1-mini` via OpenRouter (chat + embeddings), judge `deepseek/deepseek-v4-pro`, concurrency 4, both engines from commit `80ffca9`. SciQA parity deferred to the Hetzner host. A KIT re-run at the reference model is still required before flipping the default engine.

## 16. Phase 5 result (2026-08-23): KQAPro parity PASSED at the substitute model

Evidence: `langgraph-rewrite-phase5/parity_kqapro_100.md` (+ manifests, questionnaire). 100 questions, seed 42, `openai/gpt-4.1-mini`: accuracy 0.58 vs 0.58 (9/9 discordant, McNemar p = 1.0), prompt tokens +2.4% on graph (bound: 10%), tool calls 7.07 vs 6.90, graph faster per question. Intervention injection counts match within model-path variance.

Still open before flipping `[agent].engine` default to `graph` and deleting the legacy loop (Phase 6):
- [ ] Re-run the same parity at the reference model `kit.gemma4-31b-it` once KIT is responsive (PRD §5 acceptance as written).
- [ ] SciQA parity on the Hetzner host (ORKG graph), 100 questions.
- [ ] Cosmetic: emit the two missing `_trace` lines on the graph path.
- [ ] Phase 6 proper: move the swap-and-diff helpers' logic into graph-owned code, make legacy `_route_autonomously` reuse the graph helpers (or delete legacy), remove dead deps (`openai-agents` only; `fastapi` and `uvicorn` are live again since the 2026-09-20 demo-line merge brought in `ama_kbqa/api/`), replace the retired `openrouter/elephant-alpha` in `config.toml`, rewrite `System/agent_framework.md`.
- [ ] **Graph engine lacks two demo-line features (found merging the demo line, 2026-09-20).** (1) Federated dispatch: `graph/orchestrator.py` only models Router mode, so `Orchestrator.ask()` sends a Federated instance to the legacy body even when `engine = "graph"`. (2) Cooperative cancellation: neither `graph/pipeline.py` nor `graph/orchestrator.py` has checkpoints, so `_ask_impl` and `Orchestrator.ask()` only honour a token that is already cancelled before dispatch. Both must be ported before the legacy loop can be deleted, or the React demo loses its Stop button and Federated mode. Pinned by `test_federated_instance_takes_legacy_body_even_with_graph_engine` and the two `test_cancelled_token_is_honoured_before_graph_*` tests.
- [ ] `_route_autonomously` now returns `Optional[List[str]]` (federated port). The duplicated `probe`/`select_agent` graph nodes still return a single name and still send the Router-mode `select_agent` schema, which is correct for Router mode but means the "keep both copies in sync" note above now covers only the Router half of that function.

# Generic KBQA Framework (`BaseKBQAAgent`)

Shared mechanics used by every concrete agent (`KQAProAgent`, `SciQAAgent`) and,
via delegation, the Orchestrator. This doc covers the framework layer only —
question-type strategies and tool catalogs live in the per-agent docs.

## Related Docs
- [Agent System (index)](agent_system.md) — map of all agent-system docs
- [KQAProAgent](kqapro_agent.md) — KQAPro-specific lifecycle, journal, tools
- [SciQAAgent](sciqa_agent.md) — SciQA-specific lifecycle, journal, tools
- [Orchestrator Routing](orchestrator_routing.md) — routing/delegation layer
- [Project Architecture](project_architecture.md) — full repo structure, config, CI

---

## Framework Files (`ama_kbqa/framework/`)

| File | Contents | Lines |
|------|----------|-------|
| `operations.py` | Abstract operation contract: `AtomicOperation`, `ATOMIC_OPERATIONS` (11 required + 3 optional), `CoverageReport`, `validate_bindings()`. Replaced the old `types.py` response-type module (June 2026 abstract-operation-contract ADR). | ~200 |
| `deterministic.py` | Shared LLM-free math core: `parse_numeric`, `NumericComparison`, `compare_numeric`; both servers' `VerifyNumericCondition` tools wrap it (byte-identical output). | ~100 |
| `base_agent.py` | `BaseKBQAAgent` ABC — full agent lifecycle, tool-calling loop, span instrumentation, synthesis funnel. | ~600 |
| `mcp_client.py` | Shared `MCPClient` class for MCP server communication. | ~120 |
| `config.py` | Configuration dataclasses (NamespaceConfig, KnowledgeGraphConfig). | ~220 |
| `state.py` | `JournalState` (Pydantic `BaseModel`, single source of truth with caps/helpers) and `JournalManager`. | ~340 |
| `text_tool_calls.py` | Text-mode tool-call shim for models that can't emit native function calls. | ~150 |
| `trace.py` | `TraceEvent` (OTel-shaped dataclass) + `TraceRecorder` (ContextVar nesting, async/sync spans, point-in-time events, JSONL export, `add_listener`/`remove_listener`). | ~200 |
| `adapters/base_adapter.py` | `BaseKGAdapter` ABC: `_create_config()` (abstract), `get_operation_bindings()` (abstract — maps abstract op name to concrete MCP tool name), `validate_operation_coverage()`, URI/SPARQL utilities. | ~340 |
| `adapters/kqapro_adapter.py` | KQAPro adapter: binds 11 required ops + optional `select_extreme`; sources `NS_*` / `SPARQL_PREFIXES` for `kqapro_server.py`. | ~150 |
| `adapters/sciqa_adapter.py` | SciQA adapter: binds 11 required ops (`count` → `AggregateComparisonValues` via `agg="count"`) + optional `aggregate`/`frequent_values`; includes `owl:` prefix (`sparql_cap: 10`). | ~200 |

**`BaseKBQAAgent` provides:**
- Abstract methods: `get_config()`, `get_mcp_server_path()`
- Template methods: `_get_system_prompt()`, `_classify_question()`, `_extract_entities()`, `_extract_exact_attribute_constraints()`, `_get_allowed_tools_for_qtype()`, `_get_denied_tool_names()`
- Concrete methods: `ask()`, `_run_tool_loop()`, `_detect_loops()`, `reset()`, `soft_reset()`, `close()`
- MCP client management; pre-agent hooks (classification, entity extraction) — **skipped on follow-up turns** in multiturn mode
- 6-7 layer loop detection (see per-agent docs for the exact layer count each agent exercises)
- Scratchpad-enforced tool-calling loop (forced reflection, truncation, journal refresh, concurrent tool execution — see below)
- Post-agent synthesis with hard-stop guards (see Synthesis Funnel below)
- Token and tool-call tracking; `self.recorder: TraceRecorder`; `self.journal_snapshots: list`
- `parent_recorder` + `parent_span_id` kwargs for sub-agent nesting under Orchestrator
- **Multiturn conversation:** `reset(keep_history=True)` preserves `self._messages` across turns; `_catalog_injected` flag prevents re-injecting the text-mode tool catalog on a reused stack. See `Decisions/multiturn-direct-agent-conversation.md`.

---

## Tool Gating: Qtype Filter + Denylist

`ask()` applies two independent gates to the tool list sent to the LLM, in order:

1. **Qtype filter** (`_get_allowed_tools_for_qtype`) — restricts to the tools relevant to the classified question type. Saves ~2-3k tokens/iteration. Skipped for multiturn follow-ups (full tool set kept). Base default returns `None` (no filtering); subclasses override.
2. **Denylist gate** (`_get_denied_tool_names`) — applied *even when the qtype filter allowed all tools*. Base default returns an empty set. Used to A/B-test or permanently retire a tool via an env-gated flag rather than a code change. **`SciQAAgent`** overrides this to gate `RunORKGSPARQL` off when `AMA_SCIQA_DISABLE_RAW_SPARQL` is truthy (`1`/`true`/`yes`/`on`) — see [SciQAAgent doc](sciqa_agent.md#env-gated-tool-denylist-ama_sciqa_disable_raw_sparql) for the rationale and A/B evidence.

Both gates log the before/after tool count via `self._trace(...)`.

---

## Synthesis Funnel: Always Exit Through Synthesis

As of the 2026-07-05 architecture audit (items B1/B2, `Decisions/architecture-audit-2026-07-05.md`), every terminal path in the tool loop funnels through `_run_synthesis` rather than returning a raw error string — the invariant is that a best-effort synthesized answer from a partial journal strictly dominates a guaranteed miss:

- **Max-iterations reached (B1):** previously returned `"Error: Agent reached maximum iteration limit."` directly. Now calls `_run_synthesis(query, qtype=qtype)` instead, mirroring the existing `max_tool_calls` path.
- **Malformed tool-call arguments:** a `JSONDecodeError` while parsing `tool_call.function.arguments` no longer silently substitutes `{}` and executes the tool with empty args (which used to burn an iteration on a phantom call, e.g. `FindNode` with no name). The parse error is now fed back to the model as a tool-result error message asking it to re-emit the call with valid JSON.
- **`GetJournalSummary` failure during synthesis (B2):** `_run_synthesis_impl` wraps the `GetJournalSummary` call in try/except. On failure it falls back to the most recent `self.journal_snapshots[-1]["state"]` (JSON-dumped), then to an empty string — never propagates the exception and discards a completed investigation.
- **Empty `choices` array from the synthesis LLM call:** some endpoints return zero choices under load or content filtering. Guarded explicitly (`if not response.choices: return ""`) instead of letting an `IndexError` crash the question at the finish line.
- **Synthesis LLM call exceptions:** previously re-raised (`raise`). Now caught and converted to `return ""`, so the caller's existing empty-answer fallback fires instead of the whole question erroring out.

Net effect: infra hiccups at or near the very end of a question no longer throw away tool-loop progress that was otherwise complete.

---

## Final Answer Cleanup

`BaseKBQAAgent._finalize_answer_text()` is applied to fast-path, synthesis-bypass, and synthesis answers. It strips leaked `<think>...</think>` blocks from user-facing output. For KQAPro `Verify` questions it also normalizes explicit/obvious verification statements to the benchmark `yes`/`no` contract — this is answer-shape enforcement, not an extra KG lookup or question-specific rule.

## Exact-Constraint Injection

`BaseKBQAAgent` exposes `_extract_exact_attribute_constraints(query)` as a KG-specific hook. The base implementation returns no constraints; `KQAProAgent` overrides it (see [kqapro_agent.md](kqapro_agent.md)).

## Classification JSON Parsing (`_extract_json_object`)

`BaseKBQAAgent._extract_json_object(content: str) -> Optional[Dict]` is a static helper used by both `_classify_question` and `SciQAAgent`'s equivalent. Handles LLM responses with reasoning prefixes or markdown fences around the JSON payload:

1. Strip `<think>...</think>` blocks via `re.sub(..., flags=re.DOTALL)` — minimax-m2.7 emits these even with `response_format=json_object`.
2. Strip markdown code fences.
3. `json.loads()` on the cleaned string (fast path).
4. Brace-balanced fallback: scan from the first `{`, count depth, extract the balanced block, parse.

Classifier `max_tokens` was bumped 300 → 1500 to give think-prefix models room to complete both the reasoning block and the JSON output. See `Decisions/classifier-think-prefix-fix.md`.

---

## Trace Instrumentation (`trace.py`)

Every `BaseKBQAAgent` instance owns `self.recorder: TraceRecorder` and `self.journal_snapshots: list`, reset on each `ask()` call. The Chat page reads them after `ask()` returns to populate the Trace Inspector and Graph View pages.

**TraceEvent** — OTel-compatible dataclass: `trace_id`, `span_id`, `parent_span_id`, `kind`, `name`, `start_time_unix_nano`, `end_time_unix_nano`, `duration_ms`, `status` ("ok"/"error"), `is_event` (bool), `attributes`, `payload`, `error`.

**TraceRecorder API:**

| API | Usage |
|-----|-------|
| `async with recorder.span(kind, name, ...)` | Open an async span; nested spans auto-parent via `ContextVar` |
| `with recorder.span_sync(kind, name, ...)` | Synchronous variant |
| `recorder.event(kind, name, ...)` | Point-in-time event (no duration) |
| `recorder.to_dicts()` | Serialise all events to list-of-dicts |
| `recorder.to_jsonl()` | JSONL string for export/persistence |

**ContextVar nesting:** propagates correctly across `await` boundaries — no manual parent_id threading needed. See `Decisions/trace-inspector-frontend-architecture.md` §Decision 2.

**Span placement — BaseKBQAAgent:**

| Kind | Location |
|------|----------|
| `agent_run` (span) | Root span wrapping `ask()` |
| `classify` (span) | `_classify_and_extract` |
| `fast_path` (span) | `_try_fast_path` |
| `llm_call` (span) | All three `_llm_call*` variants |
| `tool_call` (span) | `_execute_single_tool` — one span per tool. Concurrent calls in the same turn create child spans off the `ContextVar`-propagated parent, so the tree stays correct even under `asyncio.gather`. |
| `synthesis` (span) | `_run_synthesis` |
| `tool_loop_iter` / `journal_refresh` / `loop_detected` / `context_trim` / `intervention` (events) | Per-iteration / per-trigger point-in-time events |

**Span placement — Orchestrator:** `agent_run` (root, in `ask()`), `classify` (in `_route_autonomously`), `delegate` (one per sub-agent call — sub-agent spans nest as children since the Orchestrator shares its `recorder` before delegating). See [orchestrator_routing.md](orchestrator_routing.md).

**Journal Snapshots:** `self.journal_snapshots` holds `JournalState.model_dump()` dicts captured after calls to tools in `JOURNAL_MUTATING_TOOLS` (frozenset in `trace.py`) — only these tools write to `session_journal.*`. `GetJournalStateJSON()` is an LLM-hidden MCP tool on both servers (filtered from `list_tools`); the agent calls it directly and dedupes snapshots by dict equality.

---

## Text-Tool-Call Mode

Some models (e.g. `minimax-m2.7`) cannot emit OpenAI-structured `tool_calls` — they output prose or `<tool_call>` XML instead.

1. **Detection:** `text_tool_calls.needs_text_tool_calls(model_name)` — substring match, currently covers `minimax-m2.7`.
2. **System prompt injection (init):** `TEXT_TOOL_CALL_INSTRUCTION` + a plain-text tool catalog from `build_text_mode_tool_catalog`.
3. **API call:** `tools=None` — no native function-call schema sent.
4. **Response parsing:** if `message.tool_calls` is empty but content contains `<tool_call>` blocks, `parse_text_tool_calls` constructs synthetic OpenAI-shaped tool_calls.
5. **Tool results:** appended as `role=user` prose wrapped in `<tool_result name="X">...</tool_result>` (not `role=tool`).
6. **Truncated-block retry:** `has_truncated_tool_call(content)` detects an opener with no parseable closing block. On hit: broken turn persisted, corrective user message injected, loop continues (reuses `zero_tool_call_retry_max` budget). If that budget is exhausted with `total_tool_calls_made == 0`, the loop returns a hard error rather than accepting a final answer from prior knowledge.

See `Decisions/text-mode-tool-calls.md` and `Decisions/truncated-tool-call-retry.md`.

---

## MCP Client Pattern

**File:** `ama_kbqa/framework/mcp_client.py`

```python
class MCPClient:
    def __init__(self, server_path: str, agent_name: str): ...
    async def start(self): ...          # stdio_client + ClientSession
    async def list_tools(self) -> List[McpTool]: ...
    async def call_tool(self, name: str, args: Dict) -> str: ...
    async def close(self): ...
```

**Close-delay trim (2026-07 reconciliation, adjacent to but distinct from audit item C6a):** `close()` used to sleep 0.5s after `exit_stack.aclose()` plus an additional 0.2s cleanup delay — pure dead time paid on every close, notably the Orchestrator, which closes its probe server on every question. Trimmed to a single 0.05s yield; on stdio, `aclose()` already tears down the transport, so a short yield is enough to let the child process reap. **Note:** an MCP-client *instance dedup* (reusing one client across probe + delegate calls, audit item A1) was drafted on a since-superseded branch and did **not** land in this change — `MCPClient` instances are still created per use; do not assume dedup exists without re-checking `mcp_client.py` and the orchestrator's client construction sites. See `Decisions/branch-reconciliation-2026-07-18.md`.

---

## Token Tracking

```python
self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
```

Tracked at 5 call sites: `_classify_question()`, `_extract_entities()`, `_llm_call()` (main loop), `_llm_call_text_only()` (no-tools fallback), `_llm_call_synthesis()`. Not currently aggregated: token usage from postprocessing functions (LLM judge, answer selection, SPARQL synthesis).

## Tool Call Duration Tracking

```python
self.tool_call_durations: List[Dict[str, Any]] = []  # tool_name, duration_seconds, success, timestamp, error
```

`agent.get_tool_call_summary()` returns `total_calls`, `total_duration_seconds`, per-tool `tool_breakdown` (count/total/avg/success/failure), and the raw `calls` list. Aggregated across all questions in batch results (`statistics.tool_breakdown`).

---

## Synthesis Configuration

```toml
[synthesis]
synthesis_enabled = true          # false → skip synthesis, use agent's last message
synthesis_provider = "openrouter"
synthesis_model = "deepseek/deepseek-v3.2"
synthesis_temperature = 0.2
synthesis_max_tokens = 8000
synthesis_mode = "benchmark"      # "benchmark" (short exact-match) or "conversational"
```

- **`synthesis_enabled`** (`get_synthesis_enabled()`, default `True`): when `false`, `_run_tool_loop` returns the agent's own last assistant message directly (falls back to synthesis if the agent produced no final content, e.g. max-iteration abort). Toggle: Settings page → Synthesis Configuration.
- **`auto_inject_journal`** (`[agent]` section, `get_auto_inject_journal()`, default `True`): gates two injection sites — (1) periodic refresh every `journal_refresh_interval` iterations (default 5, append-only since commit `9bb1086`, superseded refreshes stubbed to placeholders during compaction, `_next_trim_trigger` armed with hysteresis so compaction/refresh don't alternate every iteration); (2) the post-`GetJournalSummary` "answer now" prompt. `GetJournalSummary` remains callable regardless of the flag. Toggle: Settings page → Agent Configuration.
- **How synthesis works:** `_run_synthesis` calls `GetJournalSummary`, builds a minimal 2-message conversation (`[system_prompt, synthesis_prompt(journal + query)]` — no conversation history, the primary token saving of 30-80k tokens/question), and re-prompts once if the result looks like a failure phrase but the journal actually has data (keyed off literal renderer strings: `"discovered values"`, `"verified facts"`, `"partial answer:"`, `"orkgr:"` — do not change these marker strings in `GetJournalSummary` without updating this heuristic).

---

## Reproducibility: LLM Sampling Seed

`config.get_chat_seed()` (`ama_kbqa/config.py`) reads the `AMA_LLM_SEED` env var (set by the benchmark CLI's `--seed` flag) or an optional `[llm] chat_seed` config key; returns `None` when neither is set. As of the 2026-06-15 fix, `--seed` actually threads through to `openai`-style `seed=` on every sampled LLM call: `_classify_question`, the main tool-loop call, and the text-only fallback in `base_agent.py`, plus the Orchestrator's probe/judge calls. Previously `--seed` only controlled dataset sampling (which questions get picked), not model sampling — so two "same-seed" runs could diverge in agent behavior even on the identical question set. The LLM judge is intentionally left unseeded (`temperature=0` makes seeding moot for the judge call). Full-dataset (`-n` covering the whole split) runs emit a warning since seeding stops mattering once every question is sampled anyway.

---

## Agent Reset Pattern (4 Variants)

1. **`reset()` — Full reset (default).** Resets message history (except system prompt), token/tool-call/loop-detection tracking, MCP connection (closed + reopened), recorder (new trace_id), journal_snapshots, `_catalog_injected`. Persists: LLM client config, system prompt.
2. **`reset(keep_history=True, keep_mcp_open=True)` — Multiturn continuation.** Used by `lifecycle_runner` on Chat UI follow-ups. Persists `self._messages` (full stack) and `_catalog_injected`; resets everything else per-turn (tracking, recorder, journal_snapshots). `keep_mcp_open=True` is required because the previous turn's MCP connection is bound to a now-closed asyncio event loop — the caller sets `agent.mcp = None` before calling reset, and `ask() → _init_mcp` rebuilds a fresh connection on the current loop. See `Decisions/multiturn-direct-agent-conversation.md`.
3. **`soft_reset()` — Batch processing.** Keeps MCP server running (avoids subprocess startup overhead + maintains warm Qdrant/Virtuoso connections). Resets message history, tracking, and journal (via `ManageJournal`).
4. **`close()` — Explicit MCP shutdown.** Call at the end of a batch run.

```python
agent = KQAProAgent()
try:
    for question in questions:
        answer = await agent.ask(question)
        await agent.soft_reset()
finally:
    await agent.close()
```

---

## Console Tracing

Color-coded debug output, one line per event: `[HH:MM:SS] [agent_name] -> msg`. BLUE = general, GREEN = tool calls/success, RED = errors, YELLOW = warnings/tool calls, CYAN = state changes/reasoning.

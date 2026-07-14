# WikiKGQA Agent System

## Related Docs
- [Decisions/wikikgqa-2026-adaptation.md](../Decisions/wikikgqa-2026-adaptation.md) — why this subsystem exists: SPARQL-generate→execute head, no local Wikidata embedding, challenge endpoint backend, with-mentions-first scope
- [Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md](../Decisions/wikikgqa-tool-budget-and-resilience-2026-06-30.md) — why the tool-call budget became the binding limit and why provider-error retries were added
- [Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md](../Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md) — why synthesis context defaults to minimal (journal-only), not full transcript
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](../Decisions/wikikgqa-commit-time-recovery-2026-07-03.md) — non-empty-prior recovery at commit time, ASK self-consistency voting, run manifests; open framework-level gaps still discarding validated queries
- [Decisions/wikikgqa-conventions-default-2026-07-03.md](../Decisions/wikikgqa-conventions-default-2026-07-03.md) — why `conventions` defaults to minimal (R1-R6): held-out seed-99 A/B found no benefit from the extended R7-R10 rules
- [Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](../Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md) — commit-time class-closure expansion (rule 10 applied mechanically) and NOW() pinned to the frozen gold reference instant
- [System/agent_system.md](./agent_system.md) — the shared `BaseKBQAAgent` tool loop, journal, and synthesis machinery this subsystem reuses
- [System/project_architecture.md](./project_architecture.md) — overall repo structure, KQAPro/SciQA reference pattern

---

## Overview

WikiKGQA is the AMA-KBQA system's adaptation for the ISWC 2026 WikiKGQA challenge:
answer natural-language questions over Wikidata by **generating and executing a
SPARQL query**, not by synthesizing a natural-language answer (QALD F1 requires
exact `wd:Q…`/literal answer sets — see the linked ADR for the full rationale).

It reuses the existing `BaseKBQAAgent` tool loop (same framework as KQAPro/SciQA)
but points it at a live Wikidata SPARQL endpoint instead of a local Qdrant index,
and overrides the synthesis step to emit SPARQL instead of prose.

```
ama_kbqa/
  agents/wikidata_agent/
    agent.py       # WikidataAgent(BaseKBQAAgent)
    prompts.py      # SYSTEM_PROMPT, EXTENDED_CONVENTIONS (R7-R10), synthesis prompts
  server/
    wikidata_server.py   # MCP server: live-SPARQL exploration tools
  wikikgqa/
    dataset.py        # WikiKGQAQuestion / QALD-JSON loader
    endpoint.py        # resolve_endpoint(), execute(), WIKIDATA_PREFIXES
    generator.py        # MentionSparqlGenerator, AgentSparqlGenerator
    prompts.py          # baseline generator's prompt + repair templates
    scoring.py           # Macro QALD F1 scorer
    submission.py        # QALD-JSON submission writer
    benchmark.py          # CLI runner (--generator mention|agent, --conventions full|minimal)
    ANSWER_CONVENTIONS.md # R1-R10 gold-answer-shape conventions (human-readable)
```

Tests: `tests/wikikgqa/` plus the shared `tests/framework/` and `tests/server/`
suites (the WikidataAgent and its server are exercised by the same framework
tests as KQAPro/SciQA). As of 2026-07-14: 328 tests passing (up from 270 on
2026-07-03, up from 199 on 2026-06-30).

---

## Two generation strategies

`ama_kbqa/wikikgqa/benchmark.py --generator {mention,agent}` selects between:

| | `MentionSparqlGenerator` (`mention`, default) | `AgentSparqlGenerator` (`agent`) |
|---|---|---|
| Use case | With-mentions track: QIDs/PIDs already resolved | Questions needing graph exploration to find properties/paths not in the mentions |
| Mechanism | Single LLM call → execute → repair loop (≤`max_repairs`, default 2) | Full `BaseKBQAAgent` tool loop via `WikidataAgent` |
| Repair triggers | Execution error (`REPAIR_EXECUTION_ERROR`) or empty result set (`REPAIR_EMPTY_RESULT`) — both in `ama_kbqa/wikikgqa/prompts.py` | Tool-call budget (see below); agent self-corrects mid-loop using `RunSPARQL` feedback |
| Timeout | `timeout=120s` per `execute()` call | `agent_timeout=280s` wall-clock **safety net only** (see Decision below) — the binding limit is the tool-call budget |
| Cost | One blind LLM call (cheap) | Fresh `WikidataAgent` + fresh MCP server subprocess per question (journal resets every question) |

The empty-result repair instruction (`REPAIR_EMPTY_RESULT`) explicitly tells the
model to reconsider triple direction, `wdt:`/`p:`/`ps:`/`pq:` choice, and whether
a qualifier is needed — this is the "evidence-based verify-on-empty" repair
referenced in earlier work; it lives in `ama_kbqa/wikikgqa/prompts.py` and
predates the 2026-06-30 budget/resilience changes below.

**Non-empty-prior recovery (`AgentSparqlGenerator`, since 2026-07-03).**
Every gold answer in this benchmark is non-empty, so a committed query that
errors or returns 0 rows is known-wrong, not merely suspicious.
`_generate_once` exploits this: if the committed query fails or is empty
**and** the journal holds a different validated query from exploration, that
alternate query is executed too; if it has rows, it replaces the committed
result. This depends on `strip_sparql` correctly returning `""` for
non-SPARQL text (see below) — before that fix, agent-loop error strings like
`"Error: Agent reached maximum iteration limit."` were executed as literal
SPARQL queries, which both produced a guaranteed failure *and* skipped the
recovery branch (it only fired on genuinely empty output). See
[Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](../Decisions/wikikgqa-commit-time-recovery-2026-07-03.md).

**ASK self-consistency voting.** `AgentSparqlGenerator(ask_votes=N)` /
`benchmark.py --ask-votes N` (default `1` = off). When the committed query
executes to a boolean (`ASK`) result and `ask_votes > 1`, the full generation
re-runs up to `ask_votes` times and majority-votes the booleans; ties keep
the first run. SELECT-result questions are never re-run. Targets the ASK
slice (~7% of questions), identified as the noisier/less-deterministic class.

**Class-closure expansion (`AgentSparqlGenerator`, since 2026-07-14).**
`ANSWER_CONVENTIONS.md` rule 10 — gold uses `wdt:P31/wdt:P279*` transitive
class-membership closure, not bare `wdt:P31` — used to be a prompt rule
(part of `EXTENDED_CONVENTIONS`); the 2026-07-03 conventions A/B found the
R7-R10 block gave no net held-out benefit, but rule 10 specifically is
demonstrably needed on some questions (q110, q111). It's now applied
mechanically at commit time instead of via prompt: after the query
commits (post non-empty-prior recovery above), `_next_closure_escalation`
widens any bare `wdt:P31` pattern one ladder step at a time
(`wdt:P31` → `wdt:P31/wdt:P279*` → `wdt:P31*/wdt:P279*`), executing each
candidate and adopting it only when its answer set is a **strict superset**
of the currently committed one. Equal-set steps pass through the ladder
without adoption or stopping the loop (verified live on q110: only the
ceiling level reaches gold's answer set). `ASK`/aggregate/`GROUP
BY`/`HAVING`/literal-valued/0-row queries are excluded outright — widening
a scalar isn't something the strict-superset check can validate. Constructor
flag `AgentSparqlGenerator(closure_expansion=True)` (default on),
`benchmark.py --no-closure-expansion` to disable. See
[Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](../Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md).

**NOW() pinning (`ama_kbqa/wikikgqa/endpoint.py`, since 2026-07-14).** Gold
answers were computed once against a frozen KG snapshot at a reference
instant reverse-engineered from gold data as **2026-04-08**
(`endpoint.REFERENCE_TIME`). `execute()` rewrites every outgoing SPARQL
`NOW()` function call to that frozen instant
(`"2026-04-08T00:00:00Z"^^xsd:dateTime`) before sending the query —
string-literal- and IRI-safe, case-insensitive, whitespace-tolerant.
Default on; `WIKIKGQA_PIN_NOW=0` env var / `benchmark.py --no-pin-now` to
opt out. Because both the generator's commit path and the agent's
`RunSPARQL` MCP tool (`wikidata_server.py`) funnel through this same
`execute()`, pinning applies consistently to agent exploration and
commit-time execution alike — the model never sees one NOW() while
exploring and a different one at commit. See
[Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](../Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md).

---

## WikidataAgent (`ama_kbqa/agents/wikidata_agent/agent.py`)

A `BaseKBQAAgent` subclass with several deliberate departures from the
KQAPro/SciQA agents:

- **No fast path, no fewshots.** `enable_fast_path: False` and
  `use_fewshot=False` — there is no fixed schema to special-case against; the
  agent always explores.
- **Classification is bypassed entirely.** `_classify_question()` does not call
  the LLM; it regex-extracts `Q\d+`/`P\d+` ids straight from the question text
  and always returns `question_type="Query"`. There is only one question type
  here (generate a query), so the classifier LLM call would be pure overhead.
- **`get_config().domain_settings`** (current values):

  | Key | Value | Purpose |
  |---|---|---|
  | `max_tool_calls` | **20** | Binding limit — see Decision below |
  | `max_iterations` | **24** | Loop-iteration ceiling (slightly above `max_tool_calls` to allow non-tool turns) |
  | `sparql_cap` | 20 | Per-question cap on `RunSPARQL` calls specifically |
  | `find_resource_cap` | 12 | (reserved; no `FindResource`-equivalent tool yet exists in `wikidata_server.py`) |
  | `context_limit` | 100000 | Token budget before truncation kicks in |

- **`conventions: "full" | "minimal"`** constructor parameter, threaded through
  to `benchmark.py --conventions`. `"full"` appends `EXTENDED_CONVENTIONS` (R7-R10
  modeling rules) to `SYSTEM_PROMPT`; `"minimal"` uses only `SYSTEM_PROMPT`
  (R1-R6). **Default: `"minimal"`** (since 2026-07-03) — the held-out seed-99
  `rand-50` A/B found no benefit from R7-R10 (0.7311 minimal vs 0.6833 full
  Macro F1, noise-dominated), so the cheaper, lower-overfitting-risk minimal
  set is the default and `"full"` is opt-in. See
  [Decisions/wikikgqa-conventions-default-2026-07-03.md](../Decisions/wikikgqa-conventions-default-2026-07-03.md).
  See `ama_kbqa/wikikgqa/ANSWER_CONVENTIONS.md` for what R1-R10 actually encode
  (entity-URI-not-label answers, `ASK` for yes/no, superlative `ORDER BY`,
  age-via-`NOW()`, `COUNT` default for "how many", sovereign-state QID, the
  normalized-quantity statement path, "currently/still" exclusion filters,
  instance-of completeness).
- **Per-instance model/provider override** — `config.toml`'s `chat_model` is
  treated as stale for this subsystem; `model=`/`provider=` constructor args let
  the benchmark runner sweep multiple models without editing config, and
  `_text_tool_call_mode` is re-derived from the actual override.

---

## `wikidata_server.py` MCP tools

Exploration tools: `RunSPARQL`, `GetEntityProperties`, `GetIncomingRelations`,
`GetOutgoingRelations`, `GetRelationBetween`, `GetPropertyInfo`, `GetLabels`,
`ManageJournal` (plus the LLM-hidden `GetJournalStateJSON`, framework-standard).

Two mechanisms specific to this server, both added 2026-06-30 to address
KIT-model failure modes (see ADR for the failure analysis):

**Per-tool SPARQL timeout.** `RunSPARQL` runs with
`_TOOL_SPARQL_TIMEOUT = 30` seconds. On timeout it returns a *descriptive*
message instead of hanging or raising a generic error:

```
TIMEOUT: this query did not return within 30s. It is probably too expensive
or matches too many rows. Make it MORE SPECIFIC: add a tighter type
constraint or FILTER, restrict the entity set, or add a LIMIT for
inspection, then retry. Do not re-run the same broad query.
```

This both bounds per-call cost and steers the model away from blindly
re-running the same broad query (the dominant pattern in the over-exploration
failure mode).

**Live tool-call budget counter (`_budgeted` decorator).** A module-level
`_tool_call_count` (resets per question, since the MCP server process is
spawned fresh per question by `AgentSparqlGenerator`) is incremented on every
call to a `@_budgeted`-wrapped tool. The wrapper appends a tag to the tool's
string result:

```
[tool call N/20]
```

escalating to a "near budget" warning once `N >= _TOOL_BUDGET - 4` (i.e. from
call 16 onward):

```
[tool call 17/20] near budget: stop exploring and submit your best
validated query.
```

`_TOOL_BUDGET = 20` matches `WikidataAgent`'s `max_tool_calls` so the visible
counter lines up with when the framework forces synthesis. Implementation
notes:
- Uses `functools.wraps` so FastMCP's signature introspection (it derives tool
  parameter schemas from the wrapped function's signature) still sees the
  original function, not a generic `*args, **kwargs` wrapper.
- Skips tagging when the result looks like JSON (`out.lstrip()[:1] in "{["`),
  so `GetJournalStateJSON`'s machine-readable output is not corrupted by an
  appended string tag.
- Applied to `RunSPARQL`, `GetEntityProperties`, `GetIncomingRelations`,
  `GetOutgoingRelations`, `GetRelationBetween`, `GetPropertyInfo`, `GetLabels`,
  `ManageJournal`.

---

## Prompt budget-awareness

`ama_kbqa/agents/wikidata_agent/prompts.py SYSTEM_PROMPT` states the budget
explicitly so the model plans around it rather than discovering it only via
the live counter tag:

```
BUDGET: you have about 20 tool calls per question. Spend them on DISCOVERY,
then commit once a query works; don't polish. On TIMEOUT, narrow the query.
```

---

## Tool-call budget as the binding limit (framework mechanism, reused here)

The shared framework (`ama_kbqa/framework/base_agent.py`, ~line 1483-1498)
already had a "force synthesis at `max_tool_calls`" mechanism: once
`total_tool_calls_made >= max_tool_calls`, the loop logs an `intervention`
event and calls `_run_synthesis(query, qtype=qtype)` immediately, regardless of
where the model was in its reasoning. WikiKGQA is the first subsystem to rely
on this as the *primary* stopping condition rather than a backstop — see the
ADR for why (over-exploration on the larger KIT model was hitting the 180s
wall-clock cancel instead, which emitted **no query at all**). With the budget
as the binding limit, the framework now always emits the agent's best
validated query at call 20, and the generator's `agent_timeout` (280s) is only
a rare safety net for a genuinely stuck run.

---

## Shared framework resilience additions (2026-06-30, `base_agent.py`)

Two robustness changes landed in `BaseKBQAAgent` itself, so they apply to
**every** agent (KQAPro, SciQA, WikidataAgent), not just WikiKGQA — motivated
by WikiKGQA's exposure to transient KIT proxy failures but generally useful:

1. **Transient LLM-call retry.** `_llm_call`'s `chat.completions.create` is
   wrapped in up to 3 attempts with linear backoff (`1.5 * attempt` seconds)
   when the exception message matches a transient-error pattern: `"server
   connection error"`, `"open webui"`, `"connection error"`, `"timeout"` /
   `"timed out"`, `"502"`/`"503"`/`"504"`, `"overloaded"`, `"temporarily
   unavailable"`, `"429"`/`"rate limit"`. This directly targets KIT's "Open
   WebUI: Server Connection Error" (surfaced as an HTTP 400) that previously
   aborted the whole question. Errors that don't match (e.g. malformed
   request, auth failure) re-raise immediately on the first attempt —
   deterministic errors are not retried.
2. **Empty-response guard.** Some providers (OpenRouter/Gemma observed)
   occasionally return a response object with no `choices` — typically right
   after rejecting a malformed tool call. Previously this crashed on
   `response.choices[0]`. Now the loop tracks `empty_response_count`; on an
   empty response it nudges the model with a corrective user message ("call a
   tool with valid JSON arguments... or give your final answer") and retries,
   breaking out to synthesis after 3 consecutive empties.

---

## Generator config knobs (`ama_kbqa/wikikgqa/generator.py`)

| Knob | Value | Note |
|---|---|---|
| `AgentSparqlGenerator.agent_timeout` | **280s** (was 180s) | Wall-clock safety net; rarely fires now that the tool-call budget is the binding limit |
| `AgentSparqlGenerator.ask_votes` / `benchmark.py --ask-votes` | 1 (off) | ASK self-consistency vote count; see above |
| `WikidataAgent._synthesis_full_context()` / `WIKIKGQA_FULL_SYNTHESIS` / `benchmark.py --full-synthesis` | **minimal (off)** by default | Full exploration-transcript context for synthesis vs journal-only; A/B found no benefit (0.781 vs 0.807 F1) — see [Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md](../Decisions/wikikgqa-synthesis-context-ab-2026-07-02.md) |
| `WikidataAgent(conventions=)` / `AgentSparqlGenerator(conventions=)` / `benchmark.py --conventions` | **`"minimal"`** (was `"full"`) since 2026-07-03 | R1-R6 vs R7-R10 extended modeling rules; held-out seed-99 A/B found no benefit from R7-R10 (0.7311 vs 0.6833 F1) — see [Decisions/wikikgqa-conventions-default-2026-07-03.md](../Decisions/wikikgqa-conventions-default-2026-07-03.md) |
| `AgentSparqlGenerator.closure_expansion` / `benchmark.py --no-closure-expansion` | **on** by default, since 2026-07-14 | Rule-10 class-membership closure applied mechanically at commit time, strict-superset-gated — see above and [Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](../Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md) |
| `endpoint.execute(pin_now=)` / `WIKIKGQA_PIN_NOW` env / `benchmark.py --no-pin-now` | **on** by default, since 2026-07-14 | Rewrites outgoing `NOW()` to the frozen gold reference instant `2026-04-08T00:00:00Z` — see above and the same ADR |
| `MentionSparqlGenerator.timeout` | 120s | Per-`execute()` call for the blind single-shot generator |
| `MentionSparqlGenerator.max_repairs` | 2 | Execution-error / empty-result repair attempts |

`benchmark.py` also writes `run_manifest.json` into each run's output
directory (git commit, ISO timestamp, resolved SPARQL endpoint, full parsed
CLI args) so a `summary.json` can always be traced back to the code version
and settings that produced it. Added 2026-07-03 — see
[Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](../Decisions/wikikgqa-commit-time-recovery-2026-07-03.md).

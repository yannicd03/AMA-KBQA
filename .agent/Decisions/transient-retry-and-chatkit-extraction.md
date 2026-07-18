# Transient LLM Retry Wiring + ChatKIT Package Extraction

**Date:** 2026-07-18
**Branches:** `dev` (this repo) and `wikikgqa-2026` (sibling repo, same arc)
**Status:** Shipped, CI green at every step

## Related Docs
- [Agent Framework](../System/agent_framework.md) — `_create_with_retry` / `TransientRetry` integration detail
- [Project Architecture](../System/project_architecture.md) — Tech Stack / dependency listing
- [Orchestrator Routing](../System/orchestrator_routing.md) — routing/LLM-fallback call sites now wrapped
- [Architecture Audit 2026-07-05](architecture-audit-2026-07-05.md) — prior audit that did not touch retry behavior; this ADR is unrelated in scope but touches some of the same files

## Context

KIT (the KIT AI Toolbox LLM endpoint) has an intermittent failure mode: it returns an
`"Open WebUI: Server Connection Error"` message inside a *non-5xx* HTTP response. The
OpenAI SDK's own `max_retries` only triggers on 5xx/connection-level failures, so this
error was silently passed through as if it were a valid completion — the SDK's retry
logic never saw it as retryable. Prior incident context: `Decisions/kit-outage-watchdog.md`
(structured httpx timeouts + benchmark-level abort) and `Decisions/root-cause-tool-generalization-2026-05-14.md`
addressed adjacent KIT-outage symptoms but not this specific message-marker failure mode.

`ama_kbqa/llm/` was originally built directly in this repo (and, in parallel, in the
sibling `wikikgqa-2026` repo) to add message-marker detection with stepped backoff on
top of the raw OpenAI SDK. Once both repos needed the same retry core, and once the
separate ORCA project (LangGraph-based, different KBQA system) also needed a hardened
KIT client, the vendored code became a duplication liability — three codebases
independently maintaining the same transient-detection regexes and backoff logic.

## Decision

**1. Port `ama_kbqa/llm/` verbatim onto `dev`** (commit `bb08c55`) — `TransientRetry`
(stepped, self-resetting backoff) + a raw-SDK KIT client + a lazy `ChatKIT` class,
ported as-is from `wikikgqa-2026` so both repos started from an identical baseline
before further changes.

**2. Wire `TransientRetry` into every agent LLM call path** (commit `1d626b6`):
- `BaseKBQAAgent._create_with_retry` (`ama_kbqa/framework/base_agent.py:2256`) — new
  helper wrapping `client.chat.completions.create(**call_params)` in `self._retry.run(...)`,
  with per-attempt backoff trace logging (`_log` callback → `self._trace(...)`).
  Used by `_llm_call`, `_llm_call_text_only`, `_llm_call_synthesis`, and
  `_classify_question` (base implementation).
- `SciQAAgent._classify_question` override (`ama_kbqa/agents/sciqa_agent/agent.py:210`,
  call at `:237`) — previously called `client.chat.completions.create` directly,
  bypassing retry; now routes through `self._create_with_retry(...)`.
- `Orchestrator` (`ama_kbqa/agents/orchestrator_agent/agent.py`) — gained its own
  `_create_with_retry` (`:186`) and `self._retry = TransientRetry()` (`:144`); wraps
  both the routing decision call (`:346`) and the LLM-fallback call (`:505`). The
  Orchestrator previously had **no** retry coverage at all.
- Wrapped clients are constructed with the new optional `max_retries` parameter on
  `get_chat_client()` / `get_synthesis_client()` / `_create_client()` (`ama_kbqa/config.py:59,238,425`):
  pass `max_retries=0` so the SDK doesn't also retry underneath `TransientRetry` (that
  would double the backoff and, per `chatkit.raw`'s own rationale, still miss the
  non-5xx marker). Callers that don't wrap their client keep the SDK default of 3.

**Deliberate deviations from the `wikikgqa-2026` reference implementation**, made because
this repo's call graph differs:
- Classification (`_classify_question`) **is** wrapped here — `wikikgqa-2026` doesn't have
  the same classify-then-loop split in exactly this shape.
- The config-level `max_retries=0`/double-retry gap is closed on `dev` as part of this
  change (both `get_chat_client` and `get_synthesis_client` calls that feed a
  `TransientRetry`-wrapped path pass `max_retries=0`).
- The Orchestrator is newly covered — it had no retry wrapping before this change.
- `ama_kbqa/server/orchestrator_server.py`'s `extract_semantics` (`:124`) is **deliberately
  left on plain SDK retries** — it runs in a separate MCP subprocess, not the
  agent's own retry-wrapped client, and wrapping it would require threading
  `TransientRetry` across a process boundary for a low-value call site.

Result: 22 new tests, suite at 412.

**3. Extract the retry core into a standalone package: `github.com/yannicd03/ChatKIT`**
(external repo, commit `c9fcebc`, released as `a4a1702` = tag `v1.0.0`). Restructured to:
- A langchain-free core (`constants.py`, `retry.py`, `raw.py`) with only `openai` + `httpx`
  as dependencies — this is what AMA-KBQA and `wikikgqa-2026` consume.
- A `[langchain]` extra exposing the hardened LangChain surface (`get_kit_model`, etc.),
  via a lazy `__getattr__` that raises a helpful `ImportError` if the extra isn't
  installed and the LangChain surface is touched. This is what ORCA (a separate
  LangGraph-based project) consumes.
- The ORCA-compatible API is unchanged by the restructure. **ORCA itself remains pinned
  to `chatkit` `v0.1.0`** and must declare `chatkit[langchain]` explicitly when it
  upgrades past that pin — it has not been touched as part of this arc.

**4. Delete the vendored code, depend on the package** (commit `938ef2b` on `dev`,
`d502da8` on `wikikgqa-2026`) — `ama_kbqa/llm/` removed; imports switched to
`from chatkit import ...`; `pyproject.toml` dependency added as a git dependency:
```toml
chatkit = { git = "https://github.com/yannicd03/ChatKIT.git", rev = "..." }
```
Tests that had duplicated the vendored implementation's own unit tests were replaced
with a slim import-surface test that also asserts `langchain` is **not** importable in
this environment — a regression guard against the `[langchain]` extra sneaking into
AMA-KBQA's dependency tree (AMA-KBQA has no LangChain dependency and should not gain one
via a loose chatkit version bump). `dev` suite: 407. `wikikgqa-2026` suite: 490.

**5. Pin to the release tag** (commit `360ae26` on `dev`, `b51493f` on `wikikgqa-2026`) —
moved the git dependency's `rev` from the working commit hash to tag `v1.0.0`.

## Consequences

- A single published package (`chatkit`) now serves two different consumer shapes:
  ORCA's LangGraph surface (via the `[langchain]` extra) and AMA-KBQA / `wikikgqa-2026`'s
  raw-SDK surface (core only, no LangChain dependency pulled in).
- `BaseKBQAAgent`, `SciQAAgent`, and `Orchestrator` each own one `TransientRetry` instance
  (`self._retry`) — backoff state is per-agent-instance and self-resetting, not shared
  across concurrent agents or across questions within a batch run (a fresh agent/`reset()`
  gets a fresh ramp).
- `orchestrator_server.py`'s `extract_semantics` is a known, accepted gap in retry
  coverage — re-evaluate only if KIT transient failures are observed to concentrate in
  that MCP subprocess call path specifically.
- Any future AMA-KBQA dependency bump of `chatkit` must NOT move to a version that made
  `[langchain]` a default (non-extra) dependency without re-checking the import-surface
  test still passes — that test is the one thing standing between "core only" and an
  accidental LangChain pull-in.

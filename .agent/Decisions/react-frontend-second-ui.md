---
status: accepted
date: 2026-09-16
summary: React SPA (web/) + FastAPI/SSE backend (ama_kbqa/api/) shipped as an additive second demo UI alongside Streamlit, which stays default and behaviorally unchanged; the API reuses the Streamlit chat page's own helpers directly.
relates: [trace-inspector-frontend-architecture]
affects: [amakbqa/task/langgraph-rewrite]
---

# ADR: React Demo Frontend — Additive Second UI, Not a Migration

**Status:** Accepted (shipped 2026-09-16, commits `852d2ba` FastAPI backend, `095f79f` React app, `6b64a71` provider-aware model handling, `07afe15` tracked `web/src/lib` + duration formatting, on branch `demo-v2-int`). Merged forward: `demo-public` fast-forwarded to `07afe15`; `demo-booth` merge `5a71267`; `demo-llamacpp` merge `674bf6e`.

**Narrows:** [trace-inspector-frontend-architecture.md](./trace-inspector-frontend-architecture.md) Decision 1 (2026-05-09, "Stay on Streamlit — reject React + FastAPI migration"). Decisions 2-6 of that ADR (`TraceRecorder` ContextVar nesting, mutating-tool-only journal snapshots, shared-recorder sub-agent nesting, vis-network rendering, tool-loop-iteration-as-event) are untouched and are reused as-is by the code described here — this ADR is scoped to Decision 1 only.

## Related Docs
- [Decisions/trace-inspector-frontend-architecture.md](./trace-inspector-frontend-architecture.md) — the ADR this narrows
- [Decisions/live-trace-and-chat-unification.md](./live-trace-and-chat-unification.md) — the listener/queue live-update design this API's consumer loop reuses
- [System/demo_react_frontend.md](../System/demo_react_frontend.md) — architecture, endpoints, SSE shapes for the code this ADR governs
- [SOP/running_react_demo_locally.md](../SOP/running_react_demo_locally.md) — how to run it
- `Tasks/active/langgraph-rewrite.md` — **not present in this worktree** (lives untracked at `/home/yannic/code/AMAKBQA/.agent/Tasks/active/langgraph-rewrite.md` on branch `feat/exact-lookup-tool`'s checkout, content authored against branch `feat/langgraph-rewrite`); see "Cross-branch conflict" below

---

## Context

Decision 1 of `trace-inspector-frontend-architecture.md` rejected a React + FastAPI migration for the *internal trace/graph inspector tool*, on the grounds that the tool is SSH-tunnelled, single-user, and the migration cost (10-20x) bought nothing a Streamlit page couldn't already do. Its stated threshold to revisit was "multi-user auth requirement, real-time collaboration need, or URL-routable deep-linking."

The trigger this time is different in kind: a **booth/public demo audience** — a use case Streamlit was never asked to serve well (SSH-tunnel-only deployment, a UI aesthetic tied to Streamlit's widget chrome, and a single global model/temperature setting that a booth visitor should not be able to see mid-change). That is closer to "multi-user" than the original threshold anticipated, but the response is not a migration.

## Decision

**Streamlit stays the default demo UI and its code path is unchanged.** No Streamlit page, session-state pattern, or route was touched to ship this.

**React is an additional frontend** (`web/`, a separate Vite + TypeScript app, served by its own nginx container) for a booth/public audience, running **next to** Streamlit, not replacing it.

**The API (`ama_kbqa/api/`, FastAPI) is a thin reuse layer**, not a reimplementation. It calls the exact same helpers the Streamlit chat page (`ama_kbqa/frontend/chat.py` and friends) already calls:

| Helper | Reused for |
|---|---|
| `frontend.utils.agent_factory` | agent construction, `AGENT_INFO`, `ORCHESTRATOR_MODES`, `is_orchestrator` |
| `frontend.utils.lifecycle_runner` (`LiveLifecycleState`, `start_run`, `drain_into`, `live_graph_snapshot`, `reconstruct_orchestrator`) | the exact thread+queue live-run mechanism Streamlit's `@st.fragment(run_every=0.4)` polls |
| `frontend.utils.lifecycle_svg` + `orchestrator_svg` | server-rendered SVG strings (same figures the Streamlit page embeds via `st.components.v1.html`) |
| `frontend.utils.live_graph_data` (`to_vis_payload`, `graph_from_trace`, `answer_node_ids`, `apply_caps`) | the same "Explored subgraph" data shape from `live-graph-two-source-subgraph.md` |
| `frontend.utils.chat_controls` | model list, `apply_chat_settings`, pricing captions |
| `frontend.utils.styling.ansi_to_html` | the same live-log rendering |
| `frontend.utils.trace_render.summarise` | the same trace-summary shape as the (removed) Trace Inspector page |
| `pricing` | cost estimation |

Because both frontends run the same helper functions against the same agent classes, the `TraceRecorder`/journal/trace instrumentation this ADR's parent decided (Decisions 2-6) needed zero changes to serve a second renderer — exactly the "renderer-agnostic" property that original ADR banked on.

## Why this isn't the migration Decision 1 rejected

- Decision 1 was about replacing the *existing* internal tool. This ships a *new*, narrower surface (one chat flow: pick agent/model, ask, watch it run, see the answer) for a different audience, while the internal tool (whatever Streamlit pages still serve trace/graph inspection) is untouched.
- The FastAPI layer is deliberately not a general backend: it has 7 routes (see `System/demo_react_frontend.md`), all of them thin translations of an existing Streamlit code path into JSON/SSE. It does not reimplement agent logic, routing, or instrumentation.
- The estimated cost that made migration a bad trade in May (10-20x for two Streamlit pages) does not apply to "add one narrow booth-facing view on top of code that already exists" — this shipped in one day of subagent-delegated implementation (`852d2ba` → `07afe15`) plus same-day multi-branch portability work.

## Trade-offs (recorded honestly, not glossed over)

**1. Two UIs to keep in sync.** Every future feature (the next `live-graph-panel.md`-style addition, a new agent, a picker change) now has two implementations to maintain: a Streamlit one and a React/FastAPI one. There is no shared component layer — Streamlit renders server-side HTML/iframes, React renders client-side from JSON — so "port it to both" is manual work each time, not a shared library update. This is the direct, ongoing cost of the additive approach; it was accepted for this feature, not eliminated by it.

**2. The process-global model/temperature override now serves multiple browser sessions concurrently.** `chat_controls.apply_chat_settings` mutates process-global config/env (this predates the API — Streamlit had the same limitation, but for one tab at a time behind an SSH tunnel). `RunManager._build_lock` (`ama_kbqa/api/runs.py`) serializes the override-then-build sequence so two concurrent `POST /api/runs` calls cannot interleave their `apply_chat_settings` calls, but a run already streaming for session A is not insulated from session B changing the global model/temperature for *future* agent builds mid-flight — `runs.py`'s own comment on `RunManager` calls this out as "a known limitation (same as Streamlit)." Under SSH-tunnel single-user access this was cosmetic; under a booth with concurrent visitors it is a real cross-session bleed risk, accepted rather than solved by this ADR. A fix (per-session config override instead of process-global) is out of scope here.

**3. In-memory run state forces exactly one uvicorn worker.** `RunManager.sessions` and `RunManager.runs` (and every `Run`'s live agent, queue, and `LiveLifecycleState`) live in the API process's memory — there is no external store. `ama_kbqa/cli.py`'s `api()` command hard-codes `uvicorn.run(..., workers=1, ...)` and `docker-compose.yml`'s `api` service comment repeats the constraint. This means the API cannot scale horizontally or survive a worker restart mid-run without a redesign (e.g. Redis-backed run/session state, or moving the live-lifecycle bridge to a message broker). Acceptable for a booth/small-public-demo load profile; would need revisiting before any larger-scale deployment.

**4. Cross-branch conflict with `Tasks/active/langgraph-rewrite.md`.** That PRD (branch `feat/langgraph-rewrite`, a **separate worktree/branch not merged into `demo-v2-int`**) currently states, under "Dependencies": *"Remove: `openai-agents`, `fastapi`, `uvicorn` (declared, never imported)."* That line is now **false on `demo-v2-int`**: `fastapi` and `uvicorn` are load-bearing runtime dependencies of `ama_kbqa/api/`, exercised by `ama-kbqa-api` (the `api` compose service) and the React frontend's entire request path. The same PRD's §4.6 also states Phase 4 "switches [`lifecycle_runner.py`] from thread+queue to `graph.astream(stream_mode=["updates","custom"])`" — i.e. it plans to delete the exact thread+queue bridge (`LiveLifecycleState`, `start_run`, `drain_into`) that `ama_kbqa/api/runs.py`'s `RunManager` now depends on directly (see "Reused for" table above).
   **This ADR does not resolve that conflict** — it only records it, because the PRD file lives outside this documentation pass's scope (`/home/yannic/code/AMAKBQA/.agent/Tasks/active/langgraph-rewrite.md`, on a different branch/worktree, untracked as of this writing). Whoever next touches that PRD (or merges `feat/langgraph-rewrite`) must either: (a) keep a thread+queue-compatible facade under `lifecycle_runner.py` so the API keeps working, or (b) port `ama_kbqa/api/runs.py`'s consumer loop onto the new `graph.astream` streaming primitive *before* Phase 4 deletes the bridge, or (c) explicitly decide the React frontend is out of scope for that rewrite and freeze it on the pre-rewrite agent implementation. Flagging this loudly here is the point of this ADR entry — do not silently let the two branches drift into a broken merge.

## Threshold to revisit (updated)

Decision 1's original threshold (multi-user auth, real-time collaboration, URL-routable deep-linking) is unchanged for the *internal* trace/graph inspector — it still doesn't need a migration. For the React demo frontend specifically, the trigger to reconsider its architecture is: (a) load that a single uvicorn worker cannot serve, or (b) a requirement that concurrent sessions never see each other's model/temperature setting — either would force moving past the process-global override and in-memory run state described above.

## Addendum 2026-09-16: settings panel made data-driven, not forked

**Problem.** This ADR's premise ("one API, thin reuse layer, no per-branch React fork") had an unaddressed gap: the user asked that the React frontend ship on *every* demo branch, and the booth/local branches need settings the public branch must never show (which chat/embedding endpoint, whether its key is configured, local-server reachability), while `demo-public` needs none of that. The naive fix — branch-specific React components — would have reopened exactly the "two UIs to keep in sync" cost this ADR already accepted as a trade-off, this time *inside* the one UI, per branch.

**Decision.** Keep the React code identical across branches; make the settings panel **data-driven** from `/api/meta` instead. A new `settings` block (`level`, `controls`, `endpoints`, `diagnostics` — full contract in `System/demo_react_frontend.md`) is rendered generically by React; branches differ by a config flag (`[frontend] settings_level`) plus data returned from three `ama_kbqa/api/meta.py` seams (`endpoint_rows()`, `diagnostic_rows()`, `endpoint_notices()`), never by React code. `api_key` state is reported as a boolean + env-var-name hint, **never key material**.

**Why this over a per-branch React fork:** the whole point of this ADR was to avoid a maintenance fork; extending that principle to settings-panel content keeps forward merges to demo-public/booth/llamacpp a one-line-per-config-file conflict (`demo-public`'s `39f9666` is exactly that: 2 lines changed, no Python) instead of a UI merge conflict every time a branch's endpoint list changes.

**Why not push endpoint detail into the existing `chat_controls` picker/model-notices path instead:** that path already answers "which models can I pick," at model granularity; this needed a build-level answer ("is this endpoint reachable, is its key set") at endpoint/provider granularity, read once for the settings panel rather than re-derived per model. Keeping them separate meant `demo-booth`'s notice dedupe (§ below) is the only place the two have to be reconciled, not every row.

**Trade-off accepted:** the three seams are still branch-specific *Python*, so `PROVIDER_LABELS`/`PROVIDER_KEY_ENV` (shared dicts) and `endpoint_notices()` (substantially rewritten by `demo-booth`) are recurring forward-merge friction points — smaller and more mechanical than a React fork, but not zero. See `System/demo_react_frontend.md`'s "Follow-ups not yet done" for the specific known friction.

**Status:** shipped on `demo-v2-int` (`2f0d406`), `demo-public` (`39f9666`), `demo-booth` (`2b97e4e`), `demo-llamacpp` (`3854c7b`). Not yet ported to the v1-line branches (`demo-hetzner`, `demo-kit-models`, `demo-bwcloud*`), which don't have the React frontend at all.

---
summary: Architecture of the second demo frontend — React SPA (web/) + FastAPI/SSE backend (ama_kbqa/api/), 7 endpoints, SSE shapes, ports 8505/8506, single-uvicorn-worker rule, branch-portable model picker.
---

# React Demo Frontend + API

## Related Docs
- [System/demo_bwcloud_frontend.md](./demo_bwcloud_frontend.md) — the Streamlit demo this frontend runs alongside (default, unchanged)
- [Decisions/react-frontend-second-ui.md](../Decisions/react-frontend-second-ui.md) — why this exists, trade-offs, and the cross-branch conflict with `langgraph-rewrite.md`
- [Decisions/trace-inspector-frontend-architecture.md](../Decisions/trace-inspector-frontend-architecture.md) — `TraceRecorder`, ContextVar span nesting, journal snapshots (reused unchanged)
- [Decisions/live-trace-and-chat-unification.md](../Decisions/live-trace-and-chat-unification.md) — the thread+queue live-update design `lifecycle_runner.py` implements, reused directly
- [Decisions/live-graph-two-source-subgraph.md](../Decisions/live-graph-two-source-subgraph.md) — the "Explored subgraph" data model this API also serves
- [SOP/running_react_demo_locally.md](../SOP/running_react_demo_locally.md) — how to run it

---

## What this is

A second demo frontend, shipped 2026-09-16, running **next to** the Streamlit
app (which stays the default and is behaviourally unchanged): a React SPA
(`web/`) talking to a small FastAPI backend (`ama_kbqa/api/`). It targets a
booth/public audience — no Streamlit widget chrome, a static nginx-served
build, no SSH tunnel dependency baked into the UI itself.

The API is **not** a reimplementation of agent logic. It calls the exact same
helper functions the Streamlit chat page (`ama_kbqa/frontend/chat.py`)
calls: `agent_factory`, `lifecycle_runner` (`LiveLifecycleState`/`start_run`/
`drain_into`/`live_graph_snapshot`/`reconstruct_orchestrator`), `lifecycle_svg`
+ `orchestrator_svg` (server-rendered SVG strings), `live_graph_data`
(`to_vis_payload`/`graph_from_trace`/`answer_node_ids`/`apply_caps`),
`chat_controls`, `styling.ansi_to_html`, `trace_render.summarise`, and
`pricing`. See [react-frontend-second-ui.md](../Decisions/react-frontend-second-ui.md)
for why this is additive, not a migration, and the trade-offs it accepted.

---

## Processes and ports

| Service | Container | Host port | Purpose |
|---|---|---|---|
| `frontend` (Streamlit) | `frontend_ama_kbqa` | `127.0.0.1:8502` | Default demo UI, unchanged |
| `api` (FastAPI) | `api_ama_kbqa` | `127.0.0.1:8506` | Backend for the React app; reached internally by `web` as `http://api:8506` |
| `web` (nginx + React build) | `web_ama_kbqa` | `127.0.0.1:8505` | Static SPA + reverse proxy for `/api/` |

All three are `127.0.0.1`-only (SSH-tunnel-only deployment model, same as the
rest of this project — see `SOP/hetzner_deployment.md`). `api` and `web` are
built from the same `Dockerfile.frontend` recipe as `frontend` (`api` overrides
the container `command`); `web` builds from its own context `./web` with a
separate `Dockerfile`. The root `.dockerignore` excludes `web/` from the
Python build context so `web/node_modules` never enters `frontend`/`api`
image builds.

`api` carries `extra_hosts: host.docker.internal:host-gateway` so agents can
reach a host-side `llama-server` (the `demo-llamacpp` branch's pattern,
harmless on branches that never use it). It deliberately does **not** mount
`.streamlit_cache` — it never runs Streamlit.

## Running it

`ama-kbqa-api` (console script, `pyproject.toml` → `ama_kbqa.cli:api`) always
hard-codes `uvicorn.run(..., workers=1, ...)`. `docker-compose.yml`'s `api`
service overrides the container command to
`["ama-kbqa-api", "--host", "0.0.0.0", "--port", "8506"]`. See
`SOP/running_react_demo_locally.md` for the full dev loop.

---

## Single-worker rule (hard constraint)

`ama_kbqa/api/runs.py`'s `RunManager` holds `sessions: dict[str, Session]` and
`runs: OrderedDict[str, Run]` — including every in-flight `Run`'s live agent
object, its worker-thread queue, and its `LiveLifecycleState` — **in process
memory**. There is no external store (no Redis, no DB). A second uvicorn
worker would be a second, disjoint copy of this state: a request landing on
worker B could not see a run started on worker A, so `GET /api/runs/{id}/events`
would 404 half the time. This is why both `cli.py:api()` and the `docker-compose.yml`
comment hard-code exactly one worker. It also means the API process cannot be
horizontally scaled or safely restarted mid-run without redesigning run/session
state to live outside the process — see the ADR's trade-off #3.

---

## Request flow / concurrency model

One `asyncio` consumer task per run (`RunManager._consume`) drains the run's
worker-thread `queue.Queue` (`lifecycle_runner.drain_into`) on a 0.4s tick
(same cadence as the Streamlit page's `@st.fragment(run_every=0.4)`) and
**publishes an immutable `Snapshot`** onto the `Run` object. `GET
/api/runs/{id}/events` (the SSE handler) only ever *reads* `run.snapshot` — it
never drains the queue itself. This split is load-bearing: a `queue.Queue`
drained by two readers loses notifications at random (the same rule
`live_graph_panel` follows on the Streamlit side), so if a second reader ever
called `drain_into` directly, both readers would silently miss events.

Each SSE connection keeps its own `_Connection` (log/graph "last sent"
markers) so log HTML and graph payloads are only re-sent when they actually
changed — `snapshot_data()` nulls out `log_html`/`graph` fields the client
already has.

---

## Endpoints (all under `/api`, JSON unless noted)

| Method + path | Purpose |
|---|---|
| `GET /api/health` | `{"status": "ok"}` liveness probe |
| `GET /api/meta` | Everything the app needs for its first screen: title, agent list, suggestions, model catalog, default model/temperature, `live_graph` flag, demo-mode limits |
| `POST /api/runs` | Body `{session_id, question, agent, model, temperature}` → `201 {run_id, continuation}`. Builds or reuses the session's agent, starts the run, returns immediately (the run itself streams via `/events`) |
| `GET /api/runs/{id}/events` | Server-Sent Events: `snapshot` frames while running, then one `done` or `error` frame, then the stream closes |
| `GET /api/runs/{id}` | Current `Run.record()` — session/agent/question/status plus the final payload once finished |
| `GET /api/runs/{id}/trace` | `409` until `run.finished`; then `{trace_id, summary, events}` — the same shape `trace_render.summarise` produces for the (retired) Streamlit Trace Inspector page |
| `POST /api/sessions/{id}/reset` | `204`; clears the session's stored multiturn agent. `409` if a run is still in flight for that session |

Errors: `ApiError` → `{"detail": str}` with its own status code (404 unknown
run, 409 conflicting run/reset, 429 demo rate limit). Malformed request bodies
get `400 {"detail": str}` (not FastAPI's default 422 list) via a
`RequestValidationError` handler in `app.py`.

### SSE frame format

Each frame: `id: <seq>\nevent: <kind>\ndata: <json>\n\n`. `<kind>` is
`"snapshot"`, `"done"`, or `"error"`; a bare `: ping\n\n` comment line is sent
every 15s of silence to keep the connection alive through proxies.

**`snapshot` data** (from `_Connection.snapshot_data`):
```
{
  seq, status, stage, elapsed_s, span_count,
  figure: {kind: "lifecycle"|"orchestrator", svg: "<svg>...</svg>"},
  subagents: [{id, display, status, svg}, ...],   // orchestrator runs only
  log_html: "<span>...</span>" | null,             // null when unchanged since last frame
  graph: {nodes, edges, new_ids, stats, ...} | null // null when unchanged; live_graph=false → always null
}
```

**`done` data** (from `RunManager._final_event`):
```
{
  run_id, status: "done", agent, question, answer, duration_s,
  tokens: {prompt, completion, total} | null, cost: "$0.0071" | null,
  model, model_key, provider, trace_id, log_html,
  figure, subagents, graph  // frozen (non-live) equivalents of the snapshot fields
}
```

**`error` data:** `{run_id, status: "error", message, log_html}`.

---

## stdout routing (`ama_kbqa/api/stdout_router.py`)

Agents report progress via plain `print()` (ANSI-coloured); the live log the
UI shows *is* that captured stdout. The Streamlit page gets away with
`contextlib.redirect_stdout`, which swaps the **process-global** `sys.stdout`
— fine for one browser tab, broken for a server where two runs can be in
flight at once (whichever run started last would steal the other's log).

`stdout_router` installs **one permanent `RoutingStdout` proxy** as
`sys.stdout` at API startup (`app.py`'s lifespan `install()`/`uninstall()`)
and routes every `write()` through a `ContextVar[Optional[RunLog]]`:

- Each run's worker thread binds its own `RunLog` via a new, optional,
  backward-compatible `start_run(..., on_thread_start=...)` hook added to
  `ama_kbqa/frontend/utils/lifecycle_runner.py` — called before the thread's
  event loop exists, so every coroutine and every `asyncio.to_thread` call the
  run makes afterward inherits the binding (contexts propagate to spawned
  tasks and `to_thread` calls; they do not propagate across unrelated
  threads).
- Any other thread (uvicorn's own loop, the threadpool for unrelated work)
  has no binding and writes straight through to the real stdout.
- A `ContextVar` rather than a thread-id map, so the binding dies with the
  worker's context — a later thread reusing the same OS thread id cannot
  inherit a stale run's log.

`RunLog` caps itself at 2MB (`DEFAULT_MAX_CHARS`), dropping from the head and
setting `truncated=True`; the UI only ever renders the last ~100KB
(`runs.LOG_TAIL_CHARS`) via `log_tail_html`.

---

## Branch-portable model picker

`GET /api/meta` and `POST /api/runs` both go through `ama_kbqa/api/meta.py`'s
`ModelCatalog`, which **feature-detects** the shape of `chat_controls` at
runtime so the same `ama_kbqa/api/` code runs unchanged across branches:

- **Provider-aware** (`demo-booth`, `demo-llamacpp`): `chat_controls` exposes
  `available_choices()`, returning `ChatModelChoice` objects that carry a
  `provider` next to the model id. Each becomes a `ModelOption` whose `key`
  (the `/meta` "id", and what `POST /api/runs`'s `model` field expects back)
  is the picker key `"provider:model"`. `/api/meta`'s response carries a
  `model_notices` array (e.g. hidden/unavailable entries). `runs.py`'s
  `apply_model_settings` inspects `chat_controls.apply_chat_settings`'s
  signature (`inspect.signature`, `_accepts_provider`) and passes
  `provider=option.provider` only when the function accepts it.
- **KIT-only** (`demo-v2-int`): no `available_choices`; the catalog is built
  from the live KIT `/models` endpoint (`fetch_provider_models_meta("kit")`,
  filtered by `chat_controls.filter_selectable_models`, with an offline
  fallback list). Ids are bare KIT model ids; `provider` is always `"kit"`.
  `apply_model_settings` calls the two-argument `apply_chat_settings(model,
  temperature)`.

`meta.provider_aware()` (`callable(getattr(chat_controls,
"available_choices", None))`) is the single detection point. The catalog is
cached in-process: 300s TTL for a successful KIT/choices fetch, 60s for the
offline fallback, 30s for the provider-aware path (matching
`available_choices()`'s own internal caching: KIT/OpenRouter 300s, the
llama.cpp server 30s since the operator may swap the model behind it).

**Verified post-merge (2026-09-16):** `demo-booth` — 365 tests, 10 models
spanning kit/openrouter/deepseek. `demo-llamacpp` — 384 tests, single local
model priced "Runs locally: no API cost" with a server-down notice when the
llama-server host is unreachable.

---

## Frontend (`web/`)

Vite + React + TypeScript SPA. Stack details, the dev loop (`npm run dev`,
`npm run dev:mock`), and the build/container recipe are in `web/README.md` —
not duplicated here; see `SOP/running_react_demo_locally.md` for the
operational commands.

- **vis-network is bundled**, not CDN-loaded — the opposite of the Streamlit
  Graph View's approach (`trace-inspector-frontend-architecture.md` Decision
  5, CDN by design for zero Python deps). Here there's no equivalent
  "zero deps" motivation and a public-facing static build should not depend on
  a third-party CDN being reachable.
- Libre Franklin is bundled via `@fontsource-variable/libre-franklin` — no
  network font load at runtime.
- Nginx (`web/nginx.conf`) serves the static build and reverse-proxies `/api/`
  to `http://api:8506` with `proxy_buffering off`, `gzip off` for that
  location, and a 3600s read/send timeout — required for SSE: buffering or
  gzip on that path would delay or coalesce snapshot frames, and a multi-minute
  run would otherwise hit nginx's default read timeout.
- The lifecycle/orchestrator figures and the live log are server-rendered
  HTML/SVG strings injected as-is by the client (trusted, same-origin, server
  controlled); answers are Markdown, rendered client-side and never as raw
  HTML.
- Session identity: a random UUID in `sessionStorage` — a new tab is a new
  demo session; reloading the same tab resumes the conversation, including a
  run still in flight.

### Branding

KIT green `#009682` is the accent (`--brand`). White text on `#009682` is only
3.7:1 contrast (below WCAG AA); text-bearing fills instead use
`--brand-strong: #00796a` (5.3:1).

---

## Verification evidence (2026-09-16)

- 346 tests passing on `demo-v2-int`, `ruff` clean.
- Full stack run in Docker: `docker compose up -d --no-deps api web`.
- A real question through nginx on `127.0.0.1:8505` streamed 58 SSE snapshots
  over 57s and answered "Albert Einstein was born in Ulm," with the
  orchestrator figure, the KQAPro sub-agent pane, live graph growth, and a
  29-event trace all rendering correctly.
- Browser check (Playwright) confirmed: the About dialog, a live run in
  progress, the finished answer, and the Trace span tree.

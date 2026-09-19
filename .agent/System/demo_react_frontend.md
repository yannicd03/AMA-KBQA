---
summary: Architecture of the second demo frontend — React SPA (web/) + FastAPI/SSE backend (ama_kbqa/api/), 8 endpoints, SSE shapes, host ports 2026/2027, single-uvicorn-worker rule, cooperative run cancellation, provider-carrying model picker with DeepSeek presets and a custom OpenRouter id, data-driven per-branch settings panel.
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
| `api` (FastAPI) | `api_ama_kbqa` | `127.0.0.1:2027` | Backend for the React app; **listens on `8506` inside the container**, reached internally by `web` as `http://api:8506` |
| `web` (nginx + React build) | `web_ama_kbqa` | `127.0.0.1:2026` | Static SPA + reverse proxy for `/api/` (container port 80) |

The host ports moved from `8505`/`8506` to `2026`/`2027` on 2026-09-16 (commit
`a41ce3d`). **Host bindings only:** the container ports are unchanged, so
`web/nginx.conf` still proxies `http://api:8506`, the compose command is still
`--port 8506`, and both healthchecks are untouched. Deployment note: `2026`/
`2027` fall inside the `2025–2030` range ORCA occupies on the shared Hetzner
box — see `SOP/hetzner_demo_deployment.md` §10 before publishing them there.

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
| `POST /api/runs` | Body `{session_id, question, agent, model, temperature, custom_model?}` → `201 {run_id, continuation}`. Builds or reuses the session's agent, starts the run, returns immediately (the run itself streams via `/events`). `custom_model` is read only when `model` is the free-text picker entry |
| `POST /api/runs/{id}/cancel` | `202 {run_id, cancelling: bool}`. Asks the run to stop and frees the session slot immediately. Never 404s — see "Cooperative cancellation" below |
| `GET /api/runs/{id}/events` | Server-Sent Events: `snapshot` frames while running, then one `done`, `error` or `cancelled` frame, then the stream closes |
| `GET /api/runs/{id}` | Current `Run.record()` — session/agent/question/status plus the final payload once finished |
| `GET /api/runs/{id}/trace` | `409` until `run.finished`; then `{trace_id, summary, events}` — the same shape `trace_render.summarise` produces for the (retired) Streamlit Trace Inspector page |
| `POST /api/sessions/{id}/reset` | `204`; clears the session's stored multiturn agent. `409` if a run is still in flight for that session |

Errors: `ApiError` → `{"detail": str}` with its own status code (404 unknown
run, 409 conflicting run/reset, 429 demo rate limit). Malformed request bodies
get `400 {"detail": str}` (not FastAPI's default 422 list) via a
`RequestValidationError` handler in `app.py`.

### SSE frame format

Each frame: `id: <seq>\nevent: <kind>\ndata: <json>\n\n`. `<kind>` is
`"snapshot"`, `"done"`, `"error"`, or `"cancelled"`; a bare `: ping\n\n`
comment line is sent every 15s of silence to keep the connection alive through
proxies.

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

**`cancelled` data:** `{run_id, status: "cancelled", answer, duration_s, log_html}`.
A distinct event kind, not a reused `error` — see "Cooperative cancellation".

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

- **Provider-aware** (`demo-v2-int` since 2026-09-16, `demo-booth`,
  `demo-llamacpp`): `chat_controls` exposes `available_choices()`, returning
  `ChatModelChoice` objects that carry a `provider` next to the model id. Each
  becomes a `ModelOption` whose `key` (the `/meta` "id", and what
  `POST /api/runs`'s `model` field expects back) is the picker key
  `"provider:model"`. `/api/meta`'s response carries a `model_notices` array
  (e.g. hidden/unavailable entries). `runs.py`'s `apply_model_settings`
  inspects `chat_controls.apply_chat_settings`'s signature
  (`inspect.signature`, `_accepts_provider`) and passes
  `provider=option.provider` only when the function accepts it.
- **KIT-only** (the pre-2026-09-16 shape, still supported): no
  `available_choices`; the catalog is built from the live KIT `/models`
  endpoint (`fetch_provider_models_meta("kit")`, filtered by
  `chat_controls.filter_selectable_models`, with an offline fallback list).
  Ids are bare KIT model ids; `provider` is always `"kit"`.
  `apply_model_settings` calls the two-argument `apply_chat_settings(model,
  temperature)`.

### What the picker offers on `demo-v2-int` (2026-09-16)

The list is assembled in `chat_controls.available_choices()`:

1. **The live KIT catalog first** — whatever `available_models()` returns
   (3 models as of 2026-09-16: Mistral Small 4, DeepSeek V4 Flash, GLM-5.3).
2. **Then each `[[frontend.chat_models]]` config entry in config order** — two
   OpenRouter DeepSeek presets (`deepseek/deepseek-v4-pro`,
   `deepseek/deepseek-v4.1-flash`), shipped in both `config.toml` and
   `config.docker.toml`.
3. **Then the free-text entry last** — key `openrouter:__custom__`, emitted in
   `/api/meta` with `custom: true`, where the presenter types any OpenRouter
   model id.

So six entries on 2026-09-16, but the count is not fixed: the KIT half comes
from a live catalog, and a config entry is dropped (with a `model_notices`
line) when its provider's key is unset, when the live OpenRouter catalog no
longer knows the id, or when the catalog says it cannot call tools. If the
OpenRouter catalog is unreachable the configured entries are kept unvalidated
and unpriced rather than hidden — a flaky network must not empty the picker.

**Custom ids** are validated by *shape* (`meta.custom_model_option`,
`_CUSTOM_MODEL_RE`, 200-char cap), sent in `POST /api/runs`'s separate
`custom_model` field, and **never minted into the catalog** (it is TTL-cached,
so a minted id would be dropped on refresh and could leak between sessions).
They show **no price** rather than a guessed one. A `custom_model` sent
alongside any other pick is ignored, not rejected.

**Provider propagation is load-bearing.** `apply_chat_settings` also exports
`AMA_KBQA_CHAT_PROVIDER` (next to the existing `AMA_KBQA_CHAT_MODEL` /
`AMA_KBQA_CHAT_TEMPERATURE`), because MCP tool servers are spawned with
`env=os.environ.copy()` and load their **own** `config.toml` — an in-memory
override alone leaves every sub-agent on KIT while the parent uses OpenRouter,
and they still answer, which hides the fault completely. See
`Decisions/demo-picker-provider-routing.md`.

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

## Cooperative cancellation

**Route:** `POST /api/runs/{id}/cancel` → `202 {run_id, cancelling}`.

The run is not killed. It executes on a daemon worker thread that owns its own
event loop, so the route only flips a `CancellationToken`; the agent notices it
at its own next checkpoint and returns through its own code, letting MCP tear
down on the loop that opened it. The mechanics live in
`System/agent_framework.md` § "Cooperative Cancellation"; the rationale in
`Decisions/cooperative-run-cancellation.md`.

What the API layer adds on top:

- **202, never 404.** Unknown, evicted and already-finished runs also answer
  `202`, with `cancelling: false` — stopping something already stopped is the
  caller's intent, and a 404 would render a successful Stop as a failure.
- **The session slot frees immediately.** `RunManager.cancel_run` clears
  `active_run_id` and drops `session.stored`, so the next question and "New
  chat" work at once. Before this, a stopped run left both returning `409` for
  its full remaining duration. A cancelled agent is never reused: its message
  stack was cut mid-turn and its MCP connection belongs to a loop that is still
  unwinding.
- **Overlap is bounded, not blocked.** Because the slot frees early, a new run
  can overlap a still-draining one. `Session.detached_runs` counts those, and
  `MAX_DETACHED_RUNS_PER_SESSION = 2` caps them — a third request gets `409`
  ("still winding down"). Detached runs also keep a session from being evicted
  as idle. Safety rests on `self.model` being frozen at agent construction, so
  a new run cannot retarget a draining agent.
- **The terminal event kind is decided by the token, never by the answer
  text.** A cancelled run comes back through the agent's normal return path, so
  its queue item is `__done__`; `_final_event` checks `run.cancel_token` first
  and emits `cancelled`.

**Stop latency** is dominated by whatever the run is waiting on: the LLM call is
synchronous and not interruptible, and there is deliberately no checkpoint
inside tool execution, so a long tool call completes first. A fast-path run
already in flight has no checkpoint at all.

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

## Verification evidence

**2026-09-16 (initial ship):**

- 346 tests passing on `demo-v2-int`, `ruff` clean.
- Full stack run in Docker: `docker compose up -d --no-deps api web`.
- A real question through nginx (then on `127.0.0.1:8505`) streamed 58 SSE
  snapshots over 57s and answered "Albert Einstein was born in Ulm," with the
  orchestrator figure, the KQAPro sub-agent pane, live graph growth, and a
  29-event trace all rendering correctly.
- Browser check (Playwright) confirmed: the About dialog, a live run in
  progress, the finished answer, and the Trace span tree.

**2026-09-16 (ports, cancellation, picker — `a41ce3d`/`34c2314`/`69cd5a9`):**

- Full suite **1137 passed, 0 failed** (up from 1003).
- `AMA_KBQA_CHAT_PROVIDER` crossing the subprocess boundary was proven by a
  control experiment in the container: with the variable stripped, the child
  carried model `deepseek/deepseek-v4-pro` but resolved `base_url` to KIT; with
  it exported, OpenRouter.

**Not verified — recorded honestly, do not assume these work:**

- **No browser check at all for this change.** The Stop button and the
  custom-model field have **never been rendered**. Chrome automation is broken
  on this machine (pre-existing regression, see below).
- **The client half of cancellation is covered by static source assertions
  only, never executed.** There is no JS test runner in this repo;
  `tests/api/test_web_client_contract.py` reads `web/src/api.ts` and
  `web/src/state.ts` as text. `tsc -b` passes, but no client code was run, and
  the infinite-reconnect failure mode it guards against was never reproduced
  nor observed fixed. This is the riskiest unverified line in the feature.
- **No real billed question** has been asked through a DeepSeek preset or a
  custom model id, so no end-to-end answer from either exists.
- **The detached-run cap is tested by injecting session state**, not by driving
  two genuinely overlapping real runs.
- The three commits were not individually test-verified in isolation, so the
  history may not bisect cleanly.

---

## Data-driven settings panel (2026-09-16, commit `2f0d406` + per-branch)

Every demo branch needs the React frontend, but not the same sidebar: the
booth and local (llamacpp) builds need endpoint/key visibility the public
build must never show. Rather than fork the React code per branch (which
turns every forward merge into a UI conflict), `GET /api/meta` grew a
declarative `settings` block and the React panel renders whatever it is
given. **No React code needs to change for a new branch** — see
`react-frontend-second-ui.md` for why this was chosen over forking the UI.

### The `settings` block

`build_meta()` (`ama_kbqa/api/meta.py`) adds `settings: settings_meta()`:

```
{
  level: "minimal" | "full",
  controls: {model, temperature, simplified_view, live_graph: bool},
  endpoints: {label, rows: [...], notices: [str, ...]} | null,
  diagnostics: {label, rows: [...]} | null
}
```

`endpoints` and `diagnostics` are always `null` at `level: "minimal"`.
`controls` says which always-known controls to offer — a branch can drop one
(e.g. hide `live_graph` when `get_live_graph_enabled()` is false) without
touching React.

Each endpoint row (`endpoint_row()`) always carries all ten keys so the
client can render uniformly: `id, label, role, provider, base_url, model,
model_source, api_key, status, detail`. `api_key` is `{"configured": bool,
"hint": "<ENV_VAR_NAME>"}` (via `api_key_state()`) or `null` meaning the
endpoint needs no key — **it never carries key material, not even masked,
not even a partial hint beyond the env var's name.** `status` is one of
`STATUS_OK` / `STATUS_UNREACHABLE` / `STATUS_UNKNOWN` / `STATUS_NO_KEY`; the
React panel styles unknown providers/roles/statuses neutrally rather than
breaking, so a branch is free to invent a new value.

`diagnostic_row(label, value, detail)` is a plain read-only fact — never a
secret, never a host path (`diagnostic_rows()` reports `CONFIG_PATH.name`,
not `CONFIG_PATH`).

### `settings_level` config flag

`[frontend] settings_level = "minimal" | "full"` in `config.toml` /
`config.docker.toml`, env override `AMA_FRONTEND_SETTINGS_LEVEL` (same
precedence pattern as `AMA_FRONTEND_LIVE_GRAPH`: env always wins over toml).
Getter: `ama_kbqa.config.get_frontend_settings_level()`. **Default is
`"minimal"`** — the safe end, so an old config, a non-demo deployment, or a
branch that forgot the key never starts advertising endpoint detail by
accident. Both shipped `config.toml`/`config.docker.toml` on `demo-v2-int`
opt in to `"full"`; `demo-public` overrides back to `"minimal"` (one line per
config file — see below).

At `"full"`, `demo-v2-int` itself (no branch override) shows the *generic*
`endpoint_rows()`/`diagnostic_rows()` implementation: one row for the
configured `[llm] chat_provider`, one for `[llm] embedding_provider`, and —
since 2026-09-16 — **one row per non-KIT chat provider the picker offers**,
derived from `[[frontend.chat_models]]` via `optional_chat_providers()` rather
than hard-coded. With the shipped config that third row is `openrouter-chat`.
Plus four diagnostic facts (Retrieval, Federation, Live graph, Config
filename).
Booth and llamacpp replace/extend those two functions; they don't change the
contract.

### The three branch seams

A branch extends exactly three functions in `ama_kbqa/api/meta.py` — nothing
else in that file, and nothing in React, needs to change:

| Seam | Purpose |
|---|---|
| `endpoint_rows() -> list[dict]` | Which endpoints this build talks to. Return `[]` to hide the section entirely. |
| `diagnostic_rows() -> list[dict]` | A handful of read-only build facts. Return `[]` to hide. |
| `endpoint_notices(rows) -> list[str]` | Warning lines under the endpoint list; default turns every non-`ok` row into one line. |

Shared helpers a branch composes from: `provider_endpoint_row()` (one row
for a `config.toml`-configured provider, derives status from
`api_key_state()` alone), `endpoint_row()` (raw 10-key row), `api_key_state()`,
`diagnostic_row()`, plus the module-level `PROVIDER_KEY_ENV` (provider →
env var name) and `PROVIDER_LABELS` (provider → display name) dicts and the
`STATUS_*` constants.

### Per-branch outcomes (2026-09-16)

| Branch | Commit | `settings_level` | What the panel adds |
|---|---|---|---|
| `demo-public` | `39f9666` | `"minimal"` | Nothing — exactly one changed line per config file (`settings_level = "full"` → `"minimal"`); `/api/meta` reports `endpoints: null, diagnostics: null`. Needs zero Python. |
| `demo-booth` | `2b97e4e` | `"full"` | Rows for KIT chat, KIT embedding, OpenRouter and DeepSeek, derived from `[[frontend.chat_models]]` via new `optional_chat_providers()` — not hard-coded. **A provider without its key stays visible** with `status: "no_key"` (hiding it would answer "can I use DeepSeek?" with silence). One notice per *provider*, not per model, worded differently from the picker's own `model_notices` and deduped against it so the presenter never reads the same sentence twice. Every billed row says so in both key states. Diagnostics add "Chat providers: N of M selectable" and "Embeddings: KIT KI-Toolbox, whatever the chat pick." |
| `demo-llamacpp` | `3854c7b` | `"full"` | Rows for the local chat server (`:8080`) and local embedding server (`:8081`), `api_key: null` (this build has no keys by design), with a **probed** status — reuses the picker's cached `chat_controls._fetch_llamacpp_models()` (3s timeout, 30s TTL) for chat, a direct `fetch_provider_models()` call for embeddings. Adds its own failure cache (`_probe_cache`, 30s success / 10s failure TTL) because `st.cache_data` never caches exceptions, so a stopped server would otherwise repay the connect timeout on every `/api/meta` call. Down-server rows point at `scripts/start_local_llm.sh`. Diagnostics add "Runs locally: chat and embeddings, no cloud API." |

**Known limitation, recorded honestly:** on `demo-v2-int`'s own
`openrouter-chat` row and on every `demo-booth` row, `status: "ok"` means the
API key is present in the environment — **not** that the endpoint
answered. Nothing is probed for OpenRouter/DeepSeek (a deliberate choice:
probing would put two network round-trips in front of the demo's first
screen on every `/api/meta` call), so an expired or revoked booth key still
renders `ok`. `demo-llamacpp` is the only branch that actually probes.

### Follow-ups not yet done

- The notice dedupe (dropping a line the model picker already shows via
  `model_notices`) exists only on `demo-booth`'s `endpoint_notices()`.
  Consider pushing it down to the shared default on `demo-v2-int` so every
  branch benefits, not just booth.
- `demo-llamacpp`'s `status: "ok"` path is exercised only through
  monkeypatched fetches in tests; nobody has confirmed against a *live*
  `llama-server --embedding` process that `GET {base_url}/models` actually
  answers. If it doesn't, the embedding row would read `unreachable` even
  with the server up.
- Forward-merge friction to expect: `PROVIDER_LABELS` / `PROVIDER_KEY_ENV`
  are shared dicts both booth and llamacpp edited; `endpoint_notices()` was
  substantially rewritten on booth; `tests/api/test_meta_settings.py` picked
  up booth-only assertions that booth then split into a separate
  `tests/api/test_meta_settings_booth.py` to keep the shared file small — a
  future branch should follow that pattern rather than growing the shared
  file further.
- **Confirmed on the 2026-09-17 forward merge** (`demo-v2-int` →
  `demo-public`, `demo-booth`, `demo-llamacpp`, landing at `a68ee74` on this
  branch): git merged two same-name top-level definitions side by side **with
  no conflict marker**, twice, on two different branches. On `demo-booth`,
  booth's `_FRONTEND_CHAT_PROVIDERS = ("kit", "openrouter", "deepseek")` in
  `ama_kbqa/config.py` ended up ~8 lines above `demo-v2-int`'s
  `_FRONTEND_CHAT_PROVIDERS = ("kit", "openrouter")`; the later binding wins
  at runtime, so every direct-DeepSeek picker entry would have been silently
  dropped, with nothing but a log warning. On `demo-llamacpp`, the same
  collision recurred on `_FRONTEND_CHAT_PROVIDERS`, plus a second one on
  `register_runtime_pricing` in `ama_kbqa/pricing.py` (identical bodies,
  different docstrings — indistinguishable to a marker-only diff review).
  **A clean, marker-free merge is not evidence of correctness for this
  file pair.** After any forward merge touching `config.py` or `pricing.py`,
  grep for duplicate top-level assignments/defs rather than trusting the
  absence of `<<<<<<<` markers — e.g.
  `git grep -c "^_FRONTEND_CHAT_PROVIDERS = " HEAD -- ama_kbqa/config.py`
  should print `1`.
- **Also confirmed the same day:** a conflict can swallow a needed line
  instead of marking it. On `demo-llamacpp`, resolving the conflict in
  `tests/api/test_meta_settings.py` dropped the
  `by_role = {row["role"]: row for row in rows}` line, leaving the test
  referencing an undefined name; restored from `b5e486e`. The merge result
  imported and looked plausible — it was wrong by omission, not by marker.
- **Test-writing rule for any file that forward-merges across all demo
  branches:** assert the invariant, never a branch's own config value.
  `tests/api/test_meta_settings_openrouter.py` originally pinned
  `cfg.get_frontend_settings_level() == "full"` to guard the TOML
  array-table hazard (see `Decisions/demo-picker-provider-routing.md` —
  a plain `[frontend]` key written after `[[frontend.chat_models]]` silently
  becomes a key of the last array entry). Pinning the *value* instead tested
  `demo-v2-int`'s own identity and broke `demo-public`, which ships
  `"minimal"` by design. Fixed in `a68ee74` by relaxing to
  `cfg.get_frontend_settings_level() in cfg.FRONTEND_SETTINGS_LEVELS`;
  placement (the actual hazard) stays guarded branch-agnostically by
  `tests/test_config_frontend_settings.py`.
- The v1-line branches (`demo-hetzner`, `demo-kit-models`, `demo-bwcloud*`,
  ~280 commits diverged from `demo-v2-int`) do not have the React frontend
  at all yet, let alone this settings panel. Pending a decision on whether
  they get it.
- **Never visually reviewed.** Both new sidebar sections (booth and
  llamacpp) are verified only by tests and live `curl` against `/api/meta`.
  Browser automation was broken on this machine for the whole session
  (Chrome launches but never completes its handshake — 180s timeouts across
  the Playwright MCP, direct Playwright, a CDP port, and `chrome
  --screenshot`; it worked earlier the same session doing the verification
  in the section above, so this looks like an environment regression, not a
  code problem).

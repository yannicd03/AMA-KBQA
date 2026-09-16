---
summary: How to run the React demo frontend + FastAPI backend locally — full Docker stack (api + web services) or a fast dev loop (uv run backend, Vite hot reload frontend); checklist for adding a branch's settings-panel rows; two testing gotchas (.env load_dotenv, root-level test file skipped by scoped pytest runs).
---

# SOP: Running the React Demo Frontend Locally

## Related Docs
- [System/demo_react_frontend.md](../System/demo_react_frontend.md) — architecture, endpoints, single-worker rule
- [Decisions/react-frontend-second-ui.md](../Decisions/react-frontend-second-ui.md) — why this exists, what it trades off
- [SOP/hetzner_demo_deployment.md](./hetzner_demo_deployment.md) §10 — remote/booth deployment status (not done)

---

Two ways to run it locally: full Docker stack (closest to production), or a
fast dev loop (backend via `uv run`, frontend via Vite with hot reload).

## Full stack (Docker)

Prerequisite: `virtuoso` and `qdrant` already running and seeded (see the
root `README.md` / `SOP/hetzner_deployment.md` for first-time bootstrap).

```bash
docker compose up -d --build api web
```

This builds and starts only the `api` and `web` services (the existing
Streamlit `frontend` service, `virtuoso`, and `qdrant` are left alone if
already running — `web` `depends_on: api: condition: service_healthy`, and
`api` depends on `qdrant`/`virtuoso` being started). Then:

- React app: `http://127.0.0.1:8505`
- API directly (debugging only, SSH-tunnel-only like every other port here):
  `http://127.0.0.1:8506/api/health`, `http://127.0.0.1:8506/api/docs`
  (FastAPI's interactive docs — `app.py` deliberately exposes `/api/docs` and
  `/api/openapi.json` but not `/api/redoc`)

Rebuild after a Python change: `docker compose up -d --build api` (the `web`
service only needs a rebuild for frontend changes: `docker compose up -d
--build web`).

## Dev loop (fast iteration)

Backend, one worker, against the same `virtuoso`/`qdrant` (via
`config.toml`'s host-side settings, not `config.docker.toml`):

```bash
uv run ama-kbqa-api            # binds 0.0.0.0:8506
```

Frontend, in a second terminal:

```bash
cd web
npm run dev                    # http://localhost:5173, Vite proxies /api to 127.0.0.1:8506
```

To point the dev proxy at a different backend: `AMA_KBQA_API=http://host:port
npm run dev`. Full frontend dev-loop details (type checking, the built-in
mock mode for working on the UI without a backend, container build) are in
`web/README.md` — not duplicated here.

## The single-worker rule

`ama-kbqa-api` always runs with exactly one uvicorn worker
(`ama_kbqa/cli.py:api()` hard-codes `workers=1`) — every run, session, and
persisted multiturn agent lives in that process's memory (see
`System/demo_react_frontend.md` "Single-worker rule"). Do not try to scale
this with `--workers N` or multiple container replicas; it will silently
break run lookup (a request landing on a different worker than the one that
started the run gets a 404).

## Troubleshooting

- **`POST /api/runs` returns 409 "already has a run in flight"**: the
  session (keyed by the `sessionStorage` UUID, see `web/README.md`) has an
  unfinished run. `POST /api/sessions/{id}/reset` also 409s while a run is in
  flight — wait for it to finish or start a new browser tab (new session).
- **`GET /api/runs/{id}/trace` returns 409**: the trace is only available
  once the run's `done`/`error` event has been sent; poll `/api/runs/{id}`'s
  `status` field or wait for the SSE stream's terminal frame first.
- **SSE stream stalls or buffers in a browser but works with `curl`**: check
  whatever reverse proxy sits in front (nginx's `web` container already sets
  `proxy_buffering off` + `gzip off` on `/api/` — see `web/nginx.conf`); any
  additional proxy layer (e.g. a future cloudflared ingress) needs the same
  settings or SSE frames will be delayed/coalesced.
- **Model list is empty or shows only the offline fallback**: on the
  KIT-only path (`demo-v2-int`) this means the live KIT `/models` fetch
  failed (see `System/demo_react_frontend.md` "Branch-portable model
  picker") — check `AMA_API_KEY`/endpoint reachability, same as the
  Streamlit page.
- **`env -u OPENROUTER_API_KEY pytest ...` still behaves as if the key is
  set**: `ama_kbqa/config.py` calls `load_dotenv()` at import time, and this
  repo has a `.env` — `load_dotenv()` re-populates the var from the file
  regardless of what you unset on the command line. Use `OPENROUTER_API_KEY=
  pytest ...` (empty value) instead of `env -u`, or the "no key configured"
  path in a settings-panel test will silently not exercise what you think it
  does.
- **`pytest tests/api tests/frontend` silently skips the frontend-settings
  config tests**: `tests/test_config_frontend_settings.py` sits at the
  `tests/` root, not under `tests/api/` or `tests/frontend/` — a
  directory-scoped pytest invocation collects nothing from it. Run it
  explicitly (`pytest tests/test_config_frontend_settings.py`) or run the
  whole `tests/` tree when touching `get_frontend_settings_level()`.

## Adding a settings-panel row on a new branch

The settings panel (`/api/meta`'s `settings` block) is data-driven — no
React changes needed. See `System/demo_react_frontend.md` § "Data-driven
settings panel" for the full contract; the checklist for a new branch:

1. Set `[frontend] settings_level = "full"` in `config.toml` /
   `config.docker.toml` (or leave it `"minimal"` if the branch needs no
   endpoint detail — that's the whole `demo-public` diff).
2. Override `endpoint_rows()` and/or `diagnostic_rows()` in
   `ama_kbqa/api/meta.py` for this branch's endpoints. Compose from
   `provider_endpoint_row()` for anything `config.toml`-configured, or write
   a probed row by hand (see `demo-llamacpp`'s `_local_chat_row()`/
   `_local_embedding_row()` for the pattern: cache successes *and*
   failures — `st.cache_data` does not cache exceptions).
3. Only override `endpoint_notices()` if the default (one line per
   non-`ok` row) isn't right for the branch — e.g. `demo-booth` needed
   per-provider rather than per-model wording, deduped against the model
   picker's own `model_notices`.
4. Never put key material in a row — `api_key_state()` reports only
   `{"configured": bool, "hint": "<ENV_VAR_NAME>"}`.
5. If the new tests would collide with the shared `tests/api/test_meta_settings.py`
   file, split branch-specific assertions into their own file (booth's
   `tests/api/test_meta_settings_booth.py`), matching the pattern set
   2026-09-16.

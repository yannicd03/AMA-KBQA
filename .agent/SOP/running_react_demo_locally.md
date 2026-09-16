---
summary: How to run the React demo frontend + FastAPI backend locally — full Docker stack (api + web services) or a fast dev loop (uv run backend, Vite hot reload frontend).
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

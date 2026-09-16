# AMA-KBQA web UI

A React demo frontend for AMA-KBQA. It runs next to the Streamlit demo (which
stays the default) and talks to the FastAPI backend in `ama_kbqa/api/` over a
small JSON + Server-Sent Events API. The request and response shapes live in
`src/api.ts`; the backend implementation of the same contract is `ama_kbqa/api/`.

Stack: Vite, React, TypeScript, vis-network (bundled, no CDN), react-markdown,
and Libre Franklin via `@fontsource`. Nothing is loaded from the network at
runtime except `/api`.

## Dev loop

```bash
cd web
npm ci
npm run dev          # http://localhost:5173, proxies /api to 127.0.0.1:8506
```

Start the backend first (`uv run ama-kbqa-api`, one worker). To point the dev
proxy somewhere else, set `AMA_KBQA_API=http://host:port`.

Without a backend, use the built-in mock. It serves fixture metadata and plays
a scripted run with a live lifecycle figure, a growing graph and a trace:

```bash
npm run dev:mock     # same as VITE_MOCK=1 npm run dev
```

Mock knobs in the URL: `?mockSpeed=3` plays runs faster, `?mockPause=14`
freezes a run at tick 14, and `?mockProviders=1` serves the provider-aware
model list (grouped models plus a `model_notices` entry) that the demo-booth
and demo-llamacpp builds send. A question containing "fail" ends in an error
event, one containing "limit" gets a 429. The mock sits behind a compile-time
constant and is not part of production builds.

## Build

```bash
npm run build        # tsc -b (type check), then vite build into dist/
npm run preview      # serve dist/ locally
```

## Container

```bash
docker build -t ama-kbqa-web web/
```

The image is nginx serving `dist/` with an SPA fallback, gzip, long-lived
caching for the hashed files in `/assets/`, and `/api/` proxied to
`http://api:8506` with buffering off and a one-hour read timeout, so the event
stream of a long run is not cut or delayed. The compose service is `web`
(container `web_ama_kbqa`), next to `api`.

## Notes

- The session id is a random UUID in `sessionStorage`: a new tab is a new demo
  session, and a reload of the same tab resumes the conversation, including a
  run that is still in flight.
- The lifecycle and orchestrator figures and the log are HTML rendered and
  escaped by the server; they are injected as-is. Answers are Markdown and are
  rendered without raw HTML.
- KIT green `#009682` is the accent. White text sits on the darker
  `#00796a` so it meets WCAG AA.

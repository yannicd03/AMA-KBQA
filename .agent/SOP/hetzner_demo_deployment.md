---
summary: Public demo deployment runbook for the shared Hetzner box (compose projects, ports, Cloudflare tunnels). As of 2026-09-16, its React frontend section records that the new React frontend (web/api) is verified locally in Docker only — not deployed, staged, or given a port/ingress on this box — and that its local default ports 2026/2027 fall inside ORCA's 2025-2030 range here.
---

# SOP: Public Demo Deployment on the Shared Hetzner Box

## Related Docs
- [SOP/hetzner_deployment.md](hetzner_deployment.md) — **superseded for the public demo** (that runbook documents the old `~/AMAKBQA-main` benchmark-stack deploy on the same box; kept for the benchmark stack itself, which is unaffected). See "Relationship to the older Hetzner SOP" below.
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) — frontend architecture this SOP deploys (file/branch naming still says "bwcloud"; the code has since moved to the `dev`-based `demo-v2` line, see §9)
- [Decisions/demo-bwcloud-frontend-divergence.md](../Decisions/demo-bwcloud-frontend-divergence.md) — why the demo build diverges from the full app
- [Decisions/federated-dispatch-and-fusion.md](../Decisions/federated-dispatch-and-fusion.md) — federated dispatch backend behind demo v2's second Orchestrator picker entry (§9)
- [Decisions/public-repo-deploy-config-extraction.md](../Decisions/public-repo-deploy-config-extraction.md) — why host details live in `deploy/hetzner/deploy.env` instead of this repo
- [Project Architecture](../System/project_architecture.md) — overall system overview

---

## What happened and why

bwcloud (host `seminar`) reaches end of life on **2026-09-15**. On **2026-09-14** the public demo at **https://amakbqa.yanlab.de** moved off bwcloud onto the **shared Hetzner box** (4 vCPU / 7.6 GiB RAM + 4 GiB swap, **no buildx**).

> **Host details are not in this repository — start here on a fresh clone.**
> The origin address, ssh alias, deploy user and Cloudflare tunnel id live in
> `deploy/hetzner/deploy.env`, gitignored. Before anything else in this SOP:
>
> 1. `cp deploy/hetzner/deploy.env.example deploy/hetzner/deploy.env` and fill
>    in the real values.
> 2. `./deploy/hetzner/render-deploy-config.sh` — renders
>    `cloudflared-amakbqa.yml` and `cloudflared-amakbqa.service` from the
>    committed `.template` files; refuses to run if `CLOUDFLARE_TUNNEL_ID` is
>    still the placeholder.
> 3. Then continue with the rest of this SOP as written.
>
> Throughout this SOP, `$DEPLOY_SSH_ALIAS`, `$DEPLOY_USER` and `$DEPLOY_HOST`
> refer to `deploy.env` values. Publishing the origin address would let
> traffic bypass the Cloudflare proxy and reach the box directly — see
> [Decisions/public-repo-deploy-config-extraction.md](../Decisions/public-repo-deploy-config-extraction.md)
> for the full rationale.

That box already runs other services, none of which this migration touched:
- **ORCA** — MAS production stack, compose project `mas-in-production` at `/srv/orca`, ports 8001, 2025–2030, 6333/6334, 27017; root `cloudflared.service` tunnel `orca`.
- **AMA-KBQA benchmark stack** — `~/AMAKBQA-main`, compose project `ama-kbqa`, owns ports 8890/1111/6335/8502 (normally stopped). This is the stack `SOP/hetzner_deployment.md` documents.
- **scoresheet-translator** — port 9000, `cloudflared-scoresheet.service`.

The demo is a **fourth, independent** compose project on the same box. Port and container-name collisions with the above are the main hazard — see §2. (§9 adds a *fifth*, staging-only project for demo v2.)

---

## 1. Layout

- Checkout: `~/amakbqa-demo` on the server, branch `demo-hetzner` (= `demo-bwcloud-deploy` at `2917427` + Hetzner-specific overrides + KIT model changes, see §4).
- `.env` (`KIT_API_KEY`, `DEMO_MODE`) relayed from bwcloud, mode `600`.
- `docker-compose.hetzner.yml` — override file, always applied together with the base `docker-compose.yml` (see §5).

## 2. Compose override (`docker-compose.hetzner.yml`)

| Aspect | Value | Why |
|---|---|---|
| Compose project name | `amakbqa-demo` | Keeps container/volume/network namespace separate from `ama-kbqa` (benchmark stack) and `mas-in-production` (ORCA) on the same host |
| Containers | `virtuoso_amakbqa_demo`, `qdrant_amakbqa_demo`, `frontend_amakbqa_demo` | Distinct names from the benchmark stack's `virtuoso_ama_kbqa` / `qdrant_ama_kbqa` / `frontend_ama_kbqa` |
| Frontend image tag | `amakbqa-demo-frontend` | So a demo rebuild never overwrites the benchmark stack's `ama-kbqa-frontend` image |
| Virtuoso / Qdrant ports | `ports: !reset []` (not published on host) | Only Streamlit needs host exposure; avoids port collisions with the benchmark stack's `8890/1111/6335` |
| Streamlit | `127.0.0.1:8503` | Benchmark stack already owns `127.0.0.1:8502` |
| `qdrant` image | pinned `qdrant/qdrant:v1.17.1` | The version that originally wrote the demo's Qdrant volume; do not float `:latest` |
| `mem_limit` | virtuoso 768m, qdrant 1g, frontend 1536m | Box has 7.6 GiB RAM total, shared with ORCA + benchmark stack; unbounded containers risk starving ORCA |

## 3. Data

- Volumes `amakbqa-demo_ama_qdrant_data` (3.3 GB) and `amakbqa-demo_ama_virtuoso_data` (245 MB) were seeded via `cp -a` from the **stopped** benchmark-stack volumes `ama-kbqa_*`. Stop the benchmark stack before copying its volumes.
- Verified parity with bwcloud after the copy:
  - Qdrant: `kqapro-entities` 17,754 pts, `kqapro-relations` 363, `sciqa-entities` 171,588, `sciqa-relations` 8,596 (4096-dim, `qwen3-embedding-8b`). Hetzner's copies additionally carry a `bm25` sparse vector (harmless to the demo code — the frontend only reads the dense vector).
  - Virtuoso: graph `http://kqapro.org/kb` 1,609,805 triples, `http://sciqa.org/kg` 1,133,217 triples.
- Untracked runtime data (`db/datasets/kqapro/*.json`, `kb.nt`, SciQA Handcrafted/Autogenerated/HF, ORKG dump — 25 files, ~546 MB) was copied from `~/AMAKBQA-main` on the same box, size-matched against bwcloud's copies first.
- **Gotcha:** a bind-mounted path that doesn't exist on the host gets silently created as a **root-owned empty directory** by Docker on first `up`. Copy all runtime data into place *before* the first `docker compose up`, or `up` will "succeed" against an empty mount.

## 4. Model changes shipped with the migration

Cherry-picked from branch `demo-kit-models` onto `demo-hetzner` alongside the host move (unrelated to the host itself, but landed in the same deploy window):

- KIT retired `gemma4-31b-it` (the previous pinned default), `gpt-oss-120b`, `minimax-m2.7-229b`, `qwen3.5-397b-A17b`. Current local chat models: `kit.mistral-small-4-119b-a8b`, `kit.deepseek-v4-flash`, `kit.glm-5.3` (all do native tool calling).
- The model picker now **auto-discovers** local KIT chat models from `/models` (`connection_type` local, `kit.*` ids, no presets/aliases/hidden, non-null capabilities, keyword net excludes embedding/reranker/tts/stt/image) instead of a hardcoded whitelist. Display names come from the endpoint. `DEFAULT_MODEL_PREFERENCE` orders by responsiveness: Mistral Small 4, DeepSeek V4 Flash, GLM-5.3 — 2026-09-14 probes: Mistral ~7s, DeepSeek 1–47s erratic, GLM no answer within 120s.
- The picked model now reaches MCP subprocesses via env vars `AMA_KBQA_CHAT_MODEL` / `AMA_KBQA_CHAT_TEMPERATURE`, honored by `get_chat_model_name()` / `get_chat_temperature()`. Before this fix, the orchestrator probe's entity extraction read `config.toml` directly and failed with "Model not found" — this was live and broken on bwcloud.
- Fast-path synthetic tool-call ids are now 9-char alphanumeric. KIT's Mistral template truncates to the last 9 chars and rejects non-alnum, so the old scheme `fast_path_<n>_<tool>` truncated to e.g. `_FindNode` and got a 400 on the `QueryAttr`/`QueryRelation`/`QueryName` fast paths — also live and broken on bwcloud.

## 5. Tunnel

- Cloudflare tunnel `amakbqa-hetzner` (id in `deploy/hetzner/deploy.env` as
  `CLOUDFLARE_TUNNEL_ID`; `cloudflared tunnel list` also prints it).
- Config committed at `deploy/hetzner/cloudflared-amakbqa.yml`, installed to `~/.cloudflared/amakbqa.yml` on the server.
- Unit `deploy/hetzner/cloudflared-amakbqa.service` installed to `/etc/systemd/system` with `User=yannic` (not root — distinguishes it from ORCA's root-owned `cloudflared.service`).
- Ingress: `amakbqa.yanlab.de` and staging `amakbqa-next.yanlab.de` → `localhost:8503`. (§9: as of 2026-09-14, staging now points at `localhost:8504` instead — see below.)
- The old bwcloud tunnel `amakbqa` (id `21e852d7-...`) still exists in the Cloudflare account. DNS for `amakbqa.yanlab.de` was repointed with:
  ```bash
  cloudflared tunnel route dns --overwrite-dns amakbqa-hetzner amakbqa.yanlab.de
  ```
  Cutover: DNS moved to tunnel `amakbqa-hetzner` at 2026-09-14 09:22 UTC; verified live (public page shows the new picker label "Mistral Small 4"; `/_stcore/health` → 200).

## 6. Deploy procedure

```
local edit → commit on demo-hetzner → push
→ on server:
cd ~/amakbqa-demo
git pull --ff-only
DOCKER_BUILDKIT=0 docker build -f Dockerfile.frontend -t amakbqa-demo-frontend .
docker compose -f docker-compose.yml -f docker-compose.hetzner.yml up -d --no-build --force-recreate frontend
# wait for healthy
```

Notes:
- **Never edit files directly on the server.** Fix locally, commit, push, redeploy.
- `DOCKER_BUILDKIT=0` — this box has no buildx.
- **Always pass both `-f` files.** The base `docker-compose.yml` sets `name: ama-kbqa`, so a bare `docker compose up` in `~/amakbqa-demo` runs as the benchmark project and collides with its containers, ports and frontend image. Only the override's `name: amakbqa-demo` (and its container names, image tag and ports) keeps the demo separate.
- Only `frontend` needs `--force-recreate` on a normal code deploy; Virtuoso/Qdrant don't move.

### Smoke test

- One question per agent, **each in its own process** (`docker exec -i frontend_amakbqa_demo python -u - <model> <Agent>`). Running several agents inside one `asyncio.run` trips MCP `stdio_client` teardown across tasks ("Attempted to exit cancel scope in a different task") and kills the probe script. The demo itself is not affected (it runs each question in its own thread and event loop); this only bites test harnesses that batch agents into one loop.
- Plus a browser check on `amakbqa-next.yanlab.de` (staging ingress) before repointing the primary hostname, when doing anything riskier than a routine frontend rebuild.

### Rollback

- Previous frontend image, or `git checkout` the previous commit and rebuild.
- For a blue/green style rollout: bring up a second stack on another port, point `amakbqa-next` at it, verify, then swap `amakbqa` to match.

## 7. Relationship to the older Hetzner SOP

`SOP/hetzner_deployment.md` still accurately describes the **AMA-KBQA benchmark stack** (`~/AMAKBQA-main`, project `ama-kbqa`, ports 8890/1111/6335/8502, SSH-tunnel-only, no public demo) on this same box. It does **not** describe the public demo anymore — the demo moved off bwcloud onto this box as project `amakbqa-demo` (this document) on 2026-09-14. If you're looking for "how is the public demo deployed," use this document, not that one.

## 8. Open items (as of 2026-09-14)

- **Phase 2 documented, deploy not yet executed:** demo v2 (`dev` line + demo UI + federated orchestrator) — plan, staging layout, tunnel, and switch/rollback procedure are now written up in §9 below. Deployment execution to the Hetzner box itself is still pending; see §9's verification lines.
- `config.toml` synthesis/judge/fewshot models still point at `kit.glm-5.3` (slow as of 2026-09-14 probes). Only the synthesis path would affect the live demo, and it's currently disabled, so this is low urgency.
- `model_pricing.json` has no rows for the new KIT models (`kit.mistral-small-4-119b-a8b`, `kit.deepseek-v4-flash`, `kit.glm-5.3`) — the price caption is hidden for them rather than wrong, but it should be regenerated (see `System/demo_bwcloud_frontend.md` for the `fetch_model_pricing.py` regeneration command).
- Pre-existing, not caused by this migration, not fixed: an Orchestrator "MCP-Close error (RuntimeError)" warning at teardown, and a `GetEdgeQualifiers` tool-schema validation error logged by the KQAPro MCP server.
- Pre-existing, not caused by this migration, not fixed: `base_agent.py` has two unused imports (`Path`, `McpTool`) flagged by ruff F401, present at base commit `2917427`.
- The old bwcloud tunnel `amakbqa` (id `21e852d7-...`) can be deleted once bwcloud is decommissioned (`cloudflared tunnel delete amakbqa`) — not done yet, bwcloud isn't gone until 2026-09-15.
- **Deployed commit:** `8fee2fd` ("Fix Mistral 400 on fast-path tool calls"), deployed 2026-09-14 ~09:30 UTC. A follow-up test-only commit (fixing an import in `tests/framework/test_fast_path_tool_ids.py`) lands on top without a rebuild — it's test code, not shipped in the image.

### Post-cutover verification (2026-09-14, on `amakbqa.yanlab.de`, model Mistral Small 4)

| Check | Result |
|---|---|
| Orchestrator routing probe — "Einstein birthplace" | Ulm, 49–58s; routing probe no longer errors |
| KQAPro fast path — "Director of Inception" | Christopher Nolan, 47s (previously a 400 — the tool-call-id fix, §4) |
| SciQA — "COVID-19 detection" | Grounded answer, 33s |
| UI end-to-end via Cloudflare | Einstein → KQAPro, COVID-19 → SciQA routing, 58s |

---

## 9. Demo v2 (2026-09-14)

### Branch to stage

As of 2026-09-14 the branch to stage for v2 is **`demo-v2-graph`** once it
merges into `demo-v2-int` (it adds the live "Explored subgraph" side panel,
see [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md)
"Live Graph Panel" and
[Decisions/live-graph-two-source-subgraph.md](../Decisions/live-graph-two-source-subgraph.md)).
The setup commands below still say `demo-v2-int`; update the `git fetch`/
`worktree add` refs to whichever of `demo-v2-int` or a later integration
branch actually carries the merged graph-panel commits at deploy time. The
panel is **on by default**: `config.docker.toml`'s `[frontend] live_graph =
true` ships with the image, so no extra staging step is needed to enable it.
To disable it for a given deployment (e.g. to compare against the
pre-feature layout), set `AMA_FRONTEND_LIVE_GRAPH=0` in the compose
service's `environment:` block (same override mechanism as
`AMA_RETRIEVAL_RERANKER_ENABLED` below).

### Why

v1 (the `demo-hetzner` deploy documented in §1–§8 above) runs an old, pre-`dev`
backend — the `demo-bwcloud-deploy` base it was cut from predates the hybrid
retrieval package, the `chatkit` retry layer, the KG-adapter refactor, and the
one-round-trip Orchestrator routing that have since landed on `dev`. Demo v2 is
the **`dev` line + demo UI + federated orchestrator**: the demo frontend
(§ [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md)) rebuilt
on top of current `dev`, plus federated multi-specialist dispatch ported from
`feature/federated-retrieval` (see
[Decisions/federated-dispatch-and-fusion.md](../Decisions/federated-dispatch-and-fusion.md)).
It ships as a **second, standalone frontend staged next to v1**, not a v1
upgrade-in-place, so v1 keeps serving the public hostname unaffected until v2
is verified.

### Staging layout

A **second frontend only** — v2 does not run its own Virtuoso/Qdrant; it reuses
v1's already-seeded, already-running data services read-only.

- Compose project `amakbqa-next`, defined in a standalone
  `docker-compose.hetzner-next.yml` at the repo root (not an override file — it
  has no base `docker-compose.yml` layered under it).
- One service, `frontend`: container `frontend_amakbqa_next`, image
  `amakbqa-next-frontend`, published at `127.0.0.1:8504` (v1's frontend owns
  8503, the benchmark stack owns 8502), `mem_limit: 1536m`,
  `AMA_RETRIEVAL_RERANKER_ENABLED=false` set explicitly (belt-and-suspenders
  alongside `config.docker.toml`'s `[retrieval] reranker_enabled = false` — see
  "Reranker off rationale" below).
- Joins v1's compose network `amakbqa-demo_default` as an **external** network
  (`networks: default: name: amakbqa-demo_default, external: true`) and
  resolves v1's `qdrant`/`virtuoso` containers by v1's own service names — v2
  has no data containers of its own.
- Datasets and questionnaire JSONs are **bind-mounted read-only** straight from
  the v1 checkout on the host (`${DEMO_V1_DATA_ROOT}/db/...:...:ro`, set in
  `.env`) —
  v2 never needs its own copy.

**Setup, from a fresh checkout on the Hetzner box:**
```bash
git -C ~/amakbqa-demo fetch origin demo-v2-int
git -C ~/amakbqa-demo worktree add ~/amakbqa-next origin/demo-v2-int
cd ~/amakbqa-next
cp ~/amakbqa-demo/.env .env && chmod 600 .env
DOCKER_BUILDKIT=0 docker build -f Dockerfile.frontend -t amakbqa-next-frontend .
docker compose -f docker-compose.hetzner-next.yml up -d --no-build
```
(`docker-compose.hetzner-next.yml` also supports `up -d --build` directly, per
its own header comment — the explicit build step above matches this box's
`DOCKER_BUILDKIT=0` constraint from §6.)

### Tunnel

`deploy/hetzner/cloudflared-amakbqa.yml` on the v2 line adds
`amakbqa-next.yanlab.de → localhost:8504` while `amakbqa.yanlab.de` stays on
v1's `8503` (both hostnames share the one `amakbqa-hetzner` tunnel from §5,
which the same config file installs). Install and reload:
```bash
# copy the updated deploy/hetzner/cloudflared-amakbqa.yml to ~/.cloudflared/amakbqa.yml
sudo systemctl restart cloudflared-amakbqa
```
Both `amakbqa.yanlab.de` and `amakbqa-next.yanlab.de` drop for a few seconds
during the restart, since they share one tunnel process.

### Switch

Once v2 is verified on the staging hostname: point `amakbqa.yanlab.de`'s
ingress entry at `localhost:8504` in the same `cloudflared-amakbqa.yml`,
restart the tunnel (same command as above — again a few seconds of downtime
for both hostnames). Keep v1's frontend container running for a few days
after the switch as a fast rollback path, then `docker stop
frontend_amakbqa_demo`.

**Rollback:** point `amakbqa.yanlab.de`'s ingress back at `localhost:8503` and
restart the tunnel; `docker start frontend_amakbqa_demo` first if it was
already stopped.

### HAZARD

**Never run `docker compose ... down` on the v1 project (`amakbqa-demo`)
while v2 is running.** `down` removes the compose network
(`amakbqa-demo_default`) and the Virtuoso/Qdrant containers v2 depends on for
all its data access — it would take v2 down with it, not just v1. Only ever
`stop` or `restart <service>` on the v1 project while v2 is staged or live.

### Memory

Measured peak memory of the v2 frontend container per question (whole
container including the MCP server subprocesses; local run of the v2 image
against the same data on 2026-09-14, `docker stats` sampled every 2 s, so
sub-2 s spikes can be missed):

| Picker entry | Peak | Latency (Mistral Small 4) |
|---|---|---|
| Orchestrator (Router) | 331 MiB | 72 s |
| Orchestrator (Federated) | 467 MiB | 122 s |
| KQAPro | 209 MiB | 34 s |
| SciQA | 213 MiB | 94 s |

So one question at a time stays well inside the 1536m `mem_limit`; concurrent
visitors add roughly those amounts each. Box state on 2026-09-14 ~16:00 UTC:
about 3.7 GiB RAM available, swap 3.5/4 GiB used (ORCA's `mas-api` alone
holds ~2.4 GiB). Re-check with `free -m` / `docker stats` before staging.

### Reranker off rationale

`config.docker.toml`'s `[retrieval] reranker_enabled = false` (and v2's
explicit `AMA_RETRIEVAL_RERANKER_ENABLED=false` env override, belt-and-
suspenders) is deliberate, not an oversight: the demo image does not install
the `rerank` extra. With `reranker_enabled = true`, `ama_kbqa/retrieval/
search.py` re-raises the resulting `ImportError`, and `kqapro_server.py`
(~line 2076) swallows it into an empty semantic-search result instead of
surfacing an error — a silent degradation, not a loud failure. Enabling it
properly would cost (estimates, not measured on this box):
- ~1 GB of additional memory per specialist process (federated mode runs two
  specialists concurrently, so ~2 GB),
- several GB of `torch`/CUDA wheels added to the image,
- a 5–15 s cold model-load penalty per question, since the MCP servers
  respawn per question rather than staying warm.

Against that cost, the measured accuracy loss from turning it off is real but
modest: SciQA 77% → 73%, KQAPro 81% → 79% (run `rag-rerank-100q-2026-06-07`,
100 questions per dataset; cited in commit `a4abbd6` and the `[retrieval]`
comments of `config.toml` / `config.docker.toml`).
Given the Hetzner box's RAM budget (§ Memory above) and that this is a demo,
not the benchmark run the paper's numbers come from, leaving it off is the
right trade for this deployment.

### Local acceptance gotcha: reranker must be off without the extra installed

Running the v2 acceptance checks (§8 of the PRD, or any local
`uv run ama-kbqa-frontend` against the local containers) needs
`AMA_RETRIEVAL_RERANKER_ENABLED=false` in the shell environment **unless**
the `rerank` extra is installed (`uv sync --extra rerank`). Without either,
`ama_kbqa/retrieval/search.py` re-raises the resulting `ImportError` and
`kqapro_server.py` (~line 2076) swallows it into an empty semantic-search
result rather than surfacing an error: `FindResource` then errors out on
the SciQA side and the graph panel's SciQA half stays empty for the whole
run, which looks like a live-graph bug but is a retrieval-config gap. See
"Reranker off rationale" above for why the deployed image itself ships with
the reranker off (belt-and-suspenders `AMA_RETRIEVAL_RERANKER_ENABLED=false`
env plus `config.docker.toml`'s `[retrieval] reranker_enabled = false`);
this note is about a bare local dev environment that has neither the
container's env nor the extra installed.

### Smoke tests for v2

Use the committed `scripts/demo_smoke_ask.py`, one agent per process (the same
constraint as v1's smoke test in §6: several agents inside one `asyncio.run`
trip MCP `stdio_client` teardown across tasks). It applies the demo's default
KIT model the way the sidebar does, asks one fixed question per picker entry,
closes a specialist's MCP connection in the task that opened it (the
Orchestrator closes its own inside `ask()`), and exits non-zero on an empty
answer:
```bash
cd ~/amakbqa-next
for a in "Orchestrator (Router)" "Orchestrator (Federated)" KQAPro SciQA; do
  docker exec -i frontend_amakbqa_next python -u - "$a" < scripts/demo_smoke_ask.py
done
```
Pass a model id as the second argument to try another KIT model. The fixed
questions:
- Router: "In which city was Albert Einstein born?" (single-domain, should
  route to KQAPro).
- Federated: a deliberately cross-graph question (Inception's director plus ML
  evaluation benchmarks), so fan-out and fusion actually engage; the log should
  show `Routing successful -> federated: kqapro_agent, sciqa_agent` and
  `Fused answer from [...]`.
- KQAPro: "Who is the director of Inception?"
- SciQA: "What research contributions address COVID-19 detection?"

Plus a browser check on `amakbqa-next.yanlab.de`:
- All 4 picker entries present ("Orchestrator (Router)", "Orchestrator
  (Federated)", "KQAPro", "SciQA").
- The Federated run's lifecycle figure lights both specialist dispatch edges
  and the Answer Combination (fusion) node, not just one specialist.

### Federated notes

Federated mode is **experimental**: it costs roughly 2x the tokens of Router
mode (two specialist runs plus one fusion call), and the paper's routing-
accuracy and answer-accuracy numbers are all measured against **single-
dispatch** (Router mode). Federated mode exists for interactive demo use, not
as a benchmark claim — see
[Decisions/federated-dispatch-and-fusion.md](../Decisions/federated-dispatch-and-fusion.md)
"Trade-offs accepted".

The in-progress `feat/langgraph-rewrite` engine (not this line's default —
still `dev`'s classic `Orchestrator`) has **no federated dispatch path yet**.
Per that ADR's addendum, an `Orchestrator`-equivalent built on the LangGraph
engine must reject `federation=True` rather than silently falling back to
single dispatch, once federation is wired into it — silent fallback would let
a demo operator believe Federated mode is active when it isn't. Not relevant
to this deployment today (v2 runs the classic `Orchestrator`), but a
constraint for whoever ports federation onto that engine next.

### Verification

- Local verification (2026-09-14, image built from `ba81c3a`, local Qdrant/Virtuoso
  with production data, Mistral Small 4, reranker off): all four picker entries
  answered. Router: Ulm, routed to `kqapro_agent`, 72 s. Federated: dispatched to
  both specialists and fused (Christopher Nolan plus ML evaluation benchmarks),
  122 s. KQAPro: Christopher Nolan, 34 s. SciQA: grounded answer, 94 s. No
  "Model not found", no "Semantic search failed", no cancel-scope errors inside
  any agent run. `scripts/demo_smoke_ask.py` re-checked on KQAPro (50 s) and
  Router (53 s): both OK, exit 0, and no teardown traceback at `asyncio.run()`
  shutdown (the ad-hoc harness, which never closed the specialist, printed one
  there).
- Staging verification: <pending>

---

## 10. React demo frontend (2026-09-16) — local/Docker only, Hetzner deployment NOT done

A second frontend (`web/` + `ama_kbqa/api/`) shipped on `demo-v2-int`
(2026-09-16) for a booth/public audience, running next to Streamlit — see
[System/demo_react_frontend.md](../System/demo_react_frontend.md) and
[Decisions/react-frontend-second-ui.md](../Decisions/react-frontend-second-ui.md)
for what it is. It has been built and verified **locally in Docker only**
(`docker compose up -d --build api web` — see
[SOP/running_react_demo_locally.md](./running_react_demo_locally.md)). It has
**not** been deployed to the Hetzner box, staged, or given a public hostname.

### BLOCKER: the React demo's ports 2026/2027 fall inside ORCA's range on this box

`web`/`api` publish on **`2026`/`2027`** in the local `docker-compose.yml`
(changed from `8505`/`8506` on 2026-09-16, commit `a41ce3d`; host bindings
only — the `api` container still listens on `8506` internally). ORCA's
`mas-in-production` stack owns ports **2025–2030** on this box (see "What
happened and why" above), so **both** React ports sit inside a range another
production stack has already claimed.

**Scope: this is currently harmless.** Both bindings are `127.0.0.1`-only and
exist only on the developer machine; nothing from this line is deployed to
Hetzner, so nothing is colliding today. The collision only materialises *if
and when* the React frontend is rolled out to this host.

**It must be resolved before a Hetzner rollout begins, not during one.**
Re-check what ORCA currently binds, pick a free pair for `web`/`api`, and
record the claim in this section the way §9 recorded `8504` — before building
on the host, not after. The fix is to move the demo's ports; ORCA is the
incumbent production stack and does not move. The `2026`/`2027` default was
chosen for a developer laptop, not for this box.

Deploying it there (either as a v3 staging line following §9's pattern, or
folded into the v2 rollout) would need, at minimum:

- **A port claim** — see the BLOCKER above, which must be closed first. The
  Hetzner box's occupied `127.0.0.1` ports as of this SOP: `1111`/`8890`
  (Virtuoso), `6335` (Qdrant), `8502` (benchmark-stack Streamlit), `8503` (v1
  public Streamlit), `8504` (v2 staging Streamlit, see §9).
- **A cloudflared ingress rule.** `deploy/hetzner/cloudflared-amakbqa.yml`
  would need a new hostname entry (e.g. `amakbqa-web.yanlab.de →
  localhost:<claimed web port>`) alongside the existing `amakbqa.yanlab.de` /
  `amakbqa-next.yanlab.de` entries from §5/§9, and the same
  `sudo systemctl restart cloudflared-amakbqa` to pick it up. The React app's
  own nginx (`web/nginx.conf`) already sets `proxy_buffering off` + `gzip off`
  on `/api/` for SSE; whatever sits in front of it on the Hetzner side
  (cloudflared, and Cloudflare's edge itself) must not re-introduce buffering
  on that path, or live run updates will stall or arrive in bursts.
- Compose service definitions for `api`/`web` in whatever
  `docker-compose.hetzner*.yml` override file is used — none exist yet; the
  local `docker-compose.yml` `api`/`web` blocks are the starting point but
  were not written with the Hetzner box's `DOCKER_BUILDKIT=0` constraint (§6)
  or the v2-staging external-network pattern (§9 "Staging layout") in mind.

Until this section is updated with an actual deploy, treat the React frontend
as **local-only**.

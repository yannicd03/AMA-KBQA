---
type: decision
status: accepted
date: 2026-09-20
summary: Host-specific deployment values (origin address, Cloudflare tunnel id, deploy user, home paths) moved out of tracked files into a gitignored deploy.env, rendered into the real config via committed .template files and render-deploy-config.sh, because the repo is about to go public and the origin address in particular would let traffic bypass the Cloudflare proxy.
addresses: [issue/repo-going-public-leaks-host-details]
affects: [deploy/hetzner, System/demo_bwcloud_frontend, SOP/hetzner_demo_deployment, SOP/hetzner_deployment]
relates: [demo-bwcloud-frontend-divergence]
evidence:
  - "commit 4c4a5a0"
  - "deploy/hetzner/deploy.env.example"
  - "deploy/hetzner/render-deploy-config.sh"
  - "deploy/hetzner/cloudflared-amakbqa.yml.template"
  - ".gitignore"
---

# ADR: Deployment Host Details Move to a Gitignored `deploy.env`, Rendered Through Committed Templates

**Status:** Accepted (shipped commit `4c4a5a0`, "Move deployment host details out of the repo and into config").

## Related Docs
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) — operator runbook, now points at this flow up top
- [SOP/hetzner_deployment.md](../SOP/hetzner_deployment.md) — older benchmark-stack SOP, one absolute path parameterized
- [Decisions/demo-bwcloud-frontend-divergence.md](./demo-bwcloud-frontend-divergence.md) — the demo build this deployment serves
- [System/project_architecture.md](../System/project_architecture.md) §8 CI/Linting — a ruff gotcha hit while fixing a related hardcoded-path bug, recorded there

---

## Context

The repo (`yannicd03/AMA-KBQA`) is about to go public. Committed files hardcoded host-specific values for the public demo's Hetzner deployment, in two distinct places:

Config files:
- `deploy/hetzner/cloudflared-amakbqa.yml` — the **tunnel UUID**, and the credentials path under the deploy user's home directory. Its ingress entries point at `localhost` ports, not at an IP.
- `deploy/hetzner/cloudflared-amakbqa.service` — the deploy user and home path.
- `docker-compose.hetzner-next.yml` — an absolute host path to v1's already-seeded data.

Documentation:
- `SOP/hetzner_demo_deployment.md` — the **origin IP addresses** of the Hetzner box and the retired bwcloud host, the ssh alias, the deploy user, and the tunnel UUID again in prose.

The origin address is the sharpest risk, and it lived in the SOP rather than in any config file. A cloudflared tunnel exists precisely so the origin is never addressed directly: DNS is Cloudflare-proxied and resolves to Cloudflare, so the box's own IP is not otherwise discoverable. Publishing that IP hands anyone the ability to skip the proxy and reach the origin directly. The tunnel id, deploy user and home path are lower-severity but still identify the specific box and account.

Both places had to be fixed, and they were fixed differently: the config files are now rendered from templates, while the SOP references `deploy.env` instead of quoting values.

## Decision

Host-specific values now live in `deploy/hetzner/deploy.env`, gitignored, with `deploy/hetzner/deploy.env.example` committed as the documented template (every field, what it's for, a placeholder value). `deploy/hetzner/render-deploy-config.sh` (committed, executable) sources `deploy.env`, validates every required variable is present, **refuses to run if `CLOUDFLARE_TUNNEL_ID` is still the placeholder**, and renders `cloudflared-amakbqa.yml` and `cloudflared-amakbqa.service` from committed `.template` files using `envsubst` restricted to the variables the script owns (`${CLOUDFLARE_TUNNEL_ID}`, `${DEMO_HOSTNAME}`, `${DEMO_STAGING_HOSTNAME}`, `${DEMO_PORT}`, `${DEMO_STAGING_PORT}`, `${DEPLOY_USER}`, `${CLOUDFLARE_CREDENTIALS_DIR}`) so unrelated `$`-text in the templates is not blanked out by a bare `envsubst`.

Both rendered files, plus `deploy.env` itself, are now gitignored and untracked. The Cloudflare tunnel *credentials JSON* was already server-only (never in the repo) and is untouched by this change.

`docker-compose.hetzner-next.yml`'s v1 data-root mount now reads `${DEMO_V1_DATA_ROOT:?...}` (documented in `.env_example`); compose fails fast with a pointer to `.env_example` when the variable is unset, rather than silently mounting nothing or the wrong path.

**Operator flow, from a fresh clone:** copy `deploy/hetzner/deploy.env.example` to `deploy/hetzner/deploy.env`, fill in the real values, run `./deploy/hetzner/render-deploy-config.sh`, then continue with the existing steps in `SOP/hetzner_demo_deployment.md`.

## Alternatives considered

- **Keep values in the repo, mark the repo private instead.** Rejected — the whole point of the change in flight is publishing the repo (paper PURL, SEMANTiCS 2026 demo track); privacy was never on the table.
- **Secrets manager / vault.** Overkill for a single-operator personal deployment with no team secret-sharing requirement; a gitignored env file plus a documented example is the same pattern the project already uses for `.env`/`.env_example`, extended rather than replaced.
- **`.env`-style single file with no template rendering** (just gitignore the real `cloudflared-amakbqa.yml`/`.service` and hand-edit them per host). Rejected: cloudflared and systemd read those files directly, so hand-editing them per host loses the single source of truth `deploy.env` gives and makes it easy for a stale rendered file to silently diverge from `deploy.env`. Rendering from `.template` + `envsubst` keeps one place to change a value.

## What was deliberately left as prose, not parameterized

**The public demo hostnames (`amakbqa.yanlab.de`, `amakbqa-next.yanlab.de`) stay in the SOP and in this repo's docs/paper references.** They are not secret — they are the published URL behind the paper's PURL — and scrubbing them out of the runbooks would make the runbooks unusable without first reconstructing the hostnames from `deploy.env` by hand. They are still fully parameterized in every actual config file (`DEMO_HOSTNAME`, `DEMO_STAGING_HOSTNAME` in `deploy.env`); only the human-facing prose keeps them literal.

## Related: what else was in scope for the public-repo prep, and wasn't touched here

The Phase 5 run manifests under `.agent/Tasks/active/langgraph-rewrite-phase5/` (not present in this worktree — lives on the `feat/langgraph-rewrite` line, same cross-branch situation as `Tasks/active/langgraph-rewrite.md` itself, see `Decisions/react-frontend-second-ui.md`) contain `cwd` and invocation paths with the local username and were **deliberately not scrubbed** as part of this same public-repo prep effort. Per [SOP/refreshing_paper_numbers.md](../SOP/refreshing_paper_numbers.md), those manifests are the authoritative provenance record for the paper's benchmark numbers; rewriting them would falsify evidence. This is a different risk class from the deploy config above — a local username in a benchmark log is not a network-access vector — and the two should not be conflated when someone next audits the repo for public-readiness.

## Verification

Rendered output is byte-identical to the previously-committed `cloudflared-amakbqa.yml`/`.service` on every operative line (confirmed by diff against the pre-change committed versions), so the running deployment on the Hetzner box is unaffected by this change. `docker-compose.hetzner-next.yml` validates and substitutes `${DEMO_V1_DATA_ROOT}` correctly, and fails loudly (not silently) when the variable is unset. Full repo `ruff check .` is clean; 1137 tests pass.

## Threshold to revisit

If the project ever needs multiple people deploying to different hosts, or if `deploy.env` starts drifting from what's actually on the server without anyone noticing, that's the trigger to move to something with drift detection (e.g. a checked-in encrypted secrets file, or state tracked by the deploy tooling itself) rather than a plain gitignored env file trusted by convention.

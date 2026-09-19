# ADR 0001: Three-branch demo split (`demo-v2-int` / `demo-public` / `demo-booth`)

## Related Docs
- [SOP/hetzner_demo_deployment.md](../SOP/hetzner_demo_deployment.md) §9–§10 — deployment mechanics for all three branches
- [System/demo_bwcloud_frontend.md](../System/demo_bwcloud_frontend.md) "Model picker (booth build)" — the feature this split exists to isolate
- [Decisions/0002-deepseek-thinking-mode-disabled.md](0002-deepseek-thinking-mode-disabled.md) — a `demo-booth`-only fix downstream of this split
- [Decisions/demo-bwcloud-frontend-divergence.md](demo-bwcloud-frontend-divergence.md) — why the demo build diverges from the full app in the first place
- [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md) — federated dispatch, part of the shared `demo-v2-int` base
- `Tasks/active/booth-demo-providers.md` — PRD that made this decision (§1)

## Status
Accepted, 2026-09-14. Implemented same day (commit `f35b2a7`).

## Context

By 2026-09-14, `demo-v2-int` was the integration base for the next public
demo: the `dev` backend line, the demo UI, federated multi-specialist
dispatch, and the live "Explored subgraph" panel, merged at `ca90fe4` (see
`SOP/hetzner_demo_deployment.md` §9). It was staged as `amakbqa-next` on the
shared Hetzner box, intended to eventually replace the v1 frontend serving
`amakbqa.yanlab.de`.

Separately, the team needed a demo build for the SEMANTiCS 2026 booth
(Wednesday 2026-09-16) that lets the presenter switch the chat model between
KIT and billed third-party endpoints (OpenRouter, DeepSeek direct) — useful
on stage to show off stronger models, using real API keys and relaxed
session rate limits appropriate for one operator asking many questions back
to back, not public traffic.

These two things could not both live in the same deployable artifact:
- The public-facing build must never expose a way to route traffic through
  paid third-party APIs — the public demo is KIT-funded (free to the
  project) and unmetered per-visitor; a public build with an OpenRouter
  picker entry is a standing cost-abuse surface.
- The booth build's relaxed rate limits (`DEMO_MAX_QUERIES_PER_SESSION=500`,
  `DEMO_MIN_SECONDS_BETWEEN_QUERIES=1`) are wrong for public traffic for the
  same reason the existing stricter defaults exist.
- The booth needs real API keys on the host; the public deployment should
  not need to carry keys it never uses.

## Decision

Split into three branches:

| Branch | Role | Deployed where |
|---|---|---|
| `demo-v2-int` | Shared integration base. Backend/UI/federation/graph-panel fixes land here first. | Nowhere directly. |
| `demo-public` | Public-facing demo. KIT endpoint only, public rate limits. Identical to `demo-v2-int` today (no branch-specific changes yet). | Not on Hetzner for the near future (2026-09-14 decision) — the existing v1 stack (`amakbqa-demo`, `SOP/hetzner_demo_deployment.md` §1–§8) keeps serving `amakbqa.yanlab.de` until `demo-public` is actually staged there. |
| `demo-booth` | Booth demo. Adds OpenRouter + DeepSeek-direct chat endpoints to the model picker, relaxed session rate limits, booth-specific About-dialog copy. | Hetzner, compose project `amakbqa-next` (port 8504, `amakbqa-next.yanlab.de`), from 2026-09-16. |

Rule: shared fixes land on `demo-v2-int` and are merged forward into both
`demo-public` and `demo-booth`. Booth-only changes (billed-provider plumbing,
rate-limit relaxation) never land on `demo-v2-int` or `demo-public`;
public-only changes never land on `demo-booth`.

## Alternatives considered

**One branch (`demo-v2-int` itself) with a feature flag gating the billed
providers.** Rejected: a flag is a runtime toggle, and the risk here is not
"the feature is visible" but "the feature ships in the artifact that gets
deployed to the public hostname." A flag defaulting off is one config edit
away from an accidental public exposure of billed-provider routing (e.g. a
copy-paste of `docker-compose.hetzner-next.yml`'s env block into the public
compose file, or a default flip during a later refactor). A separate branch
makes the exposure structurally impossible: the public build's source simply
does not contain the OpenRouter/DeepSeek plumbing.

**Two branches (fold `demo-public` into `demo-v2-int` and treat `demo-v2-int`
itself as deployable).** Considered, but rejected for now because the
integration base needs to keep accepting in-flight, not-yet-verified changes
(this PRD's own commit landed the same day as the graph-panel merge), while
"what's live at the public hostname" needs a stable, deliberately-cut point.
Collapsing the two would mean either integration work pauses to keep
`demo-v2-int` stage-ready at all times, or the public deploy risks picking up
partially-verified work. A three-way split keeps the fast-moving base and the
two deployable, provenance-tracked artifacts separate. (`demo-public` is
currently just a pointer to `ca90fe4` with no divergence yet — it may in
practice stay identical to `demo-v2-int` for a while, but the branch exists
so that changes, when they come, can be made deliberately rather than by
whatever happened to be on the integration branch at deploy time.)

## Consequences

- Three branches to keep in sync manually via merge-forward; no automated
  merge-train tooling set up for this (small team, short-lived need — the
  booth demo is a one-week artifact).
- `demo-booth`'s booth-only commits (this PRD) must never be cherry-picked
  onto `demo-public` without deliberate review — the whole point of the split
  is that this doesn't happen by accident.
- Every `.agent/System` doc shared across the three branches (e.g.
  `demo_bwcloud_frontend.md`) now needs branch-scoping language for anything
  that differs between them (see that doc's "Model picker (booth build)"
  section for the pattern used: state the general behavior, then call out
  what's `demo-booth`-only).
- If `demo-public` is later staged on Hetzner and starts diverging from
  `demo-v2-int` in its own right (not just by construction), this ADR should
  be revisited to record what that divergence is and why.

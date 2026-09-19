# ADR 0004: Consolidate the demo lines back into `dev`

## Related Docs
- [Decisions/0001-three-branch-demo-split.md](0001-three-branch-demo-split.md) — the split this ADR partially supersedes (see the dated note at that file's top)
- [Decisions/federated-dispatch-and-fusion.md](federated-dispatch-and-fusion.md) — the federated dispatch ADR; its 2026-09-14 addendum is the hand-port this ADR supersedes `feature/federated-retrieval` with
- [Decisions/langgraph-adoption.md](langgraph-adoption.md) / [Tasks/active/langgraph-rewrite.md](../Tasks/active/langgraph-rewrite.md) — the LangGraph rewrite line folded in by this merge
- [Decisions/react-frontend-second-ui.md](react-frontend-second-ui.md) — 2026-09-20 addendum records the resolution of its flagged `fastapi`/`uvicorn` cross-branch conflict
- [Decisions/demo-picker-provider-routing.md](demo-picker-provider-routing.md) — the `demo-booth` picker this consolidation keeps
- [System/graph_engine.md](../System/graph_engine.md) — "What the graph engine does not cover yet", the two gaps this merge exposed

## Status

Decided 2026-09-20. Landed on `integrate/dev-consolidation` (tip `12e55f2` at
time of writing) via three merges: `feat/langgraph-rewrite` (`c37bd1a`, which
also carries `feat/exact-lookup-tool`), `demo-booth` (`3ce7f00`), and the rest
of `demo-v2-int` (`bf9c464`). 1289 tests pass, ruff clean.

**`dev` was fast-forwarded onto this work the same day (`3a1bab4` to
`12e55f2`). Nothing has been pushed to either remote.** Publishing to the
public repo is a separate step: the new commits get replayed onto its
sanitized lineage, never pushed as-is.

## Context

By 2026-09-20 three independent lines of work needed reconciling:

1. `feat/langgraph-rewrite` — the LangGraph `StateGraph` engine (see
   `Tasks/active/langgraph-rewrite.md`), developed in its own worktree,
   Phases 0-5 complete but not merged anywhere.
2. `demo-v2-int` and its children (`demo-booth`, `demo-public`,
   `demo-llamacpp`) — the demo frontend line, including federated dispatch
   (ported from `feature/federated-retrieval`, see
   `Decisions/federated-dispatch-and-fusion.md`'s 2026-09-14 addendum), the
   provider-aware model picker, the live subgraph panel, and the React/FastAPI
   second frontend.
3. SEMANTiCS 2026 (the booth event `demo-booth` was built for) is over as of
   this writing — the booth-specific branch no longer needs to stay isolated
   from `dev` for conference-readiness reasons.

`Decisions/0001-three-branch-demo-split.md` split `demo-v2-int` into
`demo-public` and `demo-booth` specifically to keep billed third-party
provider routing out of the public-facing build. That constraint doesn't
disappear, but the *branch topology* serving it can change now that there is
no more conference deadline forcing `demo-booth` to stay separate from
mainline development, and no more standalone public deployment plan that
needs `demo-public` to exist as a distinct branch (see Decision below).

## Decision

**1. The demo line collapses into `dev`.** Rather than three long-lived
demo branches forking off an integration branch that itself never merges to
`dev`, the demo frontend, federated dispatch, and the LangGraph engine all
become part of mainline `dev` history. Future demo-only work (if any) can be
a feature branch off `dev` like anything else, not a parallel long-lived line.

**2. `demo-booth` is the variant kept; `demo-llamacpp` is dropped;
`demo-public` is deprecated.**

- `demo-booth`'s content — the OpenRouter/DeepSeek-direct model picker
  (`Decisions/demo-picker-provider-routing.md`), relaxed session rate limits,
  booth-specific About-dialog copy — is the version that survives the merge.
  It is strictly more capable than `demo-public` (superset of provider
  options) and is real production content already exercised at the booth,
  unlike `demo-llamacpp` (an experimental local-inference variant that never
  shipped anywhere) or `demo-public`'s divergence from `demo-v2-int` (per
  `Decisions/0001-three-branch-demo-split.md`, it "may in practice stay
  identical to `demo-v2-int` for a while" — it did; it never diverged).
- `demo-llamacpp` is dropped outright: no deployment, no unique content worth
  preserving beyond git history.
- `demo-public` is **deprecated, not deleted from history** — its only
  actual content beyond the shared base was the public-vs-booth distinction
  itself, which is now expressed as a config value rather than a branch: the
  `[frontend].settings_level = "minimal"` setting (`config.py:1232
  get_frontend_settings_level()`, `ama_kbqa/api/meta.py:788-792`) remains a
  fully supported config value on `dev` — a future public-facing deployment
  sets that config key on the consolidated branch rather than checking out a
  separate branch. The billed-provider isolation concern from ADR 0001 is now
  a deployment-time config choice (which `.env`/`config.toml` a given host
  runs with), not a branch-structural one — acceptable because there is no
  currently-active public deployment plan forcing the stronger,
  structurally-can't-happen guarantee ADR 0001 chose branches for.

**3. `feature/federated-retrieval` is superseded by the demo line's hand-port
(`fc8e365`), not merged directly.** `feature/federated-retrieval` (commits
`034dddc`/`7195f60`/`67eb6b7`/`98882ac`, forked from an old `main`) predates
`dev`'s one-round-trip routing rework and was never fast-forwardable onto it.
`fc8e365` (`Decisions/federated-dispatch-and-fusion.md`'s 2026-09-14
addendum) rebuilt the same federation feature by hand on top of `dev`'s
routing mechanics instead of cherry-picking and resolving conflicts against
the original branch. This consolidation keeps that hand-port and does not
also pull in `feature/federated-retrieval` — the two implement the same
feature and only one is needed.

**Why the hand-port over the original branch:** the hand-port's Router mode
sends `dev`'s exact pre-federation `select_agent` tool contract
byte-for-byte (schema and prompt unchanged, pinned by a snapshot test), so
the paper's routing-accuracy benchmark numbers stay measured against an
unchanged contract. `feature/federated-retrieval`'s original design always
sent the array-typed `select_agents` tool (capped at `maxItems=1` when
federation is disabled), which is behaviorally equivalent for a compliant
provider but is a different wire contract — a risk the hand-port deliberately
avoided. The hand-port also costs one LLM round-trip for routing (the probe
is a direct MCP call, not a forced tool call) where the original branch's
design cost two (a forced probe-tool call, then a forced decision call) —
see `System/orchestrator_routing.md`'s Routing Flow section for the
one-round-trip mechanics both modes now share.

## Consequences

- `dev` now carries the demo frontend, federated dispatch and the LangGraph
  engine together. The multi-page Streamlit app (`frontend/pages/*`,
  `utils/settings_ui.py`) is gone with it, which the user accepted explicitly;
  benchmarking stays available through the CLI.
- Two known gaps block deleting the legacy (non-graph) agent loop, found
  while merging: the graph engine's orchestrator models Router mode only
  (a Federated instance falls back to the legacy body), and neither graph
  has a cooperative-cancellation checkpoint. Both are recorded in
  `Tasks/active/langgraph-rewrite.md` §16 and `System/graph_engine.md`'s
  "What the graph engine does not cover yet" section — not fixed by this
  merge, just now visible because the demo line (which depends on both
  Federated mode and Stop-button cancellation) and the LangGraph engine
  share a codebase for the first time.
- `demo-public`'s branch is dead weight (nothing should be built on it going
  forward); its one own change is `settings_level = "minimal"`, a supported
  config value on `dev`.
- `demo-llamacpp` was deleted on 2026-09-20 at the user's request, unported
  (last tip `76de335`).
- The absorbed branches (`feat/langgraph-rewrite`, `demo-v2-int`,
  `demo-booth`, `chore/scrub-local-paths`, and the `demo-v2*` ancestors) were
  deleted locally the same day, each verified as an ancestor of `dev` first.
- `feature/federated-retrieval` becomes dead weight now that its
  functionality lives in `dev` via the hand-port; its own commits are not
  part of `dev`'s history, so anyone auditing federation's implementation
  history should read `Decisions/federated-dispatch-and-fusion.md`'s
  addendum (the hand-port's ADR), not the original branch's commits.

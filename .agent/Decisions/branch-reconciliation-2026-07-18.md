# Branch Reconciliation: `dev-routing-evidence` → `dev` (2026-07-18)

## Context

`dev-routing-evidence` predated a history-scrub/rebuild of `dev` (merge-base with
`origin/dev` is `747ead6`). By 2026-07-18 the two branches had diverged hard:
`dev-routing-evidence` was 204 commits ahead of the merge-base, `origin/dev` was
229 commits ahead. PR #3 (`dev-routing-evidence` → `dev`, opened after the
2026-07-05 architecture audit, see
[architecture-audit-2026-07-05.md](architecture-audit-2026-07-05.md)) showed as
`CONFLICTING`/`DIRTY` — a normal merge would have dragged in ~200 divergent,
mostly-already-superseded commits.

## Investigation

`git cherry origin/dev dev-routing-evidence` showed `origin/dev` already
contains every "keeper" commit from `dev-routing-evidence` under new hashes —
the scrubbed-history rebuild re-landed them independently:

| Change | `dev-routing-evidence` | `origin/dev` |
|---|---|---|
| Evidence-based orchestrator routing (see [orchestrator-evidence-based-routing.md](orchestrator-evidence-based-routing.md)) | `ab79f62` | `1628816` |
| Multiturn direct-agent conversation (see [multiturn-direct-agent-conversation.md](multiturn-direct-agent-conversation.md)) | `81fbdf9` | `beb7703` |
| Graceful Qdrant degradation | `9d902e9` | `1a9e298` |
| Hide follow-up chat box for stateless Orchestrator | `b3347d5` | `267b38c` |
| Routing docs (ADR + system doc + changelog) | `038ab7f` | `dab0a44` |

The **only** commit on `dev-routing-evidence` with no counterpart on `dev` was
`e659fc6` ("Architecture audit fixes: synthesis funnel, routing latency,
benchmark concurrency" — the 2026-07-05 audit fixes).

A naive wholesale `git diff origin/dev dev-routing-evidence | git apply` was
considered and rejected: because `dev-routing-evidence` predates several
modules `dev` has since grown (`ama_kbqa/retrieval/` — embeddings/search/reranker,
`framework/deterministic.py`, `framework/operations.py`,
`utils/artifact_golds.py`), a wholesale diff would have **deleted** all of them.

## Decision

Cut a fresh branch `reconcile/routing-onto-dev` from `origin/dev` and
surgically ported only the non-overlapping parts of `e659fc6`, committed as
`7b33463`. This became PR #4 (`reconcile/routing-onto-dev` → `dev`), which
opened `MERGEABLE`/`CLEAN`. PR #3 was closed as superseded.

### Ported (new to `dev`)

- `base_agent.py` synthesis-funnel guards (B1/B2) + new test
  `tests/framework/test_base_agent_synthesis_guards.py`
- `mcp_client.py` close-delay trim (C6a-adjacent): the per-close `asyncio.sleep`
  dead time (0.5s + 0.2s) dropped to a single 0.05s yield — paid on every MCP
  close, notably the orchestrator's per-question probe-server teardown. Note: the
  A1 *client dedup* did NOT land here — it lived inside the dropped routing
  rewrite (see below), so A1 remains open.
- `benchmark_agents.py` opt-in question-level parallelism (C1): added
  `concurrency` param to `run_benchmark_for_model_agent` and a new
  `run_benchmark_for_model_agent_parallel`; default `concurrency=1` keeps the
  existing serial path as the default behavior
- `config.py` / `config.toml` additions
- `kqapro_server.py` C6c schema-attribute embed cache (memoize the static
  distinct-attribute list + their embeddings once per server lifetime). Fixup
  during review: the cherry-picked call used the old branch's `get_embedding()`
  helper, which does not exist on `dev` (dev moved embedding into
  `retrieval.embed_query`); repointed to `retrieval.embed_query`, matching the
  file's two other call sites. Without the fix the fuzzy `GetSchemaForAttribute`
  path would `NameError` silently (swallowed by a broad `except`).
- `.gitignore` (H2)
- ADR `Decisions/architecture-audit-2026-07-05.md` +
  `Tasks/active/deferred-tool-loading.md` (already carried over in an earlier
  port; unaffected by this reconciliation)

### Dropped (dev already solved differently — kept `dev`'s version via `git checkout --ours`)

- `orchestrator_agent/agent.py` + `server/orchestrator_server.py`
  routing-latency rewrite — superseded by `dev`'s later routing-fold commit
  `f0827ae`
- `sciqa_server.py` + `tests/agents/test_orchestrator_routing.py` — `dev` has
  later, independent fixes in this area
- `tests/server/test_sciqa_server.py` — asserted a module-level
  `sciqa_server._embedding_cache` that `dev`'s `retrieval/` module replaced;
  `dev` covers the equivalent behavior via `tests/retrieval/test_embeddings.py`

## Non-obvious gotcha

C1 (benchmark question-level parallelism, `benchmark_agents.py`) is
**orthogonal** to C4 (concurrent tool calls within a single agent run, `dev`'s
`5b38559`) — they parallelize different levels (across questions vs. within
one agent's tool loop) and do not duplicate or conflict.

## Verification

Full suite: 382 passed on `reconcile/routing-onto-dev`. PR #4 opened
`MERGEABLE`/`CLEAN`.

## Related Docs

- [architecture-audit-2026-07-05.md](architecture-audit-2026-07-05.md) — source of the ported `e659fc6` fixes
- [orchestrator-evidence-based-routing.md](orchestrator-evidence-based-routing.md) — routing rework already present on `dev` under a different hash
- [multiturn-direct-agent-conversation.md](multiturn-direct-agent-conversation.md) — multiturn feature already present on `dev` under a different hash

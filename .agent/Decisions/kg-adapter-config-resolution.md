# ADR: KG Adapter Is the Runtime Source of Truth for Endpoint/Collection Config

**Date:** 2026-07-24
**Commit:** `921a04d` (branch `dev`)
**Status:** Accepted

## Related Docs
- [Project Architecture](../System/project_architecture.md) — Configuration System section, directory tree
- [Decisions/abstract-operation-contract.md](abstract-operation-contract.md) — 2026-06-06 drift audit that found the same class of defect (paper claim vs. code) in a different part of the adapter contract
- [Decisions/hybrid-retrieval-architecture.md](hybrid-retrieval-architecture.md) — prior art for the env-overlay-over-config.toml pattern (`AMA_RETRIEVAL_*`)

---

## Context

`GraphConfig` and `VectorConfig` (`ama_kbqa/framework/config.py`) were populated by each KG adapter's `_create_config()`, but nothing at runtime read them. The two MCP servers instead pulled endpoint, named graph, and vector-collection values from per-KG getter functions in `ama_kbqa/config.py`: `get_virtuoso_endpoint`, `get_collection_entities`, `get_collection_relations` for KQAPro, and the SciQA-specific duplicates `get_sciqa_collection_entities` / `get_sciqa_collection_relations` / `get_sciqa_virtuoso_graph`.

Consequence: onboarding a new KG required hand-adding a new set of getters to `ama_kbqa/config.py`, which is exactly the per-KG boilerplate the adapter abstraction was supposed to eliminate.

This was discovered during a camera-ready audit of the SEMANTiCS 2026 paper, which claims "a hierarchy of dataclasses declares everything the reasoning core needs to know about a graph." That claim was false in code for everything except `namespaces` and `domain_settings` — the same "aspirational claim, unimplemented in code" defect class as the 2026-06-06 operation-binding audit (`Decisions/abstract-operation-contract.md`), just in the config layer instead of the tool-binding layer.

---

## Decision: adapters declare defaults, a new resolution layer applies deployment overrides, servers read `resolved_config`

**Chosen:**
- New `ama_kbqa/framework/adapters/resolve.py` exposes `resolve_graph_config(code, default)` and `resolve_vector_config(code, default)`.
- Resolution precedence per field, highest wins:
  1. `AMA_KBQA_<CODE>_<FIELD>` env var (e.g. `AMA_KBQA_SCIQA_ENDPOINT`) — generic channel, works for any adapter code including ones added later.
  2. `[kg.<code>]` section in `config.toml` — generic, the extension point new KGs use.
  3. Legacy KG-specific `config.toml` keys — a **closed table** (`_LEGACY_GRAPH_KEYS` / `_LEGACY_VECTOR_KEYS` in `resolve.py`) covering only `kqapro` and `sciqa`, kept so existing `config.toml` / `config.docker.toml` deployments keep working unmodified. New KGs must not be added to this table; they use tier 1 or 2.
  4. The adapter's own declared default (`_create_config()`).
- `BaseKGAdapter.resolved_config` (new cached property in `ama_kbqa/framework/adapters/base_adapter.py`) returns the KG's `KnowledgeGraphConfig` with `graph` and `vectors` run through the resolver. `.config` still returns the adapter's pure declared defaults — `namespaces`, `prompts`, and `domain_settings` are KG constants that are never overridden by deployment, so they keep coming from `.config`.
- Both MCP servers (`ama_kbqa/server/kqapro_server.py`, `ama_kbqa/server/sciqa_server.py`) now derive `VIRTUOSO_ENDPOINT`, `COLLECTION_ENTITIES`, `COLLECTION_RELATIONS`, and (SciQA only) `SCIQA_GRAPH` from `_ADAPTER.resolved_config` at module load, instead of calling the `ama_kbqa.config` getters.
- This introduces the first `AMA_KBQA_*` env vars in the codebase, following the naming pattern already established by `AMA_RETRIEVAL_*` (`Decisions/hybrid-retrieval-architecture.md`) and `AMA_LLM_SEED`.

**Rejected:**
- *Wire up `supports_reification` / `has_temporal_data` instead of just resolving endpoint/collections*: these two `GraphConfig` fields had zero runtime readers. The genuine KQAPro-vs-SciQA reification/temporal difference is implemented as separate hardcoded SPARQL patterns directly in `kqapro_server.py` (reification, temporal filters) with no counterpart in `sciqa_server.py` — a code-path difference, not a shared code path branching on a flag. Wiring the flags to actually gate behavior would require restructuring both servers into one shared implementation, which is out of scope for a config-resolution fix and carries its own risk. **They were removed from `GraphConfig` rather than left dead a second time**; a comment in `ama_kbqa/framework/config.py` records why, in case someone is tempted to re-add them for a future third KG.
- *Immediately deleting the six legacy getters*: `get_virtuoso_endpoint`, `get_collection_entities`, `get_collection_relations`, `get_sciqa_collection_entities`, `get_sciqa_collection_relations`, `get_sciqa_virtuoso_graph` are still called by `ama_kbqa/utils/artifact_golds.py` and `db/migrate_add_bm25.py`. They are kept, with deprecation docstrings pointing at `resolved_config`, rather than migrating those two call sites in the same change.
- *Editing `config.toml` / `config.docker.toml`*: deliberately untouched, so the Docker bind-mount deployment is unaffected by this change. New KGs are expected to add their own `[kg.<code>]` section rather than the legacy per-KG key style.

**Consequence:** a new KG needs an adapter (declaring `GraphConfig`/`VectorConfig` defaults) and, optionally, a `[kg.<code>]` `config.toml` section for deployment overrides — no change to `ama_kbqa/config.py` required. This is the first change that makes the paper's "thin adapter, no per-KG boilerplate elsewhere" claim true for endpoint/collection config specifically (namespaces/domain_settings were already true).

---

## Verification performed

- Full suite: 419 passed, `ruff check` clean on all touched files.
- New `tests/framework/test_adapter_resolution.py` — **16 tests** (not 23; verify the actual test count in the file before citing it elsewhere) covering: unchanged defaults with no overrides, legacy-key override, generic `[kg.<code>]` override, env-var override, precedence ordering, a synthetic new-KG-code onboarding via generic section only, invalid-int env fallback, and byte-identical resolution against both `config.toml` and `config.docker.toml` including the container-hostname case (`http://virtuoso:8890/sparql`).
- `tests/framework/test_adapters.py` lost 4 tests (`test_supports_reification` / `test_has_temporal_data` × 2 adapters) and `tests/framework/test_config.py` had assertions trimmed, matching the field removal above.

## Two footguns for future maintainers

1. **Malformed env var silently falls through.** In `_resolve_field` (`resolve.py`), a bad `AMA_KBQA_<CODE>_<FIELD>` value (e.g. a non-integer for `port`) is caught in a bare `try/except (TypeError, ValueError): pass` and silently falls through to the next precedence tier instead of raising. A typo'd env var will not error — it will just look like the override never happened. If this causes a confusing production incident, consider logging a warning at minimum before making it fail loudly (loud-fail risks breaking the container on a cosmetic typo, so this is a real trade-off, not an oversight).
2. **A new KG with no `[kg.<code>]` section silently uses its adapter's default inside Docker.** `_create_config()` defaults are written for local dev (typically `localhost`). If a new KG is onboarded without adding a `[kg.<code>]` section to `config.docker.toml`, it will resolve to that localhost default even when running in the Compose stack, because there is no legacy-key fallback for a KG that isn't `kqapro`/`sciqa`. **The KG onboarding documentation should instruct new KGs to add a `[kg.<code>]` section** — see the flag below.

## Known-stale doc (not fixed in this pass)

`docs/guides/dataset_integration.md` (repo root, outside `.agent/`) shows a "new KG server" code sample with module-level constants (`VIRTUOSO_ENDPOINT = "http://localhost:8890/sparql"`, `COLLECTION_ENTITIES = "mydata-entities"`, etc., around lines 239–426) — the exact hardcoded-getter pattern this change replaced. That guide should be updated to show `_ADAPTER.resolved_config` and the `[kg.<code>]` config.toml pattern instead, but it is out of `.agent/`'s scope to edit directly; flagging here per the docs-agent boundary rule.

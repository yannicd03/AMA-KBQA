# SOP: BM25 Hybrid Collection Migration

Run this once per environment to upgrade existing dense-only Qdrant collections to support hybrid (dense + BM25) search. New collections created by `populate_*_vectors.py` are hybrid-capable from the start and do not need this script.

## Related Docs
- [Project Architecture](../System/project_architecture.md) — retrieval architecture overview
- [Decisions: BM25 Hybrid Retrieval](../Decisions/bm25-hybrid-retrieval-architecture.md) — design rationale, capability check behavior, reranker install
- [SOP: Hetzner Deployment](./hetzner_deployment.md) — where to run this in a Hetzner environment

---

## Prerequisites

- Qdrant running and reachable (default: `localhost:6333`; adjust with `--host`/`--port`)
- Qdrant version ≥ 1.15.2 (server-side BM25 inference requirement); `docker-compose.yml` pins `v1.17.1`
- `uv` environment activated (`uv sync` or `uv run python ...`)
- `config.toml` in the repo root (the script reads Qdrant host/port and collection names from it)

---

## Script

```
db/migrate_add_bm25.py
```

**What it does (crash-safe, two-stage copy):**

1. Scroll all points from the original collection (dense vectors + payloads).
2. Build a verified copy at `{name}__bm25_tmp` with the hybrid schema (dense + BM25 sparse vector). Qdrant computes BM25 sparse vectors during upsert via `models.Document(text=..., model="Qdrant/bm25")`.
3. Delete + recreate the original with the hybrid schema; copy points back from the tmp collection (reusing the already-computed sparse vectors — no second inference pass); delete the tmp.

**Idempotent:** Collections that already carry the `"bm25"` sparse index are skipped silently.

---

## Usage

```bash
# Dry run — shows what would happen, no changes
uv run python db/migrate_add_bm25.py --dry-run

# Interactive — prompts "Proceed? [y/N]" before each collection
uv run python db/migrate_add_bm25.py

# Non-interactive — answer yes to all prompts
uv run python db/migrate_add_bm25.py --yes

# Specific collections only
uv run python db/migrate_add_bm25.py --collections kqapro-entities sciqa-entities

# Custom host/port (e.g. Hetzner where Qdrant is on 6335)
uv run python db/migrate_add_bm25.py --host localhost --port 6335 --yes
```

Default collections (from `config.toml`):
- `kqapro-entities`
- `kqapro-relations`
- `sciqa-entities`
- `sciqa-relations`

---

## Post-Migration

1. **Verify:** The script prints a summary line per collection. Check for any `ERROR` lines.
2. **Enable hybrid search** in `config.toml`:
   ```toml
   [retrieval]
   hybrid_enabled = true
   fusion = "rrf"          # or "dbsf"
   prefetch_limit = 20
   ```
3. **Restart MCP servers** so the capability cache (`_bm25_capability_cache`) is cleared (it caches per process lifetime). If a collection is migrated while servers are running, they will continue using dense-only for that collection until restarted.

---

## Gotchas

- **Capability cache is process-lifetime.** If you migrate a collection while a server is running, the server won't detect the new BM25 index until it restarts. Symptom: no errors, but hybrid is silently bypassed (the one-time warning fires once, then the dense-only path is used permanently for that run).
- **The `__bm25_tmp` copy is safe to delete manually** if the script was interrupted mid-stage-2. Stage-2 start is logged; the original collection is still intact after stage-1 completes.
- **Score semantics change after enabling hybrid.** Fused RRF/DBSF scores are rank-based, not cosine similarities. The `score_threshold` in `config.toml` still gates the dense prefetch but has no effect on fused results. This is expected behavior (see ADR).
- **Reranker requires a separate install.** Enabling `reranker_enabled = true` without `uv sync --extra rerank` will raise a clear `ImportError` at the first reranking call. The reranker model (~600 MB) is downloaded from HuggingFace on first use.
- **Fresh environments:** skip this script entirely. Run `db/populate_kqapro_vectors.py` and `db/populate_sciqa_vectors.py` directly — they create hybrid-capable collections.

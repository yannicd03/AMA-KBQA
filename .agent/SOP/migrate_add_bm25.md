# SOP: Add BM25 Sparse Index to Existing Qdrant Collections

## Related Docs
- [System/project_architecture.md](../System/project_architecture.md) — Qdrant connection config, docker-compose ports
- [Decisions/hybrid-retrieval-architecture.md](../Decisions/hybrid-retrieval-architecture.md) — why hybrid retrieval was added; capability check; score_threshold semantics
- [SOP/hetzner_deployment.md](hetzner_deployment.md) — server access, docker compose commands

---

## Overview

`db/migrate_add_bm25.py` adds BM25 sparse vectors to existing Qdrant collections **without re-embedding**. It is crash-safe (two-stage) and idempotent. Run it once per deployment after upgrading to a Qdrant image that supports server-side BM25 inference (>= 1.15.2; compose pins `v1.17.1`).

The migration is a **prerequisite** for `hybrid_enabled = true` to do anything. Before the index exists, `search()` detects its absence via `_collection_has_bm25` and falls back to dense automatically.

---

## 1. Pre-flight

```bash
# Confirm Qdrant is running and reachable
curl http://localhost:6333/collections       # local dev
curl http://localhost:6335/collections       # Hetzner (offset port)

# Confirm Qdrant version >= 1.15.2
curl http://localhost:6333/           # check "version" field in JSON
```

Check which collections exist:

```bash
uv run python db/migrate_add_bm25.py --dry-run --host localhost --port 6333
```

The `--dry-run` flag prints what would be migrated without touching any data. Review the listed collections; the four expected collections are:

| Collection | Points |
|-----------|--------|
| `kqapro-entities` | ~17,754 |
| `kqapro-relations` | ~363 |
| `sciqa-entities` | ~171,588 |
| `sciqa-relations` | ~8,596 |

---

## 2. Run the migration

```bash
# Migrate all collections (prompts for confirmation)
uv run python db/migrate_add_bm25.py --host localhost --port 6333

# Or skip the confirmation prompt
uv run python db/migrate_add_bm25.py --host localhost --port 6333 --yes

# Migrate specific collections only
uv run python db/migrate_add_bm25.py --host localhost --port 6333 \
    --collections kqapro-entities kqapro-relations --yes
```

On Hetzner (port offset):

```bash
ssh hetzner
cd ~/AMAKBQA-main
uv run python db/migrate_add_bm25.py --host localhost --port 6335 --yes
```

**Migration stages per collection:**

1. **Stage 1 — build BM25 copy.** Creates `{name}__bm25_tmp` with hybrid schema (dense + sparse `"bm25"`). Uploads each point's text payload to Qdrant with `models.Document(text=..., model="Qdrant/bm25")` so Qdrant infers the sparse vector server-side. No Python BM25 computation; no re-embedding.
2. **Stage 2 — rebuild original.** Drops the original collection, recreates it with the hybrid schema, then copies both dense and sparse vectors from the `__bm25_tmp` collection back to the original name.
3. **Cleanup.** Drops `{name}__bm25_tmp`.

If the migration crashes mid-run (network drop, OOM, etc.), the `__bm25_tmp` collection remains on disk. Re-running the script detects this and resumes from stage 2 for that collection.

---

## 3. Verify

```bash
# Check that "bm25" appears under sparse_vectors_config for each collection
curl http://localhost:6333/collections/kqapro-entities | python3 -m json.tool | grep -A5 sparse
```

Expected output includes:
```json
"sparse_vectors_config": {
    "bm25": { ... }
}
```

Quick smoke test from Python:

```python
from qdrant_client import QdrantClient
c = QdrantClient(host="localhost", port=6333)
info = c.get_collection("kqapro-entities")
assert "bm25" in info.config.params.sparse_vectors_config, "BM25 index missing"
print("OK — BM25 index present")
```

---

## 4. Enable hybrid retrieval

### Via Settings page (interactive)

Open the Streamlit frontend → **Settings** → **Retrieval Configuration** → toggle **Enable hybrid BM25 + dense search** → **Save to config.toml**.

### Via config.toml (headless / server)

```toml
[retrieval]
hybrid_enabled = true
fusion = "rrf"       # or "dbsf"
prefetch_limit = 20
```

On Hetzner the config used by the Docker frontend is `config.docker.toml`. Edit it on the server then restart the frontend container:

```bash
ssh hetzner "nano ~/AMAKBQA-main/config.docker.toml"
# set hybrid_enabled = true under [retrieval]
ssh hetzner "cd ~/AMAKBQA-main && docker compose restart frontend"
```

MCP server subprocesses (kqapro_server, sciqa_server, orchestrator_server) are spawned fresh per question turn over stdio, so they pick up the new config on the next spawn. No additional restart is needed.

---

## 5. Enable the cross-encoder reranker (optional)

The reranker is independent of hybrid retrieval — it can be used with dense-only results too.

**Install the extra:**

```bash
uv sync --extra rerank
```

**Enable in Settings page** → **Retrieval Configuration** → toggle **Enable cross-encoder reranker** (the toggle is disabled until `sentence_transformers` is importable).

Or set in `config.toml`:

```toml
[retrieval]
reranker_enabled = true
reranker_model = "Alibaba-NLP/gte-reranker-modernbert-base"
rerank_candidates = 20
```

**First-run note:** the model (~300 MB) is downloaded from HuggingFace on first use and cached locally. Subsequent process starts load from disk.

On a server without internet access, download the model on a connected machine first and copy the HuggingFace cache to the server, or point `reranker_model` at a local path.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `"falling back to dense: collection X has no BM25 index"` in logs | Migration not run or collection name mismatch | Run migration for that collection; verify `sparse_vectors_config` |
| Migration fails at stage 1 with 404 on `Qdrant/bm25` model | Qdrant < 1.15.2 | Upgrade Qdrant image to `v1.17.1` (pinned in compose) |
| `__bm25_tmp` collection left behind after crash | Stage 2 did not complete | Re-run migration; it will detect the tmp collection and resume |
| Reranker toggle greyed out in Settings page | `sentence_transformers` not installed | `uv sync --extra rerank` on the machine running the frontend |
| Reranker hangs on first question | Model download in progress | Wait for download to complete; subsequent questions will be fast |
| Fused scores look very different from previous cosine scores | Expected — RRF/DBSF scores are rank-based | Do not compare fused scores against the old `score_threshold` directly; fused scores are not cosine similarities |

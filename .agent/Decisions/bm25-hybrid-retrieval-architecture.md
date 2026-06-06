# ADR: BM25 Hybrid Search + Cross-Encoder Reranker (Shared Retrieval Layer)

**Date:** 2026-06-07
**Status:** Merged (commit `7b1b108`, branch `bm25-hybrid`)

## Related Docs
- [Project Architecture](../System/project_architecture.md) — overall system overview; retrieval section updated
- [SOP: Add BM25 Migration](../SOP/hetzner_deployment.md) — Hetzner runbook references this migration step
- [SOP: BM25 Collection Migration](../SOP/running_batch_processing.md) — batch runbook context

---

## Context

Retrieval was duplicated across three MCP servers:

- `kqapro_server.py` — `get_embedding()` with a 256-entry LRU cache, direct `qdrant.query_points()` calls
- `sciqa_server.py` — uncached `get_embedding()` copy, direct `qdrant.query_points()` calls
- `orchestrator_server.py` — batch embedding helper (`generate_embeddings`), direct `qdrant.query_points()` calls

All five call sites implemented pure dense cosine search. Retrieval quality improvements (BM25 recall lift, cross-encoder reranking) would have required touching all five sites. A bug in `generate_embeddings` returning `json.dumps({})` instead of `{}` for empty input was silently lurking.

## Decision

**Introduce `ama_kbqa/retrieval/` as the single retrieval entrypoint for all MCP servers,** with three independently-toggled capabilities:

1. **Shared embedding cache** (`embeddings.py`) — one 256-entry LRU, shared in-process across all servers
2. **Optional BM25 hybrid search** (`search.py`) — dense + BM25 prefetch branches fused server-side by Qdrant
3. **Optional cross-encoder reranking** (`reranker.py`) — lazy singleton, optional install extra

All three capabilities default to the legacy behavior (pure dense search, no rerank), so a config.toml with no `[retrieval]` section is equivalent to the old code.

## Module Design

### `embeddings.py`

- `embed_query(client, text, *, model=None)` — single text, cached
- `embed_queries(client, texts, *, model=None)` — batch, cache-aware (misses batched in one API call, sequential fallback on failure)
- Cache keyed on `model::normalized_text`; evicts oldest on overflow (simple FIFO, not true LRU by access time)

### `search.py`

Public API:
- `RetrievalParams` — frozen dataclass capturing all retrieval settings for one call
- `build_retrieval_params(*, limit, score_threshold, allow_rerank=True)` — reads from config, respects `allow_rerank=False` override
- `search(qdrant, collection, *, query_text, query_vector, params, query_filter=None)` — unified entrypoint
- `search_terms(qdrant, collection, *, vectors_map, params)` — orchestrator multi-term probe wrapper

BM25 hybrid mode uses Qdrant's `models.Document(text=..., model="Qdrant/bm25")` for server-side BM25 inference (named sparse vector `"bm25"` with IDF modifier). The dense vector remains at the unnamed default (`using=""`).

### `reranker.py`

- `get_reranker(model_name)` — lazy singleton, raises clear `ImportError` if `sentence-transformers` not installed
- `rerank(query, docs, model_name)` — batch predict, returns aligned float scores

## Key Design Decisions with Trade-offs

### 1. Threshold semantics: cosine threshold gates dense branch only

**Decision:** The caller's `score_threshold` is passed to the dense prefetch (or the dense-only query) but NOT to the BM25 prefetch or to the fused result set.

**Why:** Fused RRF/DBSF scores are rank-based and dimensionally incomparable to cosine similarities. Gating fused results with a cosine threshold would silently discard valid BM25-rescued candidates. BM25 is additive recall — it never gates.

**Trade-off:** After fusion, callers can no longer use `score_threshold` to filter results; they must rely on the `limit` parameter to bound results.

### 2. Reranker overwrites `point.score`

**Decision:** `_rerank_points` replaces `point.score` with the cross-encoder score in-place.

**Why:** All five call sites map `point.score` to `relevance_score` in their DTO. Overwriting score means zero call-site changes while surfacing the reranked order.

**Trade-off:** After reranking, `point.score` is no longer a cosine similarity — the scale changes (cross-encoder outputs are not bounded to [0,1]). Callers that display raw scores to users will show cross-encoder logits instead of cosine values.

### 3. Per-collection capability check, not exception handling

**Decision:** Before the first hybrid query against a collection, `_collection_has_bm25()` calls `get_collection()` and inspects `sparse_vectors`. If absent, it falls back to dense with a one-time warning.

**Why:** A hybrid Qdrant query against a dense-only collection returns a hard HTTP 400, not a soft retrieval failure. Catching 400s and retrying dense adds latency and ambiguity; a cheap pre-check avoids both. The result is cached per collection name for the process lifetime.

**Trade-off:** If a collection is migrated while the server is running, the capability cache will be stale (still reports False). Restart the server after migration.

### 4. Unnamed default dense vector (`using=""`)

**Decision:** Kept the existing unnamed default vector for the dense branch; the BM25 branch uses the named `"bm25"` vector. No dense renaming or re-embedding was needed.

**Why:** Validated that `upsert(vector={"": dense_vec, "bm25": models.Document(...)})` and `Prefetch(using="")` both work with Qdrant 1.17.x.

**Trade-off:** The empty-string `using=""` is implicit and slightly surprising. Documented in `search.py` (`DENSE_VECTOR_NAME = ""`).

### 5. Reranker as optional extra (`uv sync --extra rerank`)

**Decision:** `sentence-transformers` (and its torch dependency, ~GB) is declared as an optional extra in `pyproject.toml`, not a mandatory dependency.

**Why:** The default install (frontend Docker image, benchmark baseline) must stay torch-free to avoid ~GB image bloat. The reranker is only needed when `reranker_enabled = true`.

**Trade-off:** Users who set `reranker_enabled = true` without installing the extra will get a clear `ImportError` at first reranking call (not at startup).

### 6. Orchestrator routing probe permanently skips reranking

**Decision:** `orchestrator_server.py` calls `build_retrieval_params(allow_rerank=False)` unconditionally.

**Why:** The probe embeds N terms and queries each collection per-term. Reranking each term result would multiply latency by N (reranker model loads ~17s on first call, then ~0.21s per 20 candidates on CPU). Routing confidence is not sensitive enough to the reranker margin to justify this.

**Trade-off:** The orchestrator's entity-match evidence is always dense-only, even when hybrid is enabled for regular search.

### 7. Defaults reproduce legacy behavior exactly

**Decision:** `hybrid_enabled = false`, `reranker_enabled = false` in `config.toml`.

**Why:** Zero regression risk when merging. Environments that haven't run `migrate_add_bm25.py` continue to work without touching config.

## Rejected Alternatives

- **Per-server improvement** (add BM25 to each server separately): rejected because it would have left the embedding duplication and bug in place.
- **Mandatory hybrid** (require BM25 migration before merge): rejected because environments without migrated collections would break on merge.
- **In-process BM25 (e.g. BM25S library)**: rejected because Qdrant's server-side inference keeps the BM25 vectors consistent with the dense vectors and avoids a second Python dependency.

## Reranker: Operational Facts

| Fact | Value |
|------|-------|
| Default model | `Alibaba-NLP/gte-reranker-modernbert-base` (~149M params, ~600 MB) |
| Download | HuggingFace, on first use |
| `trust_remote_code` | Not required |
| `transformers` version | ≥4.48 (native ModernBERT; tested with 5.10) |
| Load time (first use) | ~17s on CPU |
| Rerank 21 candidates | ~0.21s on CPU |
| Install | `uv sync --extra rerank` |

## Qdrant Version Requirement

Server-side BM25 inference (`models.Document` in `Prefetch`) requires **Qdrant ≥ 1.15.2**. The `docker-compose.yml` now pins `qdrant/qdrant:v1.17.1`.

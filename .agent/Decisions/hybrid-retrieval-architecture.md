# ADR: Hybrid BM25 + Dense Retrieval and Cross-Encoder Reranking

**Date:** 2026-06-08  
**Commit:** `4988f0a` (yannic-dev branch)  
**Status:** Shipped  
**Files:** `ama_kbqa/retrieval/embeddings.py`, `ama_kbqa/retrieval/search.py`, `ama_kbqa/retrieval/reranker.py`, `ama_kbqa/config.py`, `ama_kbqa/frontend/pages/4_Settings.py`, `db/migrate_add_bm25.py`, `config.toml`, `config.docker.toml`

## Related Docs
- [System/project_architecture.md](../System/project_architecture.md) — configuration system, Settings page, MCP server call sites
- [SOP/migrate_add_bm25.md](../SOP/migrate_add_bm25.md) — runbook for adding the sparse BM25 index to existing Qdrant collections

---

## Problem

Entity and predicate/relation lookup used a single dense vector pass (`query_points` with an unnamed vector). For entity names that are proper nouns, acronyms, or codes, lexical overlap matters more than distributional similarity. The dense-only path could retrieve semantically proximate but wrong entities while missing the exact term. A secondary problem: the returned top-k candidates from any retrieval mode were unranked beyond cosine distance, so a slightly better vocabulary match deep in the list was silently discarded.

---

## Decision 1: Server-side BM25 hybrid via the Qdrant Query API (RRF / DBSF fusion)

### Architecture

Retrieval is centralised in the new `ama_kbqa/retrieval/` module:

- **`embeddings.py`** — `get_query_embedding(text)`: cached (`functools.lru_cache`) query embedding call. Prevents redundant embedding calls when the same term is searched by multiple call sites in one question turn.
- **`search.py`** — `RetrievalParams` dataclass, `build_retrieval_params(config)`, `search(collection, query, ...)` and `search_terms(collection, terms, ...)`. All call sites in `kqapro_server.py` (`FindNode`, relation search) and `sciqa_server.py` (`FindResource`, `FindPredicate`) and `orchestrator_server.py` (`_search_qdrant` probe) were migrated here.
- **`reranker.py`** — lazy `CrossEncoder` singleton, `rerank(query, results)`.

When hybrid is enabled, `search()` issues a Qdrant **Query API** request that fuses two prefetch branches server-side:

1. Dense prefetch — top `prefetch_limit` candidates by cosine similarity on the unnamed vector `using=""`, optionally gated by `score_threshold` (cosine score; gates only this branch, not the fused result).
2. BM25 prefetch — top `prefetch_limit` candidates by sparse score on the named sparse vector `"bm25"`, with `IDF` modifier.

Fusion is applied server-side by Qdrant before results are returned. The default fusion mode is **RRF** (Reciprocal Rank Fusion).

BM25 inference is **server-side**: documents were indexed with `models.Document(text=..., model="Qdrant/bm25")` during migration; query-time scoring also uses `Qdrant/bm25` inference inside Qdrant, so no Python BM25 library is needed at query time.

### Capability check and graceful fallback

Before issuing a hybrid query, `search()` calls `_collection_has_bm25(client, collection)`, which inspects the collection's `sparse_vectors_config` for a `"bm25"` entry. The result is cached (per-process `dict`) so the check costs at most one Qdrant API call per collection per process lifetime.

If the collection lacks the sparse index, `search()` logs `"falling back to dense: collection <name> has no BM25 index"` and issues a plain dense query. This means `hybrid_enabled = true` in config is safe even before migration has run: the system degrades gracefully rather than erroring.

### `score_threshold` semantics

`score_threshold` (from `[retrieval]` config) is a **cosine similarity gate** applied only to the dense prefetch branch. It does not apply to the BM25 branch and does not apply to the fused result list. Fused scores are rank-based (RRF formula or DBSF) and are not comparable to cosine scores. This is intentional: filtering the dense branch prunes obviously-unrelated dense candidates while leaving the BM25 branch to contribute lexically exact matches regardless of embedding distance.

### Qdrant version requirement

Server-side BM25 inference via `models.Document(model="Qdrant/bm25")` requires **Qdrant >= 1.15.2**. The Docker Compose stack pins `qdrant/qdrant:v1.17.1`.

---

## Decision 2: Optional cross-encoder reranker (independent of hybrid)

A cross-encoder reranking stage is available **independently** of hybrid retrieval: it reranks dense-only results too.

The reranker loads lazily (singleton in `reranker.py`) on first use and scores each `(query, candidate_label)` pair with a cross-encoder model. The per-point `.score` field is overwritten with the cross-encoder score in-place.

Default model: `Alibaba-NLP/gte-reranker-modernbert-base`.

The reranker lives behind an **optional pip extra** (`[rerank]`), which adds `sentence-transformers` and `transformers`. Enable with `uv sync --extra rerank`. The Settings page disables the reranker toggle and shows a `uv sync --extra rerank` hint when `sentence_transformers` is not importable at runtime.

The orchestrator's `_search_qdrant` routing probe always passes `allow_rerank=False`: routing should not pay reranker latency, and reranking a few entity names for routing purposes would not change the evidence quality meaningfully.

---

## Decision 3: Configuration via `[retrieval]` TOML section + `AMA_RETRIEVAL_*` env overlay

All retrieval knobs live in a dedicated `[retrieval]` section in `config.toml` (and `config.docker.toml`). `config.py::get_retrieval_config()` reads this section and then overlays `AMA_RETRIEVAL_<KEY>` environment variables (upper-cased key, typed parsing; env wins). The `RETRIEVAL_ENV_PREFIX = "AMA_RETRIEVAL_"` constant documents the convention.

Typed getters: `get_hybrid_enabled`, `get_fusion`, `get_prefetch_limit`, `get_reranker_enabled`, `get_reranker_model`, `get_rerank_candidates`, `get_rerank_threshold`.

Defaults (all off by default, so existing deployments are unaffected without explicit opt-in):

```toml
[retrieval]
hybrid_enabled = false
fusion = "rrf"          # "rrf" or "dbsf"
prefetch_limit = 20
reranker_enabled = false
reranker_model = "Alibaba-NLP/gte-reranker-modernbert-base"
rerank_candidates = 20
```

The env overlay is a secondary channel intended for headless benchmark runs and CI where you want to flip hybrid on/off without editing a config file. The Settings page does not use the env overlay; it uses the disk/session-cache path (see Decision 4).

---

## Decision 4: Frontend exposure via Settings page disk/session persistence

On yannic-dev the frontend is the **multi-page Streamlit app**. Retrieval options are surfaced in the "Retrieval Configuration" section of `pages/4_Settings.py`:

- Hybrid toggle (`hybrid_enabled`)
- RRF / DBSF fusion selectbox (gated — only shown when hybrid is on)
- Prefetch limit number input
- Reranker toggle (disabled with a `uv sync --extra rerank` hint when `sentence_transformers` is absent)
- Reranker model text input
- Rerank candidates number input

The page mutates `edited["retrieval"]` and persists through the existing Settings page save flow:

- **"Save to config.toml"** — `config_editor.save_config` writes the `[retrieval]` section to disk. MCP server subprocesses are spawned fresh per question turn (stdio), so they pick up the new config on the next spawn without a restart.
- **"Apply to Session Only"** — `config_editor.apply_to_session` sets `_config_cache` for the currently running Streamlit process only; no disk write.

There is no mechanism to propagate settings changes to already-running MCP subprocesses mid-turn: the subprocess-per-turn model makes this unnecessary.

---

## Rejected alternatives

### Client-side BM25 (e.g., rank-bm25 Python library)

Running BM25 in the Python process would require loading the full document corpus into memory and maintaining a separate tokenized index. For 170K+ SciQA entities this would be expensive both in memory and index-load time. Server-side inference inside Qdrant keeps the index collocated with the vectors and requires no extra Python dependency on the query path.

### Reranker always-on

A cross-encoder adds 100–400 ms per question turn depending on candidates and hardware. Routing probes (orchestrator) are latency-critical. Making reranking opt-in per call site (via `allow_rerank`) avoids this cost where it doesn't help while still making it available for the main entity/predicate lookups.

### Separate dense and BM25 query-then-merge in Python

Issuing two separate Qdrant queries and merging in Python is equivalent to what the Qdrant Query API does server-side, but it doubles round-trips and requires Python-side RRF/DBSF implementation. The Query API was specifically designed for this use case.

---

## Trade-offs accepted

| Aspect | Trade-off |
|--------|-----------|
| Qdrant version floor | Minimum Qdrant 1.15.2 required for server-side BM25 inference. Pinned in compose. |
| Migration required | Existing collections need `db/migrate_add_bm25.py` before `hybrid_enabled = true` takes effect. The capability check provides graceful fallback in the meantime. |
| Reranker cold-start | First use of the cross-encoder downloads the model (~300 MB) from HuggingFace. Subsequent starts load from disk cache. Acceptable for interactive use; not suitable for latency-sensitive paths (hence `allow_rerank=False` on the routing probe). |
| Fused scores are opaque | After RRF/DBSF fusion, `.score` values are rank-based, not cosine similarities. Code downstream of `search()` must not interpret them as cosine scores. The `score_threshold` parameter therefore only gates the dense prefetch branch. |

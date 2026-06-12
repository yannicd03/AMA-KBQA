"""Unified Qdrant retrieval: dense, BM25 hybrid, and optional reranking.

Mode selection (per the [retrieval] config section):

- ``hybrid_enabled = false`` → a single dense ``query_points`` call,
  byte-identical to the legacy per-server behavior.
- ``hybrid_enabled = true`` → two prefetch branches (dense + server-side
  BM25 via ``models.Document``) fused server-side with RRF or DBSF.
  Requires the collection to carry a ``bm25`` sparse vector (see
  ``db/migrate_add_bm25.py``); collections without it fall back to dense
  with a one-time warning, because a hybrid query against them is a hard
  400 error, not a catchable soft failure.
- ``reranker_enabled = true`` → the candidate list (dense OR hybrid) is
  re-scored by a cross-encoder and ``point.score`` is OVERWRITTEN with the
  rerank score, so existing ``relevance_score=point.score`` mappings
  surface the final ranking. Note the scale changes: cross-encoder scores
  are not cosine similarities and not necessarily in [0, 1].

Threshold semantics: the caller's cosine ``score_threshold`` gates only
the dense branch (dense-only query, or the dense prefetch in hybrid mode).
Fused RRF/DBSF scores are rank-based and are never gated with the cosine
value; the BM25 branch is purely additive recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient, models

from ama_kbqa import config as app_config

BM25_SPARSE_VECTOR_NAME = "bm25"
BM25_MODEL = "Qdrant/bm25"
# The collections use Qdrant's unnamed default dense vector, addressed by "".
DENSE_VECTOR_NAME = ""


@dataclass(frozen=True)
class RetrievalParams:
    """Resolved retrieval settings for a single search call."""

    limit: int                      # final number of hits returned
    score_threshold: Optional[float]  # cosine gate, dense branch only
    hybrid_enabled: bool
    fusion: str                     # "rrf" | "dbsf"
    prefetch_limit: int             # per-branch candidates before fusion
    reranker_enabled: bool
    reranker_model: str
    rerank_candidates: int          # hits fed to the cross-encoder
    rerank_threshold: Optional[float]


def build_retrieval_params(
    *,
    limit: int,
    score_threshold: Optional[float],
    allow_rerank: bool = True,
) -> RetrievalParams:
    """Build RetrievalParams from the [retrieval] config section.

    Args:
        limit: Final number of hits the caller wants back.
        score_threshold: Caller's cosine threshold (dense branch only),
            or None for no gate.
        allow_rerank: Set False to hard-disable reranking for this call
            site regardless of config (used by the orchestrator routing
            probe, where per-term reranking latency is not worth it).
    """
    return RetrievalParams(
        limit=limit,
        score_threshold=score_threshold,
        hybrid_enabled=app_config.get_hybrid_enabled(),
        fusion=app_config.get_fusion(),
        prefetch_limit=app_config.get_prefetch_limit(),
        reranker_enabled=app_config.get_reranker_enabled() and allow_rerank,
        reranker_model=app_config.get_reranker_model(),
        rerank_candidates=app_config.get_rerank_candidates(),
        rerank_threshold=app_config.get_rerank_threshold(),
    )


# Per-collection capability check, cached so we don't pay a get_collection
# round-trip on every search. Only definitive answers are cached; transient
# get_collection failures fall back to dense for that call without caching.
_bm25_capability_cache: Dict[str, bool] = {}
_fallback_warned: set = set()


def reset_capability_cache() -> None:
    """Clear the cached per-collection BM25 capability checks (tests/migrations)."""
    _bm25_capability_cache.clear()
    _fallback_warned.clear()


# Per-question cache of full search results, keyed by the complete query
# identity (collection, normalized text, params, serialized filter). Within
# one question the collections are static, so a byte-identical repeat call
# (loop retry, plan revisiting the same search) returns the same hits
# without re-running Qdrant + the reranker. Scope: ONE question. The MCP
# servers clear it on journal clear (the question boundary) so benchmark
# timings reflect real single-question usage; it never persists across
# questions. Distinct from _bm25_capability_cache above, which is a schema
# property and intentionally process-wide.
_result_cache: Dict[tuple, List[models.ScoredPoint]] = {}
_RESULT_CACHE_MAX = 128


def clear_result_cache() -> None:
    """Clear the per-question search-result cache (question boundary hook)."""
    _result_cache.clear()


def _result_cache_key(
    collection_name: str,
    query_text: str,
    params: RetrievalParams,
    query_filter: Optional[models.Filter],
) -> Optional[tuple]:
    """Build a hashable identity for a search call, or None if not cacheable."""
    if query_filter is None:
        filter_key = None
    else:
        try:
            filter_key = query_filter.model_dump_json()
        except Exception:
            return None
    return (
        collection_name,
        (query_text or "").strip().lower(),
        params,  # frozen dataclass, hashable
        filter_key,
    )


def _collection_has_bm25(qdrant: QdrantClient, collection_name: str) -> bool:
    cached = _bm25_capability_cache.get(collection_name)
    if cached is not None:
        return cached
    try:
        info = qdrant.get_collection(collection_name)
        sparse = info.config.params.sparse_vectors or {}
        has_bm25 = BM25_SPARSE_VECTOR_NAME in sparse
    except Exception as e:
        logger.warning(
            f"Could not inspect collection '{collection_name}' for BM25 "
            f"capability ({e}); using dense-only for this call"
        )
        return False
    _bm25_capability_cache[collection_name] = has_bm25
    return has_bm25


def _doc_text(payload: Optional[dict]) -> str:
    """Document text for BM25/rerank scoring, via payload fallback chain."""
    payload = payload or {}
    for key in ("name", "predicate", "label"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def search(
    qdrant: QdrantClient,
    collection_name: str,
    *,
    query_text: str,
    query_vector: List[float],
    params: RetrievalParams,
    query_filter: Optional[models.Filter] = None,
) -> List[models.ScoredPoint]:
    """Unified retrieval entrypoint for all MCP servers.

    Returns Qdrant ScoredPoint objects so each call site keeps its own
    payload→DTO mapping (``point.score``, ``point.payload``).
    """
    cache_key = _result_cache_key(collection_name, query_text, params, query_filter)
    if cache_key is not None and cache_key in _result_cache:
        return list(_result_cache[cache_key])

    # When reranking, over-fetch so the cross-encoder has a candidate pool.
    if params.reranker_enabled:
        fetch_limit = max(params.rerank_candidates, params.limit)
    else:
        fetch_limit = params.limit

    use_hybrid = params.hybrid_enabled and _collection_has_bm25(qdrant, collection_name)
    if params.hybrid_enabled and not use_hybrid:
        # Degraded call (missing bm25 index or transient capability error):
        # don't cache it, so a retry within the question can recover to the
        # full hybrid path once the collection is inspectable again.
        cache_key = None
    if params.hybrid_enabled and not use_hybrid and collection_name not in _fallback_warned:
        _fallback_warned.add(collection_name)
        logger.warning(
            f"Hybrid search is enabled but collection '{collection_name}' has no "
            f"'{BM25_SPARSE_VECTOR_NAME}' sparse index; falling back to dense-only. "
            f"Run db/migrate_add_bm25.py to upgrade the collection."
        )

    if use_hybrid:
        dense_prefetch = models.Prefetch(
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=params.prefetch_limit,
            score_threshold=params.score_threshold,
            filter=query_filter,
        )
        bm25_prefetch = models.Prefetch(
            query=models.Document(text=query_text, model=BM25_MODEL),
            using=BM25_SPARSE_VECTOR_NAME,
            limit=params.prefetch_limit,
            filter=query_filter,
            # No score_threshold: BM25 scores are not cosine similarities.
        )
        fusion = models.Fusion.DBSF if params.fusion == "dbsf" else models.Fusion.RRF
        points = qdrant.query_points(
            collection_name=collection_name,
            prefetch=[dense_prefetch, bm25_prefetch],
            query=models.FusionQuery(fusion=fusion),
            limit=fetch_limit,
            with_payload=True,
        ).points
    else:
        points = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=fetch_limit,
            with_payload=True,
            score_threshold=params.score_threshold,
            query_filter=query_filter,
        ).points

    if params.reranker_enabled and len(points) > 1:
        points = _rerank_points(query_text, list(points), params)

    result = list(points)[: params.limit]
    if cache_key is not None:
        if len(_result_cache) >= _RESULT_CACHE_MAX:
            _result_cache.pop(next(iter(_result_cache)))
        _result_cache[cache_key] = result
    return list(result)


def _rerank_points(
    query_text: str,
    points: List[models.ScoredPoint],
    params: RetrievalParams,
) -> List[models.ScoredPoint]:
    """Re-score points with the cross-encoder, overwriting ``point.score``."""
    # Lazy import keeps torch/sentence-transformers out of the default path.
    from ama_kbqa.retrieval import reranker

    docs = [_doc_text(p.payload) for p in points]
    try:
        scores = reranker.rerank(query_text, docs, params.reranker_model)
    except ImportError:
        raise
    except Exception as e:
        logger.error(f"Reranking failed ({e}); keeping retrieval order")
        return points

    for point, score in zip(points, scores):
        point.score = float(score)
    points.sort(key=lambda p: p.score, reverse=True)

    if params.rerank_threshold is not None:
        points = [p for p in points if p.score >= params.rerank_threshold]
    return points


def search_terms(
    qdrant: QdrantClient,
    collection_name: str,
    *,
    vectors_map: Dict[str, Optional[List[float]]],
    params: RetrievalParams,
) -> Dict[str, List[models.ScoredPoint]]:
    """Run :func:`search` over a {term: vector} map (orchestrator probe).

    Terms with falsy vectors are skipped (absent from the result); per-term
    errors yield an empty list, matching the legacy probe behavior.
    """
    results: Dict[str, List[models.ScoredPoint]] = {}
    for term, vector in vectors_map.items():
        if not vector:
            continue
        try:
            results[term] = search(
                qdrant,
                collection_name,
                query_text=term,
                query_vector=vector,
                params=params,
            )
        except Exception as e:
            logger.error(f"[search_terms] Error for '{term}' in {collection_name}: {e}")
            results[term] = []
    return results

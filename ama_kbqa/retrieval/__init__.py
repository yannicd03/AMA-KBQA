"""Shared retrieval layer for all MCP servers.

Unifies dense embedding (with a shared LRU cache), optional BM25 hybrid
search with server-side score fusion (Qdrant Query API), and an optional
cross-encoder reranker stage. Behavior is controlled by the [retrieval]
section of config.toml; with the default settings retrieval is identical
to the legacy pure-dense `query_points` calls.
"""

from ama_kbqa.retrieval.embeddings import (
    clear_embedding_cache,
    embed_queries,
    embed_query,
)
from ama_kbqa.retrieval.search import (
    RetrievalParams,
    build_retrieval_params,
    clear_result_cache,
    search,
    search_terms,
)


def clear_question_caches() -> None:
    """Reset the per-question retrieval caches (query embeddings + results).

    Called by the MCP servers when the journal is cleared, which is the
    question boundary (BaseKBQAAgent.soft_reset issues
    ManageJournal(action="clear") between questions). The caches deliberately
    do not survive across questions: cross-question reuse would let benchmark
    questions about the same resources subsidize each other's latency, which
    real single-question usage never does.
    """
    clear_embedding_cache()
    clear_result_cache()


__all__ = [
    "RetrievalParams",
    "build_retrieval_params",
    "clear_embedding_cache",
    "clear_question_caches",
    "clear_result_cache",
    "embed_queries",
    "embed_query",
    "search",
    "search_terms",
]

"""Shared retrieval layer for all MCP servers.

Unifies dense embedding (with a shared LRU cache), optional BM25 hybrid
search with server-side score fusion (Qdrant Query API), and an optional
cross-encoder reranker stage. Behavior is controlled by the [retrieval]
section of config.toml; with the default settings retrieval is identical
to the legacy pure-dense `query_points` calls.
"""

from ama_kbqa.retrieval.embeddings import embed_queries, embed_query
from ama_kbqa.retrieval.search import (
    RetrievalParams,
    build_retrieval_params,
    search,
    search_terms,
)

__all__ = [
    "RetrievalParams",
    "build_retrieval_params",
    "embed_queries",
    "embed_query",
    "search",
    "search_terms",
]

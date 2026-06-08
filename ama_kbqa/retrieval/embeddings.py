"""Dense query embedding with a shared LRU cache.

Replaces the per-server `get_embedding` copies (kqapro had a cached one,
sciqa an uncached one, the orchestrator a batch variant) with a single
implementation used by every MCP server.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ama_kbqa.config import get_embedding_model_name

# TODO(instruct-prefix): qwen3-embedding expects an instruct prefix on
# queries but NOT on documents. Adding it changes embedding semantics and
# would require re-tuning the score thresholds, so it is deliberately not
# implemented. If added later: apply in embed_query/embed_queries only
# (never on the populate/migration document side) and fold the prefix into
# the cache key.

# LRU cache for embeddings to avoid redundant API calls (shared by all
# servers in the process; keyed on model + normalized text).
_embedding_cache: Dict[str, List[float]] = {}
_EMBEDDING_CACHE_MAX = 256


def _cache_key(text: str, model: str) -> str:
    return f"{model}::{text.strip().lower()}"


def _cache_put(key: str, embedding: List[float]) -> None:
    # Evict oldest entry if cache is full
    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]
    _embedding_cache[key] = embedding


def embed_query(client: OpenAI, text: str, *, model: Optional[str] = None) -> List[float]:
    """Embed a single query text, using the shared cache.

    Args:
        client: OpenAI-compatible embeddings client.
        text: The text to embed (newlines are normalized to spaces).
        model: Embedding model name; defaults to the configured one.

    Returns:
        The dense embedding vector.
    """
    model = model or get_embedding_model_name()
    text = text.replace("\n", " ")
    key = _cache_key(text, model)

    if key in _embedding_cache:
        return _embedding_cache[key]

    response = client.embeddings.create(
        model=model,
        input=[text],
        encoding_format="float",
    )
    embedding = response.data[0].embedding
    _cache_put(key, embedding)
    return embedding


def embed_queries(
    client: OpenAI,
    texts: List[str],
    *,
    model: Optional[str] = None,
) -> Dict[str, Optional[List[float]]]:
    """Embed a batch of query texts, using the shared cache.

    Cache hits are served locally; only misses go to the API, batched in a
    single call. If the batch call fails, falls back to sequential requests
    (preserving the orchestrator's legacy behavior); texts that still fail
    map to None.

    Args:
        client: OpenAI-compatible embeddings client.
        texts: Texts to embed (newlines are normalized to spaces).
        model: Embedding model name; defaults to the configured one.

    Returns:
        Mapping of each (newline-normalized) input text to its vector,
        or None when embedding that text failed.
    """
    model = model or get_embedding_model_name()
    clean_texts = [t.replace("\n", " ") for t in texts]

    results: Dict[str, Optional[List[float]]] = {}
    misses: List[str] = []
    for text in clean_texts:
        cached = _embedding_cache.get(_cache_key(text, model))
        if cached is not None:
            results[text] = cached
        elif text not in misses:
            misses.append(text)

    if not misses:
        return results

    try:
        response = client.embeddings.create(model=model, input=misses)
        for i, text in enumerate(misses):
            embedding = response.data[i].embedding
            _cache_put(_cache_key(text, model), embedding)
            results[text] = embedding
    except Exception as e:
        logger.warning(
            f"Batch embedding failed ({e}), falling back to sequential processing..."
        )
        for text in misses:
            try:
                response = client.embeddings.create(model=model, input=text)
                embedding = response.data[0].embedding
                _cache_put(_cache_key(text, model), embedding)
                results[text] = embedding
            except Exception as e:
                logger.error(f"Error embedding '{text}': {e}")
                results[text] = None

    return results

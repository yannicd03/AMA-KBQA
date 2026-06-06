"""Lazy cross-encoder reranker.

sentence-transformers (and the torch it pulls in) is an optional extra:
``uv sync --extra rerank``. The import happens inside :func:`get_reranker`
so the default install never touches torch; enabling
``retrieval.reranker_enabled`` without the extra raises a clear hint.

The default model, Alibaba-NLP/gte-reranker-modernbert-base (~149M params),
needs ``transformers>=4.48`` (native ModernBERT support, no
``trust_remote_code`` required) and downloads from HuggingFace on first
use. Runs on CPU by default; reranking ~20 candidates takes roughly
0.5-2s, which is fine for interactive tool calls.
"""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

_RERANKER_CACHE: Dict[str, object] = {}


def get_reranker(model_name: str):
    """Return a cached CrossEncoder for ``model_name``, loading it on first use.

    Raises:
        ImportError: If the optional rerank dependencies are not installed.
    """
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "Reranking is enabled (retrieval.reranker_enabled = true) but "
            "sentence-transformers is not installed. Install the optional "
            "extra: uv sync --extra rerank"
        ) from e

    logger.info(f"Loading cross-encoder reranker '{model_name}' (first use)...")
    model = CrossEncoder(model_name)
    _RERANKER_CACHE[model_name] = model
    return model


def rerank(query: str, docs: List[str], model_name: str) -> List[float]:
    """Score (query, doc) pairs with the cross-encoder in one batch.

    Returns:
        Relevance scores aligned to ``docs``.
    """
    model = get_reranker(model_name)
    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs)
    return [float(s) for s in scores]

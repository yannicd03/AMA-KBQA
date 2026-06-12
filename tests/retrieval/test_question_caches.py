"""Tests for the per-question retrieval caches (embeddings + full results).

Scope contract: both caches live for exactly one question. The MCP servers
clear them on ManageJournal(action="clear"), which BaseKBQAAgent.soft_reset
issues between questions. They must never leak hits across questions.
"""

import importlib
from types import SimpleNamespace

import pytest

from ama_kbqa import retrieval
from ama_kbqa.retrieval.search import RetrievalParams

# The package re-exports a `search` FUNCTION, shadowing the submodule name;
# resolve the actual modules for cache-internals access.
emb_mod = importlib.import_module("ama_kbqa.retrieval.embeddings")
search_mod = importlib.import_module("ama_kbqa.retrieval.search")


@pytest.fixture(autouse=True)
def _clean_caches():
    retrieval.clear_question_caches()
    yield
    retrieval.clear_question_caches()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeEmbeddingClient:
    def __init__(self):
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeQdrant:
    def __init__(self):
        self.calls = 0

    def query_points(self, **kwargs):
        self.calls += 1
        point = SimpleNamespace(
            id=self.calls, score=0.9, payload={"name": f"hit{self.calls}"}
        )
        return SimpleNamespace(points=[point])


def _dense_params(limit=5):
    return RetrievalParams(
        limit=limit,
        score_threshold=0.0,
        hybrid_enabled=False,
        fusion="rrf",
        prefetch_limit=20,
        reranker_enabled=False,
        reranker_model="x",
        rerank_candidates=20,
        rerank_threshold=None,
    )


# ---------------------------------------------------------------------------
# Embedding cache scoping
# ---------------------------------------------------------------------------

def test_embed_query_caches_within_question():
    client = FakeEmbeddingClient()
    v1 = retrieval.embed_query(client, "installed capacity", model="m")
    v2 = retrieval.embed_query(client, "installed capacity", model="m")
    assert client.calls == 1
    assert v1 == v2


def test_clear_question_caches_forces_reembedding():
    client = FakeEmbeddingClient()
    retrieval.embed_query(client, "installed capacity", model="m")
    retrieval.clear_question_caches()
    retrieval.embed_query(client, "installed capacity", model="m")
    assert client.calls == 2


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

def test_identical_search_served_from_cache():
    qdrant = FakeQdrant()
    params = _dense_params()
    kwargs = dict(query_text="Montreal", query_vector=[0.1] * 4, params=params)
    r1 = retrieval.search(qdrant, "kqapro-entities", **kwargs)
    r2 = retrieval.search(qdrant, "kqapro-entities", **kwargs)
    assert qdrant.calls == 1
    assert [p.payload for p in r1] == [p.payload for p in r2]
    # Caller gets a fresh list each time (mutating it must not poison the cache).
    r2.clear()
    r3 = retrieval.search(qdrant, "kqapro-entities", **kwargs)
    assert len(r3) == 1 and qdrant.calls == 1


def test_different_text_params_or_collection_miss():
    qdrant = FakeQdrant()
    params = _dense_params()
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1], params=params)
    retrieval.search(qdrant, "c1", query_text="b", query_vector=[0.1], params=params)
    retrieval.search(qdrant, "c2", query_text="a", query_vector=[0.1], params=params)
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1],
                     params=_dense_params(limit=3))
    assert qdrant.calls == 4


def test_clear_question_caches_drops_results():
    qdrant = FakeQdrant()
    params = _dense_params()
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1], params=params)
    retrieval.clear_question_caches()
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1], params=params)
    assert qdrant.calls == 2


def test_filtered_searches_cache_on_filter_identity():
    from qdrant_client.http import models
    qdrant = FakeQdrant()
    params = _dense_params()
    f1 = models.Filter(must=[models.FieldCondition(
        key="kind", match=models.MatchValue(value="entity"))])
    f2 = models.Filter(must=[models.FieldCondition(
        key="kind", match=models.MatchValue(value="relation"))])
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1],
                     params=params, query_filter=f1)
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1],
                     params=params, query_filter=f1)
    assert qdrant.calls == 1
    retrieval.search(qdrant, "c1", query_text="a", query_vector=[0.1],
                     params=params, query_filter=f2)
    assert qdrant.calls == 2


def test_result_cache_is_bounded():
    qdrant = FakeQdrant()
    params = _dense_params()
    for i in range(search_mod._RESULT_CACHE_MAX + 10):
        retrieval.search(qdrant, "c1", query_text=f"q{i}", query_vector=[0.1],
                         params=params)
    assert len(search_mod._result_cache) <= search_mod._RESULT_CACHE_MAX


# ---------------------------------------------------------------------------
# Question boundary: servers clear the caches on journal clear
# ---------------------------------------------------------------------------

def _seed_caches():
    client = FakeEmbeddingClient()
    retrieval.embed_query(client, "seed", model="m")
    retrieval.search(FakeQdrant(), "c1", query_text="seed", query_vector=[0.1],
                     params=_dense_params())
    assert emb_mod._embedding_cache and search_mod._result_cache


def test_kqapro_journal_clear_drops_question_caches():
    from ama_kbqa.server import kqapro_server as kqa
    _seed_caches()
    fn = getattr(kqa.ManageJournal, "fn", kqa.ManageJournal)
    fn(action="clear", content="", context=None)
    assert not emb_mod._embedding_cache
    assert not search_mod._result_cache


def test_sciqa_journal_clear_drops_question_caches():
    import asyncio
    from ama_kbqa.server import sciqa_server as sci
    _seed_caches()
    fn = getattr(sci.ManageJournal, "fn", sci.ManageJournal)
    asyncio.run(fn(app_context=None, action="clear", content=""))
    assert not emb_mod._embedding_cache
    assert not search_mod._result_cache

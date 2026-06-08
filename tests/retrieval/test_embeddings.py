"""Tests for the shared embedding helpers (ama_kbqa/retrieval/embeddings.py)."""
from types import SimpleNamespace

import pytest

from ama_kbqa.retrieval import embeddings as emb_mod
from ama_kbqa.retrieval.embeddings import embed_queries, embed_query


@pytest.fixture(autouse=True)
def _clear_cache():
    emb_mod._embedding_cache.clear()
    yield
    emb_mod._embedding_cache.clear()


class FakeClient:
    """OpenAI-compatible embeddings client recording create() calls."""

    def __init__(self, fail_batch=False, fail_texts=()):
        self.calls = []
        self.fail_batch = fail_batch
        self.fail_texts = set(fail_texts)
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model, input, **kwargs):
        self.calls.append(input)
        if isinstance(input, list):
            if self.fail_batch and len(input) > 1:
                raise RuntimeError("batch too large")
            texts = input
        else:
            texts = [input]
        for t in texts:
            if t in self.fail_texts:
                raise RuntimeError(f"cannot embed {t!r}")
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(t))]) for t in texts]
        )


class TestEmbedQuery:
    def test_second_identical_call_is_served_from_cache(self):
        client = FakeClient()
        v1 = embed_query(client, "Barack Obama", model="m")
        v2 = embed_query(client, "barack obama", model="m")  # case-insensitive key
        assert v1 == v2
        assert len(client.calls) == 1

    def test_newlines_are_normalized(self):
        client = FakeClient()
        embed_query(client, "barack\nobama", model="m")
        assert client.calls[0] == ["barack obama"]

    def test_cache_is_keyed_by_model(self):
        client = FakeClient()
        embed_query(client, "obama", model="m1")
        embed_query(client, "obama", model="m2")
        assert len(client.calls) == 2


class TestEmbedQueries:
    def test_batches_misses_in_one_call(self):
        client = FakeClient()
        results = embed_queries(client, ["a", "bb", "ccc"], model="m")
        assert results == {"a": [1.0], "bb": [2.0], "ccc": [3.0]}
        assert client.calls == [["a", "bb", "ccc"]]

    def test_cache_hits_are_not_re_requested(self):
        client = FakeClient()
        embed_query(client, "a", model="m")
        results = embed_queries(client, ["a", "bb"], model="m")
        assert results["a"] == [1.0]
        # Only the miss went out in the batch call.
        assert client.calls == [["a"], ["bb"]]

    def test_batch_failure_falls_back_to_sequential(self):
        client = FakeClient(fail_batch=True)
        results = embed_queries(client, ["a", "bb"], model="m")
        assert results == {"a": [1.0], "bb": [2.0]}
        # 1 failed batch + 2 sequential calls
        assert len(client.calls) == 3

    def test_unembeddable_text_maps_to_none(self):
        client = FakeClient(fail_batch=True, fail_texts={"bad"})
        results = embed_queries(client, ["a", "bad"], model="m")
        assert results["a"] == [1.0]
        assert results["bad"] is None

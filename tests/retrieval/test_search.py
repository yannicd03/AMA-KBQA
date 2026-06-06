"""Tests for the shared retrieval module (ama_kbqa/retrieval/search.py).

Hermetic: a fake Qdrant client records the request shapes; the reranker is
stubbed at the ama_kbqa.retrieval.reranker module boundary so torch /
sentence-transformers is never imported.
"""
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
from loguru import logger
from qdrant_client import models

# NOTE: `import ama_kbqa.retrieval.search as search_mod` would bind the
# re-exported search() FUNCTION (package __init__ shadows the submodule
# attribute), so resolve the module through importlib.
search_mod = importlib.import_module("ama_kbqa.retrieval.search")
from ama_kbqa.retrieval.search import (
    BM25_MODEL,
    BM25_SPARSE_VECTOR_NAME,
    RetrievalParams,
    build_retrieval_params,
    search,
    search_terms,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    search_mod.reset_capability_cache()
    yield
    search_mod.reset_capability_cache()


def _point(pid: int, score: float, name: str = "doc") -> models.ScoredPoint:
    return models.ScoredPoint(id=pid, version=0, score=score, payload={"name": name})


def _params(**overrides) -> RetrievalParams:
    defaults = dict(
        limit=5,
        score_threshold=0.6,
        hybrid_enabled=False,
        fusion="rrf",
        prefetch_limit=20,
        reranker_enabled=False,
        reranker_model="stub-model",
        rerank_candidates=20,
        rerank_threshold=None,
    )
    defaults.update(overrides)
    return RetrievalParams(**defaults)


class FakeQdrant:
    """Records query_points kwargs and serves canned points."""

    def __init__(self, points=None, has_bm25=True, get_collection_error=None):
        self.points = points if points is not None else [_point(1, 0.9)]
        self.has_bm25 = has_bm25
        self.get_collection_error = get_collection_error
        self.query_calls = []
        self.get_collection_calls = 0

    def get_collection(self, collection_name):
        self.get_collection_calls += 1
        if self.get_collection_error is not None:
            raise self.get_collection_error
        sparse = {BM25_SPARSE_VECTOR_NAME: object()} if self.has_bm25 else {}
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(sparse_vectors=sparse))
        )

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=list(self.points))


class TestDenseRequestShape:
    def test_dense_only_request(self):
        qdrant = FakeQdrant()
        vector = [0.1, 0.2]
        hits = search(
            qdrant, "col",
            query_text="obama", query_vector=vector,
            params=_params(hybrid_enabled=False),
        )
        assert len(qdrant.query_calls) == 1
        call = qdrant.query_calls[0]
        assert call["collection_name"] == "col"
        assert call["query"] == vector
        assert call["limit"] == 5
        assert call["score_threshold"] == 0.6
        assert "prefetch" not in call
        assert hits == qdrant.points

    def test_dense_mode_never_inspects_collection(self):
        # With hybrid disabled there must be no get_collection round-trip.
        qdrant = FakeQdrant()
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(hybrid_enabled=False))
        assert qdrant.get_collection_calls == 0

    def test_query_filter_passthrough(self):
        qdrant = FakeQdrant()
        flt = models.Filter(should=[])
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(), query_filter=flt)
        assert qdrant.query_calls[0]["query_filter"] is flt


class TestHybridRequestShape:
    def test_hybrid_prefetch_and_fusion(self):
        qdrant = FakeQdrant(has_bm25=True)
        vector = [0.1, 0.2]
        search(qdrant, "col", query_text="barack obama", query_vector=vector,
               params=_params(hybrid_enabled=True))
        call = qdrant.query_calls[0]

        dense_branch, bm25_branch = call["prefetch"]
        assert dense_branch.using == ""
        assert dense_branch.query == vector
        assert dense_branch.limit == 20
        # The cosine threshold gates ONLY the dense branch.
        assert dense_branch.score_threshold == 0.6

        assert bm25_branch.using == BM25_SPARSE_VECTOR_NAME
        assert isinstance(bm25_branch.query, models.Document)
        assert bm25_branch.query.text == "barack obama"
        assert bm25_branch.query.model == BM25_MODEL
        assert bm25_branch.score_threshold is None

        assert call["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
        assert call["limit"] == 5
        assert "score_threshold" not in call  # fused scores are rank-based

    def test_dbsf_fusion_toggle(self):
        qdrant = FakeQdrant(has_bm25=True)
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(hybrid_enabled=True, fusion="dbsf"))
        assert qdrant.query_calls[0]["query"] == models.FusionQuery(
            fusion=models.Fusion.DBSF
        )


class TestFallback:
    def test_falls_back_to_dense_without_sparse_index(self):
        qdrant = FakeQdrant(has_bm25=False)
        warnings = []
        sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        try:
            search(qdrant, "old-col", query_text="q", query_vector=[0.1],
                   params=_params(hybrid_enabled=True))
        finally:
            logger.remove(sink)
        call = qdrant.query_calls[0]
        assert "prefetch" not in call
        assert call["query"] == [0.1]
        assert any("old-col" in w and "falling back" in w for w in warnings)

    def test_fallback_warning_only_once_per_collection(self):
        qdrant = FakeQdrant(has_bm25=False)
        warnings = []
        sink = logger.add(lambda m: warnings.append(str(m)), level="WARNING")
        try:
            for _ in range(3):
                search(qdrant, "old-col", query_text="q", query_vector=[0.1],
                       params=_params(hybrid_enabled=True))
        finally:
            logger.remove(sink)
        assert len([w for w in warnings if "falling back" in w]) == 1

    def test_capability_check_is_cached(self):
        qdrant = FakeQdrant(has_bm25=True)
        for _ in range(3):
            search(qdrant, "col", query_text="q", query_vector=[0.1],
                   params=_params(hybrid_enabled=True))
        assert qdrant.get_collection_calls == 1

    def test_transient_inspection_error_is_not_cached(self):
        qdrant = FakeQdrant(has_bm25=True,
                            get_collection_error=RuntimeError("down"))
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(hybrid_enabled=True))
        assert "prefetch" not in qdrant.query_calls[0]  # fell back to dense

        qdrant.get_collection_error = None  # next call re-inspects
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(hybrid_enabled=True))
        assert "prefetch" in qdrant.query_calls[1]


class TestReranker:
    def _stub_rerank(self, monkeypatch, scores_fn):
        calls = []

        def fake_rerank(query, docs, model_name):
            calls.append((query, list(docs), model_name))
            return scores_fn(docs)

        monkeypatch.setattr("ama_kbqa.retrieval.reranker.rerank", fake_rerank)
        return calls

    def test_rerank_reorders_and_overwrites_scores(self, monkeypatch):
        # Cross-encoder inverts the retrieval order.
        calls = self._stub_rerank(
            monkeypatch, lambda docs: list(range(len(docs)))
        )
        points = [_point(1, 0.9, "a"), _point(2, 0.8, "b"), _point(3, 0.7, "c")]
        qdrant = FakeQdrant(points=points)
        hits = search(qdrant, "col", query_text="q", query_vector=[0.1],
                      params=_params(reranker_enabled=True, limit=3))
        assert [h.id for h in hits] == [3, 2, 1]
        assert [h.score for h in hits] == [2.0, 1.0, 0.0]  # rerank scores
        assert calls[0][0] == "q"
        assert calls[0][1] == ["a", "b", "c"]
        assert calls[0][2] == "stub-model"

    def test_rerank_overfetches_then_truncates_to_limit(self, monkeypatch):
        self._stub_rerank(monkeypatch, lambda docs: list(range(len(docs))))
        points = [_point(i, 1.0 - i / 10) for i in range(10)]
        qdrant = FakeQdrant(points=points)
        hits = search(qdrant, "col", query_text="q", query_vector=[0.1],
                      params=_params(reranker_enabled=True, limit=2,
                                     rerank_candidates=10))
        assert qdrant.query_calls[0]["limit"] == 10  # over-fetched pool
        assert len(hits) == 2
        assert [h.id for h in hits] == [9, 8]  # best rerank scores

    def test_rerank_threshold_drops_low_scores(self, monkeypatch):
        self._stub_rerank(monkeypatch, lambda docs: [5.0, -1.0])
        points = [_point(1, 0.9, "a"), _point(2, 0.8, "b")]
        qdrant = FakeQdrant(points=points)
        hits = search(qdrant, "col", query_text="q", query_vector=[0.1],
                      params=_params(reranker_enabled=True,
                                     rerank_threshold=0.0))
        assert [h.id for h in hits] == [1]

    def test_rerank_runs_on_hybrid_candidates_too(self, monkeypatch):
        calls = self._stub_rerank(monkeypatch,
                                  lambda docs: list(range(len(docs))))
        points = [_point(1, 0.9, "a"), _point(2, 0.8, "b")]
        qdrant = FakeQdrant(points=points, has_bm25=True)
        hits = search(qdrant, "col", query_text="q", query_vector=[0.1],
                      params=_params(hybrid_enabled=True,
                                     reranker_enabled=True))
        assert "prefetch" in qdrant.query_calls[0]  # hybrid path taken
        assert len(calls) == 1  # reranker ran
        assert [h.id for h in hits] == [2, 1]

    def test_rerank_skipped_for_single_candidate(self, monkeypatch):
        calls = self._stub_rerank(monkeypatch, lambda docs: [1.0])
        qdrant = FakeQdrant(points=[_point(1, 0.9)])
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(reranker_enabled=True))
        assert calls == []

    def test_rerank_failure_keeps_retrieval_order(self, monkeypatch):
        def fail(query, docs, model_name):
            raise RuntimeError("model exploded")

        monkeypatch.setattr("ama_kbqa.retrieval.reranker.rerank", fail)
        points = [_point(1, 0.9, "a"), _point(2, 0.8, "b")]
        qdrant = FakeQdrant(points=points)
        hits = search(qdrant, "col", query_text="q", query_vector=[0.1],
                      params=_params(reranker_enabled=True))
        assert [h.id for h in hits] == [1, 2]
        assert [h.score for h in hits] == [0.9, 0.8]  # untouched


class TestToggleMatrix:
    """All four hybrid x rerank combinations select the right branches."""

    @pytest.mark.parametrize("hybrid", [False, True])
    @pytest.mark.parametrize("rerank", [False, True])
    def test_combination(self, monkeypatch, hybrid, rerank):
        rerank_calls = []
        monkeypatch.setattr(
            "ama_kbqa.retrieval.reranker.rerank",
            lambda q, docs, m: rerank_calls.append(1) or [0.0] * len(docs),
        )
        points = [_point(1, 0.9, "a"), _point(2, 0.8, "b")]
        qdrant = FakeQdrant(points=points, has_bm25=True)
        search(qdrant, "col", query_text="q", query_vector=[0.1],
               params=_params(hybrid_enabled=hybrid, reranker_enabled=rerank))
        assert ("prefetch" in qdrant.query_calls[0]) == hybrid
        assert bool(rerank_calls) == rerank


class TestBuildRetrievalParams:
    def test_defaults_from_config(self, monkeypatch):
        monkeypatch.setattr("ama_kbqa.config.get_hybrid_enabled", lambda: True)
        monkeypatch.setattr("ama_kbqa.config.get_fusion", lambda: "dbsf")
        monkeypatch.setattr("ama_kbqa.config.get_prefetch_limit", lambda: 33)
        monkeypatch.setattr("ama_kbqa.config.get_reranker_enabled", lambda: True)
        monkeypatch.setattr("ama_kbqa.config.get_reranker_model", lambda: "m")
        monkeypatch.setattr("ama_kbqa.config.get_rerank_candidates", lambda: 7)
        monkeypatch.setattr("ama_kbqa.config.get_rerank_threshold", lambda: 0.5)

        params = build_retrieval_params(limit=4, score_threshold=0.9)
        assert params == RetrievalParams(
            limit=4, score_threshold=0.9, hybrid_enabled=True, fusion="dbsf",
            prefetch_limit=33, reranker_enabled=True, reranker_model="m",
            rerank_candidates=7, rerank_threshold=0.5,
        )

    def test_allow_rerank_false_overrides_config(self, monkeypatch):
        # The orchestrator probe path: rerank stays off even when enabled.
        monkeypatch.setattr("ama_kbqa.config.get_reranker_enabled", lambda: True)
        params = build_retrieval_params(
            limit=5, score_threshold=None, allow_rerank=False
        )
        assert params.reranker_enabled is False


class TestSearchTerms:
    def test_skips_falsy_vectors_and_maps_errors_to_empty(self):
        class FlakyQdrant(FakeQdrant):
            def query_points(self, **kwargs):
                super().query_points(**kwargs)
                if kwargs["query"] == [9.9]:
                    raise RuntimeError("boom")
                return SimpleNamespace(points=list(self.points))

        qdrant = FlakyQdrant()
        results = search_terms(
            qdrant, "col",
            vectors_map={"good": [0.1], "missing": None, "bad": [9.9]},
            params=_params(),
        )
        assert set(results) == {"good", "bad"}  # "missing" skipped entirely
        assert results["good"] == qdrant.points
        assert results["bad"] == []


@pytest.mark.skipif(
    importlib.util.find_spec("sentence_transformers") is not None,
    reason="rerank extra installed; ImportError hint not reachable",
)
class TestRerankerImportHint:
    def test_missing_extra_raises_clear_hint(self):
        from ama_kbqa.retrieval import reranker

        with pytest.raises(ImportError, match="uv sync --extra rerank"):
            reranker.get_reranker("any-model")

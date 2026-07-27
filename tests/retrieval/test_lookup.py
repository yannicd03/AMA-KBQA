"""Tests for the shared lexical lookup module (ama_kbqa/retrieval/lookup.py).

Hermetic: a fake Qdrant client evaluates `Filter(should=[FieldCondition(...)])`
against an in-memory point list, mirroring how a real Qdrant `scroll` call
would answer it, without talking to a live collection. This lets the tests
exercise the actual matching logic (exact / contains / prefix, ambiguity,
n-gram probing, the probe cap) rather than just asserting on request shapes.
"""

import pytest
from qdrant_client import models

from ama_kbqa.retrieval.lookup import (
    MAX_NGRAM_PROBES,
    LookupResult,
    _ngram_candidates,
    lookup_by_label,
    normalize_label,
)


class FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


def _field_matches(payload: dict, condition: models.FieldCondition) -> bool:
    value = payload.get(condition.key)
    match = condition.match
    if isinstance(match, models.MatchValue):
        return value == match.value
    return False


class FakeQdrant:
    """Evaluates should-filters against an in-memory point list.

    Records every scroll call's should-values so tests can assert on how
    many probes were issued and what strings they tried, without depending
    on internal call ordering beyond what's asserted explicitly.
    """

    def __init__(self, points):
        self.points = points
        self.scroll_calls = []

    def scroll(self, collection_name, scroll_filter, limit, with_payload=True):
        should = scroll_filter.should or []
        probed_values = [c.match.value for c in should if isinstance(c.match, models.MatchValue)]
        self.scroll_calls.append({"limit": limit, "values": probed_values})
        matched = [
            p for p in self.points
            if any(_field_matches(p.payload or {}, c) for c in should)
        ]
        return matched[:limit], None


# =============================================================================
# normalize_label
# =============================================================================

def test_normalize_label_collapses_whitespace_and_case():
    assert normalize_label("  Texas   Metro ") == "texas metro"
    assert normalize_label("texas metro") == "texas metro"


# =============================================================================
# mode="exact"
# =============================================================================

def test_exact_single_match_not_ambiguous():
    qdrant = FakeQdrant([FakePoint(1, {"name": "Duran Duran"})])
    result = lookup_by_label(qdrant, "col", "Duran Duran", mode="exact", limit=50)

    assert result.total_matched == 1
    assert not result.ambiguous
    assert result.exhaustive
    assert result.matches[0].payload["name"] == "Duran Duran"


def test_exact_ambiguous_duplicate_labels():
    """13 KG entities named 'Roger Moore' — the historical KQAPro bug fix
    this module preserves: all must be reachable, and ambiguity flagged."""
    points = [FakePoint(i, {"name": "Roger Moore"}) for i in range(13)]
    qdrant = FakeQdrant(points)

    result = lookup_by_label(qdrant, "col", "Roger Moore", mode="exact", limit=50)

    assert result.total_matched == 13
    assert result.ambiguous
    assert not result.truncated
    assert result.exhaustive


def test_exact_truncated_at_limit():
    points = [FakePoint(i, {"name": "Common Name"}) for i in range(60)]
    qdrant = FakeQdrant(points)

    result = lookup_by_label(qdrant, "col", "Common Name", mode="exact", limit=50)

    assert result.total_matched == 50
    assert result.truncated
    assert not result.exhaustive


def test_exact_extra_fields_matches_original_id():
    """KQAPro's Phase 1 also matches on original_id / attributes.value.value."""
    qdrant = FakeQdrant([FakePoint(1, {"name": "Something Else", "original_id": "Q42"})])

    result = lookup_by_label(
        qdrant, "col", "Q42", mode="exact",
        label_field="name", extra_fields=["original_id"], limit=50,
    )

    assert result.total_matched == 1
    assert result.matches[0].payload["original_id"] == "Q42"


def test_exact_is_case_sensitive_by_default():
    """Default (normalize=False) preserves KQAPro's historical byte-exact
    behavior — no silent case-insensitivity change for the refactor."""
    qdrant = FakeQdrant([FakePoint(1, {"name": "Texas"})])

    result = lookup_by_label(qdrant, "col", "texas", mode="exact", limit=50)

    assert result.total_matched == 0


def test_exact_normalize_true_catches_case_variant():
    qdrant = FakeQdrant([FakePoint(1, {"name": "Texas"})])

    result = lookup_by_label(qdrant, "col", "texas", mode="exact", normalize=True, limit=50)

    assert result.total_matched == 1


def test_exact_empty_text_short_circuits():
    qdrant = FakeQdrant([FakePoint(1, {"name": "Texas"})])

    result = lookup_by_label(qdrant, "col", "   ", mode="exact")

    assert result.total_matched == 0
    assert result.exhaustive
    assert qdrant.scroll_calls == []


def test_unknown_mode_raises():
    qdrant = FakeQdrant([])
    with pytest.raises(ValueError):
        lookup_by_label(qdrant, "col", "x", mode="fuzzy")


# =============================================================================
# mode="prefix" / "contains" — the over-specified-mention case
# =============================================================================

def test_prefix_mode_drops_trailing_qualifier_film():
    """The motivating case: agent searches 'Abraham Lincoln (film)', gold
    entity label is 'Abraham Lincoln'. A byte-exact match on the whole
    string never fires; prefix n-gram probing must find it."""
    qdrant = FakeQdrant([FakePoint(1, {"name": "Abraham Lincoln"})])

    result = lookup_by_label(qdrant, "col", "Abraham Lincoln (film)", mode="prefix", limit=50)

    assert result.total_matched == 1
    assert result.matched_text == "Abraham Lincoln"
    # Longest-first means the full string is tried before the correct
    # 2-token prefix, and probing continues through all 3 candidates
    # ("Abraham Lincoln (film)", "Abraham Lincoln", "Abraham") rather than
    # stopping at the first hit — the fix under test.
    assert len(qdrant.scroll_calls) == 3


def test_prefix_mode_drops_trailing_qualifier_metropolitan_area():
    """The other motivating case: 'Texas metropolitan area' -> gold 'Texas'."""
    qdrant = FakeQdrant([FakePoint(1, {"name": "Texas"})])

    result = lookup_by_label(qdrant, "col", "Texas metropolitan area", mode="prefix", limit=50)

    assert result.total_matched == 1
    assert result.matched_text == "Texas"
    # length=3 (full string), length=2 ("Texas metropolitan"), length=1 ("Texas").
    assert len(qdrant.scroll_calls) == 3


def test_contains_mode_matches_interior_substring():
    """'contains' tries every start position, so a label buried mid-query
    (not just a prefix) is still reachable."""
    qdrant = FakeQdrant([FakePoint(1, {"name": "metropolitan"})])

    result = lookup_by_label(qdrant, "col", "Texas metropolitan area", mode="contains", limit=50)

    assert result.total_matched == 1
    assert result.matched_text == "metropolitan"


def test_contains_mode_no_match_is_not_exhaustive_when_capped():
    """A long query with no matching label anywhere hits MAX_NGRAM_PROBES
    before trying every possible n-gram; absence must not be reported as
    proven (exhaustive=False)."""
    long_query = " ".join(f"word{i}" for i in range(10))  # 10 tokens, 55 possible n-grams
    qdrant = FakeQdrant([])  # nothing in the KB matches anything

    result = lookup_by_label(qdrant, "col", long_query, mode="contains", limit=50)

    assert result.total_matched == 0
    assert not result.exhaustive
    assert len(qdrant.scroll_calls) == MAX_NGRAM_PROBES


def test_contains_mode_short_query_no_match_is_exhaustive():
    """A short query where every n-gram was actually tried (never hit the
    cap) correctly reports exhaustive=True even though nothing matched."""
    qdrant = FakeQdrant([])

    result = lookup_by_label(qdrant, "col", "two words", mode="contains", limit=50)

    assert result.total_matched == 0
    assert result.exhaustive
    assert len(qdrant.scroll_calls) < MAX_NGRAM_PROBES


def test_ambiguous_result_from_ngram_probe():
    points = [FakePoint(i, {"name": "Texas"}) for i in range(3)]
    qdrant = FakeQdrant(points)

    result = lookup_by_label(qdrant, "col", "Texas metropolitan area", mode="prefix", limit=50)

    assert result.ambiguous
    assert result.total_matched == 3


# =============================================================================
# Regression: a longer-but-wrong n-gram must not shadow a correct shorter
# one (or vice versa) — every probe that hits contributes its matches.
# =============================================================================

def test_contains_mode_surfaces_shorter_match_alongside_longer_one():
    """Reproduces the reported defect against 'Texas metropolitan area':
    the KB has an (irrelevant) entity literally named 'metropolitan area'
    (a 2-token n-gram, probed before the 1-token 'Texas') AND the correct
    entity 'Texas'. The old "first hit wins" logic returned only
    'metropolitan area' and silently dropped 'Texas'. Both must come back."""
    qdrant = FakeQdrant([
        FakePoint(1, {"name": "metropolitan area"}),
        FakePoint(2, {"name": "Texas"}),
    ])

    result = lookup_by_label(qdrant, "col", "Texas metropolitan area", mode="contains", limit=50)

    names = {p.payload["name"] for p in result.matches}
    assert "Texas" in names
    assert "metropolitan area" in names
    assert result.total_matched == 2
    assert result.ambiguous
    # matched_text reports a representative (the longest matching n-gram),
    # not proof it's the only one — callers must check `matches`/`ambiguous`.
    assert result.matched_text == "metropolitan area"


def test_prefix_mode_surfaces_shorter_match_alongside_longer_literal_one():
    """Reproduces the same class of defect for 'Abraham Lincoln (film)':
    the KB has a literal entity named 'Abraham Lincoln (film)' itself (the
    full 3-token string, probed first) AND the correct 'Abraham Lincoln'
    (the 2-token prefix). Both must come back, not just the longer one."""
    qdrant = FakeQdrant([
        FakePoint(1, {"name": "Abraham Lincoln (film)"}),
        FakePoint(2, {"name": "Abraham Lincoln"}),
    ])

    result = lookup_by_label(qdrant, "col", "Abraham Lincoln (film)", mode="prefix", limit=50)

    names = {p.payload["name"] for p in result.matches}
    assert "Abraham Lincoln" in names
    assert "Abraham Lincoln (film)" in names
    assert result.total_matched == 2
    assert result.ambiguous
    assert result.matched_text == "Abraham Lincoln (film)"


# =============================================================================
# _ngram_candidates (internal helper, tested directly for ordering/stopwords)
# =============================================================================

def test_ngram_candidates_prefix_only_ordering():
    candidates = _ngram_candidates("Abraham Lincoln (film)", prefix_only=True, max_candidates=10)
    assert candidates == ["Abraham Lincoln (film)", "Abraham Lincoln", "Abraham"]


def test_ngram_candidates_contains_all_positions():
    candidates = _ngram_candidates("a b c", prefix_only=False, max_candidates=10)
    # length 3: "a b c"; length 2: "a b", "b c"; length 1: "b", "c" ("a" is a stopword)
    assert candidates == ["a b c", "a b", "b c", "b", "c"]


def test_ngram_candidates_skips_single_token_stopwords():
    candidates = _ngram_candidates("the area", prefix_only=False, max_candidates=10)
    # length 2: "the area"; length 1: "the" is skipped (stopword), "area" kept.
    assert "the" not in candidates
    assert "area" in candidates


def test_ngram_candidates_respects_cap():
    text = " ".join(f"w{i}" for i in range(10))
    candidates = _ngram_candidates(text, prefix_only=False, max_candidates=5)
    assert len(candidates) == 5


# =============================================================================
# LookupResult.ambiguous
# =============================================================================

def test_lookup_result_ambiguous_property():
    assert LookupResult(matches=[1, 2], mode="exact", total_matched=2, truncated=False, exhaustive=True).ambiguous
    assert not LookupResult(matches=[1], mode="exact", total_matched=1, truncated=False, exhaustive=True).ambiguous
    assert not LookupResult(matches=[], mode="exact", total_matched=0, truncated=False, exhaustive=True).ambiguous

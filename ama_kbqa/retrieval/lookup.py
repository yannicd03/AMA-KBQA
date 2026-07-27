"""Shared lexical (non-vector, non-BM25) label lookup for MCP servers.

Why this exists: the KQAPro/SciQA retrieval A/B (2026-07) found the dominant
retrieval failure is not "search finds nothing" but "search finds k
CONFIDENT-but-WRONG things" — gated dense search returns a full result set
at cosine 0.86-1.00 with the correct entity simply absent. No score
threshold can detect that; only the caller (an automatic exact-match phase,
or the agent recognising its own search went wrong) can. This module is the
one place both failure-recovery paths call into.

Two matching strategies are supported, and they solve two DIFFERENT
problems — do not conflate them:

- ``mode="exact"`` — a single Qdrant ``scroll`` with a payload
  ``Filter(should=[MatchValue(...), ...])`` across one or more fields. This
  is the exact technique KQAPro's original ``_find_node_impl`` Phase 1
  used; it is a server-side filtered scan, cheap in NETWORK terms (only
  matching points come back) but still an O(collection) server-side scan
  when the collection carries no payload index (see
  ``db/migrate_add_label_index.py`` — neither ``kqapro-entities`` nor
  ``sciqa-entities`` has one today). This is the mode the AUTOMATIC
  exact-match phase inside ``FindNode``/``FindResource`` uses on every
  call, same as KQAPro already unconditionally pays for; it costs exactly
  ONE scan, never multiplied.

- ``mode="contains"`` / ``mode="prefix"`` — solve the OVER-SPECIFIED-MENTION
  case, which ``exact`` structurally cannot: the agent searched "Texas
  metropolitan area" when the gold entity's label is "Texas", or "Abraham
  Lincoln (film)" when the gold label is "Abraham Lincoln". The entity's
  LABEL is a substring of the QUERY, not the other way around — this is
  the opposite of what a full-text index finds, so it cannot be expressed
  as a single ``MatchText`` filter. Instead we enumerate contiguous token
  n-grams of the query text, longest first, and issue a bounded number of
  cheap ``exact`` probes (one per n-gram) — collecting the matches from
  EVERY probe that hits, not just the first. Stopping at the first hit
  would let a longer-but-irrelevant n-gram silently shadow the correct,
  shorter one (e.g. if some other KB entity happens to be labelled
  "metropolitan area", that 2-token probe fires before the 1-token "Texas"
  probe and — if only the first hit were kept — would hide the entity the
  caller actually wanted). So all hits are unioned (deduped by point id,
  longest-n-gram-first) and returned together, exactly like the
  ``disambiguation_notice`` path does for duplicate exact labels: the agent
  gets the candidate set and disambiguates, instead of a heuristic
  discarding the right answer. ``prefix`` only tries n-grams anchored at
  the start of the query (dropping trailing tokens: "Texas metropolitan
  area" -> "Texas metropolitan" -> "Texas"), which is cheaper and covers
  the common case; ``contains`` tries every start position too (catches a
  label buried mid-query), at the cost of more probes.

  Each probe is a full ``exact``-mode scan when no payload index exists,
  so an n-gram walk genuinely multiplies scan cost by up to
  ``MAX_NGRAM_PROBES`` — this is exactly why the keyword payload index in
  ``db/migrate_add_label_index.py`` is worth adding (an optimisation, not a
  prerequisite: both modes work without it today, just slower). The probe
  count is bounded by a named constant regardless of query length so one
  lookup call can never turn into an unbounded number of collection scans.
  This mode is for the EXPLICIT lookup tool (an agent decides to pay for it
  when it already suspects the semantic hits are wrong), not the automatic
  per-call phase.

Neither mode touches vectors or BM25 — this module has no reranker/
embedding dependency, only ``qdrant_client``, so it stays importable and
unit-testable without pulling in the rest of the retrieval stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import QdrantClient, models

# Matching modes accepted by lookup_by_label().
_VALID_MODES = ("exact", "contains", "prefix")

# Single-token n-gram candidates this common are cheap to match against
# almost anything and would burn a probe on a near-certain false lead
# ("area", "the", "of" ...); skip them as standalone single-token probes.
# They still participate in longer n-grams, so a real label containing one
# of these words is unaffected.
_NGRAM_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "by", "and",
    "or", "is", "was", "are", "with", "from",
})

# Hard cap on exact-match probes issued per lookup_by_label(mode="contains"
# or "prefix") call. Each probe is a Filter(should=[MatchValue]) scroll —
# cheap with the keyword payload index from db/migrate_add_label_index.py,
# but an O(collection) scan without it (same cost class as the existing
# automatic exact phase). The cap bounds worst-case latency to
# MAX_NGRAM_PROBES scans regardless of how many words the caller's query
# text has.
MAX_NGRAM_PROBES = 12


def normalize_label(text: str) -> str:
    """Case- and whitespace-normalised form of a label for comparison/tests.

    Collapses internal whitespace runs to a single space (in addition to
    trimming) and lower-cases, so "  Texas   Metro " and "texas metro"
    compare equal. Not used to change Qdrant filter semantics by default
    (see ``normalize=`` on :func:`lookup_by_label`) — it is a plain string
    utility for callers/tests that need to compare labels directly.
    """
    return " ".join(text.split()).lower()


@dataclass(frozen=True)
class LookupResult:
    """Structured result of a lexical label lookup.

    Attributes:
        matches: Raw Qdrant records (``.payload`` accessible). For "exact",
            in the order Qdrant returned them. For "contains"/"prefix",
            the UNION of every probe that matched anything, deduped by
            point id, ordered longest-matching-n-gram-first — this is
            deliberately not just the single longest hit (see
            ``_lookup_ngram``'s docstring for why collapsing to one would
            silently discard a correct shorter match).
        mode: The matching mode that actually ran ("exact", "contains", or
            "prefix").
        total_matched: Number of points present in ``matches``.
        truncated: True when ``total_matched`` hit the caller's ``limit``,
            so there may be additional matches beyond what's returned.
        exhaustive: True when the lookup covered everything it was
            entitled to check: for "exact", the filter ran to completion
            (i.e. not truncated); for "contains"/"prefix", every planned
            n-gram probe was issued (the ``MAX_NGRAM_PROBES`` cap was not
            hit before the candidate list was exhausted) AND the combined
            result wasn't truncated. False means the absence of a match is
            NOT proof the label doesn't exist elsewhere — only that the
            bounded search didn't find it.
        matched_text: For "contains"/"prefix", the LONGEST n-gram that
            matched anything (``None`` for "exact", or when nothing
            matched) — a representative label, not necessarily the only
            one: check ``matches``/``ambiguous`` for the full candidate set.
    """

    matches: List[object]
    mode: str
    total_matched: int
    truncated: bool
    exhaustive: bool
    matched_text: Optional[str] = None

    @property
    def ambiguous(self) -> bool:
        """True when more than one point matched the label lookup.

        Vector similarity can't discriminate between byte-identical (or
        near-identical) labels, so callers should treat this as a signal
        to surface a disambiguation notice rather than trust the first
        match, exactly as KQAPro's original Phase 1 handling did.
        """
        return self.total_matched > 1


def _case_variants(text: str) -> List[str]:
    """A small set of case/whitespace variants for lenient exact matching.

    Still a single server-side scan: these are additional ``should``
    (OR) conditions in the SAME filter, not extra scroll calls.
    """
    variants = {text, text.title(), text.lower(), text.upper(), " ".join(text.split())}
    return [v for v in variants if v]


def _lookup_exact(
    qdrant: QdrantClient,
    collection_name: str,
    text_clean: str,
    label_field: str,
    extra_fields: Optional[List[str]],
    limit: int,
    normalize: bool,
) -> LookupResult:
    fields = [label_field, *(extra_fields or [])]
    values = _case_variants(text_clean) if normalize else [text_clean]
    should_conditions = [
        models.FieldCondition(key=field, match=models.MatchValue(value=value))
        for field in fields
        for value in values
    ]
    scroll_results, _ = qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(should=should_conditions),
        limit=limit,
        with_payload=True,
    )
    matches = list(scroll_results)
    truncated = len(matches) >= limit
    return LookupResult(
        matches=matches,
        mode="exact",
        total_matched=len(matches),
        truncated=truncated,
        exhaustive=not truncated,
    )


def _ngram_candidates(text: str, *, prefix_only: bool, max_candidates: int) -> List[str]:
    """Contiguous token n-grams of ``text``, longest first.

    At a given length, the window anchored at the start of the text is
    tried before any other start position — dropping a single trailing
    qualifier (the common over-specification case: "Texas metropolitan
    area" -> "Texas metropolitan") is tried before anything else at that
    length. ``prefix_only`` restricts to exactly that anchored-at-start
    family (cheaper: one candidate per length instead of one per
    start position).
    """
    tokens = text.split()
    n = len(tokens)
    candidates: List[str] = []
    for length in range(n, 0, -1):
        starts = [0] if prefix_only else range(0, n - length + 1)
        for start in starts:
            if length == 1 and tokens[start].lower() in _NGRAM_STOPWORDS:
                continue
            candidates.append(" ".join(tokens[start:start + length]))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _lookup_ngram(
    qdrant: QdrantClient,
    collection_name: str,
    text_clean: str,
    label_field: str,
    extra_fields: Optional[List[str]],
    limit: int,
    normalize: bool,
    max_probes: int,
    prefix_only: bool,
) -> LookupResult:
    """Probe n-gram candidates and return the UNION of every probe's hits.

    Earlier versions returned only the first (longest) candidate that
    matched anything, on the theory that "longest match wins". That is
    wrong and reproduces exactly the failure this module exists to fix:
    for "Texas metropolitan area", the 2-token candidate "metropolitan
    area" happens to probe before the 1-token "Texas" (longest-first
    ordering) and, if some OTHER KB entity is literally labelled
    "metropolitan area", that entity would silently shadow the correct
    "Texas" match the caller actually wanted — discarding the right answer
    because a heuristic preferred a longer but different string, the same
    class of bug the whole retrieval investigation is about. So every
    probe that hits contributes its matches to the result (deduped by
    point id, capped at `limit`, still bounded to `max_probes` probes
    total); the caller sees the full candidate set and disambiguates,
    exactly like the `disambiguation_notice` path does for duplicate exact
    labels. Probing still proceeds longest-n-gram-first so the most
    specific candidate's matches lead the list.
    """
    mode = "prefix" if prefix_only else "contains"
    all_candidates = _ngram_candidates(text_clean, prefix_only=prefix_only, max_candidates=max_probes + 1)
    capped = len(all_candidates) > max_probes
    probes = all_candidates[:max_probes]

    combined_matches: List[object] = []
    seen_ids: set = set()
    longest_hit: Optional[str] = None
    any_probe_truncated = False

    for candidate in probes:
        result = _lookup_exact(qdrant, collection_name, candidate, label_field, extra_fields, limit, normalize)
        if not result.total_matched:
            continue
        if longest_hit is None:
            # Candidates are generated longest-first, so the first probe to
            # hit anything IS the longest matching n-gram.
            longest_hit = candidate
        any_probe_truncated = any_probe_truncated or result.truncated
        for point in result.matches:
            point_id = getattr(point, "id", id(point))
            if point_id in seen_ids:
                continue
            seen_ids.add(point_id)
            combined_matches.append(point)
        if len(combined_matches) >= limit:
            # Already have as many distinct matches as the caller asked
            # for; further (shorter, less specific) probes would only be
            # truncated away below, so stop spending scans on them.
            break

    if longest_hit is None:
        return LookupResult(
            matches=[],
            mode=mode,
            total_matched=0,
            truncated=False,
            exhaustive=not capped,
        )

    truncated = any_probe_truncated or len(combined_matches) > limit
    return LookupResult(
        matches=combined_matches[:limit],
        mode=mode,
        total_matched=len(combined_matches[:limit]),
        truncated=truncated,
        exhaustive=(not capped) and not truncated,
        matched_text=longest_hit,
    )


def lookup_by_label(
    qdrant: QdrantClient,
    collection_name: str,
    text: str,
    *,
    mode: str = "exact",
    label_field: str = "name",
    extra_fields: Optional[List[str]] = None,
    limit: int = 50,
    normalize: bool = False,
    max_probes: int = MAX_NGRAM_PROBES,
) -> LookupResult:
    """Look up points by a lexical (non-vector, non-BM25) label match.

    Args:
        qdrant: Qdrant client.
        collection_name: Collection to search.
        text: The literal name/label/identifier (or, for "contains"/
            "prefix", the over-specified mention it's embedded in) to look
            up.
        mode: "exact" (default) — single server-side
            ``Filter(should=[MatchValue])`` scroll across ``label_field`` +
            ``extra_fields``. "contains"/"prefix" — bounded n-gram probing
            (see module docstring); only ever matches on ``label_field`` +
            ``extra_fields`` per probe, same as "exact".
        label_field: Payload field holding the display label ("name" for
            both the KQAPro and SciQA entity collections).
        extra_fields: Additional payload fields to also match in "exact"
            mode (e.g. KQAPro's ``original_id`` / ``attributes.value.value``).
        limit: Max points returned per successful match.
        normalize: When True, also match case/whitespace variants of each
            probed string (title-case, lower-case, upper-case, whitespace-
            collapsed) as extra OR conditions in the SAME filter call — no
            extra scan cost, just a wider net. Defaults to False so
            callers that need byte-for-byte parity with historical
            behavior (KQAPro's original Phase 1) get it by default.
        max_probes: Cap on n-gram probes for "contains"/"prefix" (default
            ``MAX_NGRAM_PROBES``). Ignored for "exact".

    Returns:
        LookupResult with the raw matches plus ambiguity/completeness
        signals for the caller to act on.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown lookup mode {mode!r}; expected one of {_VALID_MODES}")

    text_clean = text.strip()
    if not text_clean:
        return LookupResult(matches=[], mode=mode, total_matched=0, truncated=False, exhaustive=True)

    if mode == "exact":
        return _lookup_exact(qdrant, collection_name, text_clean, label_field, extra_fields, limit, normalize)

    return _lookup_ngram(
        qdrant, collection_name, text_clean, label_field, extra_fields, limit,
        normalize, max_probes, prefix_only=(mode == "prefix"),
    )

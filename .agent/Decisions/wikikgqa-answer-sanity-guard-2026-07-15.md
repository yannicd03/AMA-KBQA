# ADR: WikiKGQA Answer-Sanity Guard and Property-Namespace Normalization

**Date:** 2026-07-15
**Commit:** `fde0c62` "WikiKGQA: normalize property-namespace URIs in `_bare_id`, answer-sanity guard at commit time"
**Status:** Settled — implemented, 347 tests passing (+19)

## Related Docs
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md) — the journal-alternate recovery mechanism this ADR reuses for unsane (not just 0-row) results; also carried the now-fixed "inert bug" note this ADR resolves
- [Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](./wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md) — closure-expansion escalation now also gated on `_result_is_sane`, since a strict-superset candidate can be unsane; that ADR's Validation section is where the (at-the-time) inert `_bare_id` bug was first surfaced
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference; updated alongside this ADR with the sanity-guard stage in the commit-time pipeline
- **External:** wiki note on validating against reference-value grammar, not just observed reference data (see Key Lesson below)

---

## Background

The 2026-07-15 post-deadline resubmission scored WM (with-mentions) 0.85 →
0.84 (a regression) and WOM (without-mentions) 0.78 → 0.79 (an improvement).
The WM regression traced entirely to **q113**: an `AgentSparqlGenerator(votes=3)`
self-consistency round committed a query whose result was 50 rows of
`entity/statement/` URIs and `Special:EntityData` URLs, replacing what had
previously been a correct single-answer result. That one question accounts
for ~-0.013 macro F1 — the entire with-mentions regression.

The root cause is two compounding gaps:

1. `submission.py::_bare_id`'s prefix-strip loop matched the generic
   `.../prop/` prefix before ever checking the longer, more specific
   `.../prop/statement/` and `.../prop/qualifier/` prefixes, so those
   namespaced property URIs serialized into the submission as junk like
   `"statement/Pxxx"` instead of the bare `"Pxxx"`.
2. Nothing at commit time recognized that a result **with rows** could still
   be structurally wrong — the existing recovery logic (2026-07-03) only
   fires on 0 rows or execution errors, not on "50 rows, all garbage."

This bug had been noted in passing the day before
([Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md),
"Known inert bug") and assessed as **inert** because a scan of all 497
training gold answers found zero occurrences of these URI shapes. q113 is
exactly the counterexample: gold never produces these shapes, but the
*system's own generated queries* can and did.

## Decisions

### A. `_bare_id` normalizes `prop/statement/` and `prop/qualifier/` namespaces (`submission.py`)

**Chosen:** `_WD_URI_PREFIXES` in `submission.py` now lists
`http://www.wikidata.org/prop/statement/` and
`http://www.wikidata.org/prop/qualifier/` ahead of the generic
`http://www.wikidata.org/prop/`, mirroring `dataset.py::_PROPERTY_PREFIXES`
(most-specific first, generic last so it never shadows the statement/
qualifier/direct forms). Property-namespaced URIs now reduce to the bare
`Pxxx` correctly.

**Deliberately not fixed the same way:** entity **statement nodes**
(`http://www.wikidata.org/entity/statement/Q123-UUID`) are not entities and
have no correct bare form — there is no `Qxxx` a statement node can honestly
be laundered into. `_bare_id` leaves them with the `"statement/..."`
remainder attached rather than inventing a plausible-looking bare id. This
is intentional: a statement node passed through un-laundered is still wrong,
but it's *visibly* wrong (contains `"/"`) — see Decision B, which is what
actually catches it.

**Rejected alternative:** map statement nodes to their subject entity's bare
`Qxxx` (parsing the UUID-suffixed id back to its parent entity). Rejected —
a statement node in the answer position means the query itself asked for the
wrong thing (a fact-node, not an entity), and silently rewriting it to look
like a normal entity answer would hide that the query is structurally wrong
rather than surface it for recovery.

### B. Commit-time answer-sanity guard (`generator.py::_result_is_sane`)

**Chosen:** A committed result is now recognized as **known-wrong** (not
just "has rows, might be fine") when any binding is:
- a blank node (`type == "bnode"`),
- a `/entity/statement/` URI,
- a `Special:EntityData` URL, or
- (catch-all) still contains `"/"` after `to_codabench_answers`
  bare-id normalization — i.e. unmapped URI junk that survived Decision A.

`ASK` (boolean) results have no bindings to inspect and are always sane.

**Gold-scan evidence for the catch-all rule.** All 497 training answers in
`data/wikikgqa/wikikgqa.json` were checked: zero gold answer values contain
`"/"`. The `"/"` catch-all therefore has no false-positive risk against the
only ground truth available — a `"/"` surviving normalization is always
junk, never a legitimate literal/URL answer.

**Wired in at three points**, all reusing the sanity check rather than
introducing new recovery machinery:
1. **Journal-alternate recovery** (`_generate_once`): if the committed
   result has rows but is unsane, and the journal holds a different
   validated query, that alternate is tried — the same recovery path
   already used for the 0-row case (2026-07-03). The alternate is adopted
   only if it both has rows *and* is sane. If no sane alternate exists, the
   original (unsane) result is kept — unsane is not worse than empty (both
   score 0), and blanking it out gains nothing.
2. **Closure escalation** (class-closure expansion, 2026-07-14): a
   candidate that is a strict superset of the committed answer set is now
   adopted only if it is *also* sane. A widened pattern that happens to
   pull in a statement/blank node alongside genuine new rows must not
   replace a sane committed result just because it nominally "grew."
3. **Vote tie-breaks** (`votes=N` self-consistency): when multiple runs tie
   on vote count, the tie now resolves to a sane tied answer over an
   earlier-seen unsane one (previously: earliest-seen always won). This is
   the exact q113 failure mode — a 3-way vote where the winning tie-break
   picked the earliest-seen run, which happened to be the unsane one.

**Rejected alternative:** filter unsane rows out of the result and submit
the remainder. Rejected — a partially-filtered result set has no principled
stopping point (which rows are "real" vs collateral junk from the same
malformed query pattern is not determinable from the bindings alone), and
would risk quietly submitting a still-wrong partial answer instead of
triggering recovery.

## Validation

Suite: `uv run pytest` → **347 passed** (up from 328 as of the 2026-07-14
closure-expansion ADR), +19 tests covering `_bare_id` namespace stripping
(`tests/wikikgqa/test_submission.py`, new file) and `_result_is_sane` plus
its three call sites (`tests/wikikgqa/test_generator.py`).

## Key transferable lesson

The `_bare_id` bug was assessed **inert** on 2026-07-14 because gold never
contains `prop/statement/`/`prop/qualifier/` URIs — true, but the wrong
check. Gold is *observed reference data*; it doesn't bound what a live
system can produce. The system's own generated SPARQL queries are not
constrained to only ever touch the same URI shapes gold happens to use, and
in fact q113's committed query did not. **Validate system output against
the reference value's *grammar* (the set of URI/literal shapes a correct
answer can legally take), not just against the specific values observed in
the available reference data.** A zero-occurrence scan over gold answers is
evidence the bug hasn't been *observed* yet, not evidence it's *unreachable*.

This lesson is also recorded in the wiki
(`0_Claude/AMA-KBQA/`) as a cross-project note, since the "validate against
grammar, not observed data" pattern applies beyond this codebase.

# `LocateTerm` — Schema-Role Locator Tool

**Status:** 📋 Planned
**Priority:** High (highest of the deferred set — largest single addressable failure cluster)
**Identified:** 2026-07-25 (reasoning-error and capability-gap analysis, recommendation #1)

## Related Docs
- [../../Decisions/reasoning-error-analysis-2026-07-25.md](../../Decisions/reasoning-error-analysis-2026-07-25.md) — full analysis; §1 "Primary reasoning error: schema-role confusion" is the source finding for this task
- [../../Decisions/root-cause-tool-generalization-2026-05-14.md](../../Decisions/root-cause-tool-generalization-2026-05-14.md) — the 2026-05-15 follow-up already flagged that `GetAttributeDetails` doesn't expose qualifier existence; this task is the generalized fix for that class
- [../../System/kqapro_agent.md](../../System/kqapro_agent.md) — KQAPro tool catalog and classification this tool slots into

## Problem

The agent has no way to ask "what kind of thing is this term, in this KG's schema" before querying.
It guesses whether a question term denotes an **attribute**, a **relation**, or a **qualifier on a
statement**, queries the wrong namespace, gets `{}` back, and concludes the fact is absent — with no
signal that it queried the wrong namespace rather than a genuinely absent fact.

This single error pattern accounts for **5 of the 14** KQAPro questions that fail in 5-6 of 6 run
variants (idx 6, 68, 74, 86, 90 — see the analysis §1 for the full per-question table verified
against `db/datasets/kqapro/kb.nt`). Two cases are especially costly because they're silent:
`CountEntities.or_conditions` drops the wrong-namespace branch without complaint and still returns
`"trusted": true` (idx 86: gold 53 → answered 1; idx 90: gold 3 → answered 2).

**Root enabler:** a wrong-namespace query returns an empty result rather than "that predicate exists
in `prop:`, not `attr:`." Nothing in the current tool surface forces the model to revise its model of
the question.

## Goal

Give the agent a way to check a term's schema role *before* committing to a namespace, so it can
self-correct in one extra tool call instead of silently failing on an empty result.

## Design

`LocateTerm(entity_id: str, term: str) -> LocateTermResponse`

1. **Input:** an anchor entity (or entity type/concept, for schema-wide checks) and a natural-language
   or near-verbatim term from the question (e.g. "reviewer", "postal code", "located in", "main
   subject").
2. **Resolution:** fuzzy/embedding-match the term against the KG's actual predicate vocabulary across
   all three namespaces (`attr:`, `prop:`/relation, `qual:`) scoped to what's actually attached to
   the given entity (or entity type, if doing a schema-wide check) — reuse the existing
   attribute/relation embedding infrastructure (`get_embedding` cache, per
   [architecture-audit-2026-07-05.md](../../Decisions/architecture-audit-2026-07-05.md) C7/C6c) rather
   than building new embedding plumbing.
3. **Output:** the resolved role (`attribute` / `relation` / `qualifier`), the namespace-qualified
   predicate name, and — critically for the qualifier case — the **owning statement** (subject,
   predicate, object triple that the qualifier attaches to), so the agent can chain directly into
   `GetQualifierValue` or `QualifierFilter` without a second lookup.
4. **Zero/ambiguous-match behavior:** if the term matches nothing in any namespace for this entity,
   say so explicitly (`"no predicate matching 'X' found in any namespace for this entity"`) rather
   than returning an empty result indistinguishable from "the fact doesn't exist." This is the same
   self-diagnosis principle as recommendation #2
   ([self-diagnosing-empty-results.md](self-diagnosing-empty-results.md)) — the two tasks should be
   designed together since they share the "make emptiness informative" goal, even though this one is
   a new tool and #2 is a contract change to existing tools.

## Risks

- **Embedding-match precision.** Namespace disambiguation is only as good as the fuzzy match between
  the question's natural-language term and the KG's actual predicate labels. A wrong match could
  introduce a *new* failure mode (confidently wrong role) instead of the current honest failure
  (empty result). Needs a confidence threshold and a "multiple plausible roles" response shape, not
  a forced single answer.
- **Adds a tool-call round trip.** Per the analysis's turn-count finding (153.6:1 prompt-to-completion
  ratio), any new tool call has to earn its keep by *reducing* downstream wrong-namespace retries, not
  just adding overhead. Should net-reduce turns for the failure class it targets (one `LocateTerm`
  call replacing what is currently 2+ failed guesses), but this needs to be measured, not assumed.
- **Prompt/fewshot adoption.** Per §4.1/§4.2 of the analysis, new tools have a documented history of
  near-zero adoption when introduced only via docstring, especially if the classifier routes the
  question to a qtype whose strategy text doesn't mention the new tool. Needs explicit strategy-text
  and fewshot coverage for the qtypes most affected (`QueryAttr`, `QueryAttrQualifier`,
  `QueryRelationQualifier`, `Count` with `or_conditions`), coordinated with the fewshot-bank audit
  ([fewshot-bank-audit.md](fewshot-bank-audit.md)).

## Validation plan

- Smoke-test against the 5 identified failures (idx 6, 68, 74, 86, 90) directly — confirm `LocateTerm`
  correctly identifies the role/namespace for "reviewer", "postal code", "located in", "main subject",
  and the Barbara McLean qualifier-value case.
- Full n=100 KQAPro benchmark A/B (tool added vs. not) before/after, since this is exactly the kind of
  tool-visibility change that can shift agent behavior in ways a token-count comparison won't catch.
- Confirm no regression on the `CountEntities.or_conditions` cases already fixed by recommendation #5
  (this session's concurrent fix rejects unmatched branches instead of silently dropping them) —
  `LocateTerm` should make the *correct* namespace discoverable before that branch would ever need to
  reject anything.

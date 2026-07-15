# ADR: WikiKGQA Round-2 Repairs — ASK Repair, Projection Trim, Sanity-Guard Rescope, Conditional Conventions, Auto-Escalation

**Date:** 2026-07-15
**Commit:** `17a409f` "WikiKGQA: ASK repair, projection trim, conditional conventions, auto-escalation"
**Status:** Settled — implemented, 389 tests passing (+42)

## Related Docs
- [Decisions/wikikgqa-answer-sanity-guard-2026-07-15.md](./wikikgqa-answer-sanity-guard-2026-07-15.md) — same-day predecessor ADR: introduced `_result_is_sane` with the original any-`"/"` catch-all; Decision C here rescopes that catch-all
- [Decisions/wikikgqa-conventions-default-2026-07-03.md](./wikikgqa-conventions-default-2026-07-03.md) — the held-out A/B (0.7311 minimal vs 0.6833 full) that Decision D bounds the blast radius of, rather than reversing
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md) — non-empty-prior recovery and `run_manifest.json`; Decision E (auto-escalation) automates the manual escalation flow this ADR's era relied on by hand
- [Decisions/wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md](./wikikgqa-closure-expansion-and-now-pinning-2026-07-14.md) — commit-time mechanical rewrite pattern (strict-superset-gated escalation ladder) that Decisions A and B follow for ASK repair and projection trim
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference; updated alongside this ADR with the new commit-time pipeline stages
- `ama_kbqa/wikikgqa/ANSWER_CONVENTIONS.md` — rule text itself (rule 2 count-threshold sub-case, new rule 13, and the "2026-07-15 decision: conditional injection" section) was updated in commit `17a409f` directly; not duplicated here

---

## Context

This is the round-2 improvement package landed the same day as, and after,
the [answer-sanity-guard ADR](./wikikgqa-answer-sanity-guard-2026-07-15.md)'s
fix. The 2026-07-15 resubmission scored WM (with-mentions) 0.84 / WOM
(without-mentions) 0.79. Both tracks are being rerun with the full stack
(this commit included) before the challenge portal closes later today, so
each decision below is grounded in a training-gold check rather than a
scored resubmission — there wasn't time to A/B each change independently
against the leaderboard before the deadline.

## Decisions

### A. Yes/no ASK repair (`generator.py`)

**Chosen:** Gold-scan of `data/wikikgqa/wikikgqa.json` found all 35/35
yes/no-form questions (by surface form — `_YESNO_QUESTION_RE` matches
is/are/was/were/has/have/had/does/do/did/can/could/will/would as the leading
token) use an `ASK` gold query, zero exceptions. When a yes/no question
commits a non-boolean result, `_generate_once` re-runs generation **exactly
once** with an explicit instruction appended to the question
(`_ASK_REPAIR_INSTRUCTION`, including the count-threshold `ASK { { SELECT
(COUNT(...) AS ?cnt) {...} } FILTER(?cnt >= N) }` shape) and adopts the
re-run's result only if it actually is boolean; otherwise the original
non-boolean result is kept (a failed repair is not worse than the status
quo). Recursion is prevented by an internal `_repair_instruction` parameter:
the repair re-run passes it, so the repair re-run's own copy of the
triggering condition is false and can never trigger a second repair. Flag:
`GeneratedQuery` records the attempt; constructor gains
`AgentSparqlGenerator(ask_repair: bool = True)`.

**Evidence for the count-threshold shape specifically:** q403/q405 ("Has
France won the Eurovision at least twice?") committed a bare `COUNT` scalar
instead of the gold's wrapped-`ASK` shape and scored 0 — this is the case
that motivated strengthening rule 2's prompt text in Decision D alongside
the repair mechanism itself (belt-and-suspenders: prompt guidance to avoid
the mistake, plus a commit-time repair if it happens anyway).

**Rejected alternative:** always force `ASK` synthesis for yes/no questions
up front (skip generation of a non-`ASK` query entirely). Rejected — the
agent's tool-exploration trace is unaffected by the question's surface
form; forcing the output shape earlier would require plumbing the
constraint through the whole tool loop rather than a one-shot repair at the
existing commit boundary.

### B. Projection trim (`generator.py`)

**Chosen:** Gold-scan found 416/442 (94%) of gold `SELECT` queries project
exactly one variable. `to_codabench_answers` treats *every* projected
column as an answer value, so a committed multi-column `SELECT` is almost
always precision poison from spurious extra columns. `_trim_projection`
rewrites a multi-variable, non-aggregate, non-`SELECT *` `SELECT` query
(detected on string-literal-masked text, so a literal containing "SELECT"
or a brace can't confuse the match) to project only its first variable. The
caller executes the trimmed query and adopts it only if it still executes,
has rows, and is sane (`_result_is_sane`) — the first column's own value
set is unchanged by construction (same `WHERE` clause, same first
variable), so unlike closure expansion there is no superset check to make.
Guarded off for `ASK`/aggregate (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`/`GROUP
BY`/`HAVING`) queries, `SELECT *`, and `(...)`-expression projections (e.g.
`(?x + 1 AS ?y)`). Runs before the sanity guard and before class-closure
expansion in the commit-time chain (a trimmed result can drop junk that
only lived in a dropped column; closure expansion expects a single-column
query anyway). Flag: `AgentSparqlGenerator(projection_trim: bool = True)`.

**Companion prompt change:** `SYSTEM_PROMPT` gained an always-on rule
("SELECT exactly ONE variable... extra columns are scored as spurious
answers") — this is `ANSWER_CONVENTIONS.md` rule 13, unconditional (unlike
the R8/R9/R12 rules in Decision D). The mechanical trim is the safety net
for when the prompt rule is followed imperfectly.

### C. Sanity catch-all rescoped to `wikidata.org/` residue

**Chosen:** `_result_is_sane`'s catch-all (introduced same-day in the
[sanity-guard ADR](./wikikgqa-answer-sanity-guard-2026-07-15.md) as "any
`"/"` survives normalization → unsane") is narrowed to specifically
`"wikidata.org/"` (case-insensitive) surviving normalization. The gold-scan
justification is unchanged (no gold answer in the 497-question set contains
`"wikidata.org/"`, so the catch-all loses no true-positive coverage against
training gold) — what changes is the **false-positive** exposure: the
original any-`"/"` rule would flag a legitimate URL-literal gold answer
(e.g. an official-website value like `"https://www.louvre.fr/"`) as unsane,
which happens not to occur in this benchmark's gold set but would
misfire the moment the guard (or its underlying pattern) is reused in a
more general setting.

**Why this matters despite zero benchmark impact today:** this was a
user-driven correction, not a training-gold-triggered one — the original
any-`"/"` rule was calibrated tightly to this specific benchmark's absence
of URL-literal answers, and the user flagged it as overfit to that
coincidence rather than a property of "sane answer" in general. Rescoping
to the Wikidata-URI-junk pattern specifically (blank node /
`entity/statement/` / `Special:EntityData` / any other `wikidata.org/`
residue) keeps the same detection power for the actual failure mode
(unmapped Wikidata URIs) without over-claiming that *any* slash is
suspicious.

**Rejected alternative:** leave the any-`"/"` catch-all as-is, since it
scores identically on the current gold set. Rejected on user instruction —
correctness of the underlying rule (not just its measured benchmark score)
was the deciding factor.

### D. Conditional convention injection (`wikidata_agent/prompts.py`, `agent.py`)

**Chosen:** The 2026-07-03 held-out A/B ([ADR](./wikikgqa-conventions-default-2026-07-03.md))
found injecting the *entire* extended rule bundle (rules 7–10 and 12 of
`ANSWER_CONVENTIONS.md`) unconditionally on every question **hurt** overall
score (0.7311 with only R1–R6 vs 0.6833 with the full bundle always on).
Root cause: each extended rule's supporting evidence is a small, specific
slice of the training set, and always-on injection adds that rule's text
as noise to the large majority of questions it doesn't apply to (rule 9
was observed actively over-applied on q81, dropping a valid answer).

Rather than leave R7–R10 off entirely, `conventions="minimal"` (the
default) now injects three of those rules **conditionally**, per-question,
only when the question text matches a narrow trigger regex:

| Rule | Trigger regex | Training hits |
|---|---|---|
| 8 (quantity normalization) | `\b(heavy\|mass\|weigh\|weight\|tall\|height\|net worth)\b` | 5 |
| 9 (currently/exclude-ended) | `\b(currently\|still\|nowadays\|present-day)\b` | 4 |
| 12 (class-vs-instance location) | `\b(found in\|are there in\|which countries have)\b` | 15 |

`get_conditional_conventions(question)` (`prompts.py`) returns the matched
rules' text (empty string if none match — the common case); it's called
from `WikidataAgent._build_analysis_context` (the per-question hook, unlike
`_get_system_prompt` which runs once at `__init__` before any question is
known) and appended only when `conventions == "minimal"` — `"full"` already
gets the whole `EXTENDED_CONVENTIONS` bundle unconditionally via the system
prompt, so conditional injection would just duplicate it there.

This bounds the blast radius that made the full-bundle A/B lose: each rule
now only reaches the small number of questions its trigger fires on,
instead of every question. "Minimal" conventions now means: R1–R6 always
on, rule 2 (`ASK`) strengthened with the count-threshold sub-case
(Decision A's prompt half), a new always-on single-variable-projection rule
(Decision B's rule 13), plus rules 8/9/12 injected conditionally per the
table above. Rules 7, 10, 11 are **not** conditionally injected — they have
no clean textual trigger (e.g. "do not over-constrain" applies to reasoning
generally, not a detectable keyword) — and remain `conventions="full"`
opt-in only.

**Rejected alternative:** flip the 2026-07-03 default back to `"full"` now
that specific rules have narrow, checkable value. Rejected — the A/B
evidence that always-on injection nets negative still holds; conditional
injection is a strictly more targeted mechanism than the binary
full/minimal toggle, not a reversal of the earlier finding.

### E. Auto-escalation (`benchmark.py`)

**Chosen:** Every gold answer in this benchmark is non-empty, so a
submission answer that ends up empty after the main pass is known-wrong.
`_escalate_empty_answers` (new function, called from `run_benchmark` after
the main pass, before the final submission write) re-runs each
still-empty-answer question exactly once with the generator's
`tool_budget` raised by a fixed `_ESCALATION_BUDGET_BUMP = 15` (fixed
rather than a CLI knob — the prior manual escalation used +15 by hand and
it was sufficient; kept simple until evidence says otherwise), then
restores the original budget. **Replace-only-empty policy**, mirroring
`scripts/assemble_voted_submission.py`'s existing `--patch` flow: a
question with a non-empty answer is never re-run; an escalation re-run
that comes back still-empty or errored never overwrites the original
outcome; only a still-empty→non-empty transition patches `outcomes` and
`score_pairs` in place. No-op for non-`AgentSparqlGenerator` generators or
when there are zero empty answers after the main pass — never touches the
generator otherwise. `summary.json` gains `escalated_qids` /
`escalated_filled_qids` / `escalated_still_empty_qids`, additive-only (the
common zero-empties case leaves `summary.json`'s shape unchanged). CLI:
`benchmark.py --no-auto-escalate` disables (default: on).

This automates a manual flow already used once by hand (per the
`_ESCALATION_BUDGET_BUMP` code comment: the EN-without-mentions run with
one manually-escalated empty answer, q3, escalated by hand with +15 and
found sufficient) rather than requiring a human to notice and re-run empty
outcomes after every benchmark pass.

**Rejected alternative:** make the budget bump a CLI-configurable
parameter from the start. Rejected for now — one prior manual data point
(+15 sufficient for q3) is not enough evidence to justify a tunable
surface; revisit if escalation runs start needing a different bump.

## Validation

Suite: `uv run pytest` → **389 passed** (up from 347 as of the
[same-day sanity-guard ADR](./wikikgqa-answer-sanity-guard-2026-07-15.md)),
+42 tests across `tests/wikikgqa/test_generator.py` (ASK repair, projection
trim, rescoped sanity catch-all), `tests/wikikgqa/test_benchmark.py`
(auto-escalation), and the new `tests/wikikgqa/test_conventions_injection.py`
(conditional convention injection).

No dedicated held-out A/B was run for this package before the portal
closes — each decision's grounding is a training-gold count or a single
observed failure case (q403/q405, q113-adjacent, q81, q3), not a scored
A/B. Treat these as targeted, evidence-motivated fixes for specific
observed failure modes rather than independently-validated net-positive
changes; if the full-stack rerun (in progress as of this ADR) surfaces a
regression traceable to one of these five, revisit that decision
specifically rather than the package as a whole.

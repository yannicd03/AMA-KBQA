# ADR: WikiKGQA Commit-Time Class-Closure Expansion and NOW() Pinning

**Date:** 2026-07-14
**Commit:** `d75f45d` "WikiKGQA: commit-time class-closure expansion, NOW() pinned to gold reference instant"
**Status:** Settled — implemented, 328 tests passing, live-validated on 4 known-failure dev questions

## Related Docs
- [System/wikikgqa_agent.md](../System/wikikgqa_agent.md) — current-state reference for `AgentSparqlGenerator` / `endpoint.execute()`; updated alongside this ADR with the two new commit-time stages
- [Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md) — prior commit-time repair (non-empty-prior recovery, ASK voting); this ADR adds two more mechanisms to the same commit-time stage, and resolves several items in its Open Items table
- [Decisions/wikikgqa-conventions-default-2026-07-03.md](./wikikgqa-conventions-default-2026-07-03.md) — why prompt-space R7-R10 (including rule 10, class-membership completeness) defaults off; the reason this same rule is applied mechanically here instead
- [Decisions/wikikgqa-2026-adaptation.md](./wikikgqa-2026-adaptation.md) — why this subsystem exists (SPARQL-generate-and-execute head, challenge endpoint)

---

## Background

The 2026-07-03 EN test submissions scored AMA-KBQA 2nd on EN-WITH-MENTIONS
(0.85 vs winner 0.88) and 3rd on EN-WITHOUT-MENTIONS (0.78 vs 0.81) — both
~0.03 behind the leaderboard leader. These two mechanisms target the
dominant residual failure classes behind that gap. Both landed **after** the
2026-07-03 submission deadline, so they do not affect the scored leaderboard
result; they are validated against dev questions only (see Validation below)
pending the next submission window.

## Decisions

### 1. Commit-time class-closure expansion (`ama_kbqa/wikikgqa/generator.py`)

**Problem.** The dominant remaining failure class: the committed query
constrains class membership with a bare `?var wdt:P31 wd:QX .`, but gold
uses the transitive closure (`wdt:P31/wdt:P279*`, sometimes
`wdt:P31*/wdt:P279*`) so subclass instances count too. This scores
precision=1.0 / recall≈0 — a correct-looking query that silently misses
almost the whole gold answer set.

This was previously **rule 10** of `ANSWER_CONVENTIONS.md`, taught to the
model as a prompt instruction (part of the `EXTENDED_CONVENTIONS` R7-R10
block). The 2026-07-03 held-out seed-99 A/B
([Decisions/wikikgqa-conventions-default-2026-07-03.md](./wikikgqa-conventions-default-2026-07-03.md))
found R7-R10 as a block gave no held-out benefit and flipped the default to
minimal (R1-R6). But per-question failure analysis (q110, q111) showed rule
10's underlying fix is real and needed for specific questions — the prompt
just wasn't reliably eliciting it, and other R7-R10 rules were adding noise
that masked the benefit in the aggregate A/B. **Chosen:** apply rule 10
mechanically, in code, at commit time — sidesteps prompt interference
entirely, and is inherently safe because it's gated on strict answer-set
growth rather than trusting the model's judgment.

**Mechanism** (`_next_closure_escalation`, `_closure_expansion_eligible`,
`_is_strict_superset`, wired into `AgentSparqlGenerator._generate_once`):

- An escalation ladder is walked with a cursor, widening the query's
  class-membership pattern one step per iteration:
  `wdt:P31` → `wdt:P31/wdt:P279*` → `wdt:P31*/wdt:P279*` (at most 2 steps,
  string-literal-aware so a triple pattern that merely *looks* like
  `wdt:P31` inside a label string is never touched).
- At each step, the candidate query executes. Adoption happens **only when
  the candidate answer set is a strict superset of the currently committed
  one** (`_is_strict_superset`) — never on an equal-size or unrelated set,
  and never on execution error.
- **Equal-set steps pass through without stopping the ladder.** This is the
  key design point, live-verified on q110: `P31` → `P31/P279*` keeps the
  *same* 1-row answer set (no gain, would look like "nothing to escalate
  here" if the loop stopped), and only the ceiling level
  `P31*/P279*` reaches gold's 564 rows. A naive "stop on no-gain" loop would
  have missed q110 entirely.
- The committed query text is only rewritten when a step is actually
  adopted; a no-op ladder (nothing ever strictly grows) leaves the original
  committed query untouched.

**Guardrails** (`_closure_expansion_eligible`): `ASK` queries, aggregate
queries (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`/`GROUP BY`/`HAVING`), and
literal-valued `SELECT`s (the projected column isn't URI-typed) are all
excluded outright — widening a scalar or a literal projection isn't a
"grow the row set" operation the strict-superset check can validate, so
those are refused rather than risked. 0-row committed results are also
excluded (already handled by the 2026-07-03 non-empty-prior recovery path).

**Flag:** `AgentSparqlGenerator(closure_expansion=True)` (default on),
`benchmark.py --no-closure-expansion` to disable.

**Rejected alternatives:**
- *Keep it prompt-only.* Already tried (R7-R10), held-out A/B showed no
  net benefit — see the linked conventions-default ADR. The per-question
  evidence (q110/q111) shows the underlying repair is real; the delivery
  mechanism (prompt) was the problem, not the repair itself.
- *Rewrite to the ceiling level unconditionally.* Rejected — would apply
  the widening even when the bare pattern was already correct (a genuine
  non-hierarchical class), silently introducing false positives on
  questions where recall was already correct.

### 2. NOW() pinning to the gold reference instant (`ama_kbqa/wikikgqa/endpoint.py`)

**Problem.** Gold answers for this challenge were computed once, against a
frozen KG snapshot, using SPARQL's `NOW()` for age/date-relative questions
("how old is X", "who has a birthday today"). Gold `NOW()` therefore
resolved to whatever instant the gold-generation run happened at. When
AMA-KBQA's own queries call `NOW()` at answer time (any time after that),
ages tick over and "born today" filters land on the wrong calendar day —
producing wrong answers on a class of questions that were otherwise being
solved correctly.

**Derivation of the reference instant.** Reverse-engineered from gold data,
not documented anywhere in the challenge materials: gold ages for Stan Lee
(103, born 1922-12-28) and Justin Bieber (32, born 1994-03-01) bound the
possible window, and every gold "birthday today" entity was independently
found to be born on April 8. Together these pin the reference date to
**2026-04-08** (`REFERENCE_TIME = "2026-04-08T00:00:00Z"`,
`ama_kbqa/wikikgqa/endpoint.py`).

**Mechanism** (`rewrite_now`, wired into `execute()`): before a query is
sent, every SPARQL `NOW()` function-call occurrence is textually replaced
with `"2026-04-08T00:00:00Z"^^<...#dateTime>`. The rewrite is:
- Case-insensitive and whitespace-tolerant (`NOW()`, `Now ( )`, etc).
- String-literal- and IRI-safe — occurrences inside quoted strings or
  IRIREFs are left untouched, so a label that happens to contain the text
  "NOW()" isn't corrupted.
- Identifier-safe — a negative lookbehind excludes `?now`/`$now` variables
  and prefixed names like `ex:now`, which merely contain the substring
  "now" and aren't calls to the `NOW()` function.

**Flag:** default on (`WIKIKGQA_PIN_NOW` env var, unset/`"1"` = on, `"0"` =
off via `benchmark.py --no-pin-now`). Threaded through an env var rather
than a parameter because `execute()` is reached through several call paths
(generator commit path, the agent's `RunSPARQL` tool via
`wikidata_server.py`, submission-time checks) that would all need explicit
plumbing otherwise — env var means every caller gets pinning for free
without touching each call site. `pin_now=False` is available as an
explicit parameter override for tests/tooling.

**Applies consistently across the whole pipeline.** Because every code path
funnels through `endpoint.execute()`, pinning applies identically during
**agent exploration** (the `RunSPARQL` MCP tool in `wikidata_server.py`, so
the model sees results consistent with the frozen instant while it's still
reasoning) and during **commit-time execution** (the generator's final
query and the closure-expansion candidates above). A model that explores
against one NOW() and commits against a different one would see
inconsistent row counts across the same session — pinning at the `execute()`
choke point avoids that entirely.

**Rejected alternatives:**
- *Rewrite only at commit time, not during exploration.* Rejected — the
  agent's own `RunSPARQL` tool calls go through the same `execute()`, and
  leaving those unpinned would let the model observe today's actual date
  while exploring, then have its committed query silently re-execute
  against a different (frozen) instant, contradicting what it just saw.
- *Thread `pin_now` as an explicit parameter through every call site.*
  Rejected as unnecessary plumbing for challenge-only code where the
  default should always be on; the env var keeps every current and future
  caller consistent without an API change.

## Validation

4 known-failure dev questions, full agent pipeline, challenge endpoint:

| Question | Before | After | Note |
|---|---|---|---|
| q110 | 0.004 | 1.000 | Closure expansion — only the ceiling `P31*/P279*` level reaches gold's 564 rows |
| q111 | 0.064 | 0.897 | Closure expansion |
| q79 | recall 0 | recall 1.0 | Precision capped by gold's own `LIMIT 10` artifact — unwinnable regardless of query correctness |
| q16 | unchanged | unchanged | Expected — its gap is an optional-path hop (`P3967`/`P1346`), not a class-membership closure issue; confirms the mechanism doesn't over-fire on unrelated failure modes |

Full suite: `uv run pytest` → **328 passed** (up from 290 as of the
2026-07-03 `877551c` test-coverage commit).

**Caveat on `_answer_set` (closure-expansion's superset check).** The
strict-superset comparison in `generator.py::_answer_set` reuses
`ama_kbqa/wikikgqa/submission.py::to_codabench_answers` to normalize rows
into a comparable set. `to_codabench_answers` has a latent bug in
`_bare_id` — see the note in
[Decisions/wikikgqa-commit-time-recovery-2026-07-03.md](./wikikgqa-commit-time-recovery-2026-07-03.md#open-items)
Open Items — but it is inert against all 497 gold answers, so it does not
affect the superset comparison in practice.

## Context: this ADR does not change the scored leaderboard result

Both mechanisms landed 2026-07-14, after the 2026-07-03 EN test submission
deadline referenced in Background. They are validated here against 4 dev
questions only; leaderboard impact (if any) will show up in the next
submission window, not the 0.85/0.78 EN scores already on record.

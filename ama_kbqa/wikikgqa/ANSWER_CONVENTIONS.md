# WikiKGQA gold-answer conventions

QALD-F1 scores the **exact answer set** against a specific gold query, so a query
that is "correct" but follows a different convention than the gold scores 0. These
are the answer-shape conventions we inferred from the v2026.2 training gold, with
the evidence and our confidence. They are encoded in the Wikidata agent's system prompt
and, for the conditionally-injected rules, its per-question analysis context
(`ama_kbqa/agents/wikidata_agent/prompts.py`, `ama_kbqa/agents/wikidata_agent/agent.py`).

This is **challenge-only** guidance. It reflects the benchmark's (sometimes
arbitrary or inconsistent) choices, not a universal "right" way to answer.

## Rules

| # | Rule | Evidence | Confidence |
|---|------|----------|------------|
| 1 | **Return entity URIs, not labels.** Never `SELECT ?xLabel`; never `SERVICE wikibase:label`. | Gold answers are `wd:Q...` URIs; QLever has no label service. Caused the v1 easy-bucket regression. | High (consistent) |
| 2 | **Yes/no questions → `ASK`.** Use `ASK { FILTER NOT EXISTS {...} }` for negatives ("is X alive" = no death date). Sub-case: "at least/more than N times" (count-threshold yes/no) → wrap a `COUNT` subquery and `FILTER(?cnt >= N)` inside the `ASK`, not a bare `COUNT` scalar: `ASK { { SELECT (COUNT(DISTINCT ?x) AS ?cnt) {...} } FILTER(?cnt >= N) }`. | q5, q6, q12, q22, q23 gold are all `ASK`. q403/q405 gold use the count-threshold `ASK` shape exactly; the agent had emitted a bare `COUNT` scalar and scored 0. | High |
| 3 | **"Which/what" list questions → `SELECT DISTINCT` of the entity ids.** | q15–17, q20, q24 gold. | High |
| 4 | **Superlatives → return ALL entities tied for the extreme via a `MAX`/`MIN` subquery + `FILTER(?v = ?m)`**, not `ORDER BY DESC(?v) LIMIT 1` (which drops ties). For "most/fewest", the value is a `COUNT` computed per-subject in an inner `GROUP BY` subquery. | q1, q2 (single extreme) match either way, but q327 ("Kubrick movie that won the most Oscars") gold returns **2** tied movies via a MAX-subquery; `LIMIT 1` scored R=0.5. | High (ties are common) |
| 5 | **Age questions → current age via `NOW()` from `wdt:P569`, even for the dead**, with month/day correction. | q9, q10 gold use `YEAR(NOW())-YEAR(?dob) - IF(MONTH...)`. q9 ("what age IS Stan Lee", who is dead) computes as-if-alive. | Medium (n=2; arguably counterintuitive) |
| 6 | **"How many" → default to a `COUNT` scalar.** Corrected from an earlier "prefer list" rule after checking the full corpus. | Of 29 "how many" questions, **21 gold answers use `COUNT`, only 8 return a list** (and 42/497 gold queries use COUNT overall). The earlier list-default came from one example (q21) and was wrong. | Medium — 72% majority, not unanimous (8 list exceptions remain) |
| 7 | **"Country/countries" = `Q3624078` (sovereign state)**, with `wdt:P31/wdt:P279*`, NOT `Q6256` (country). | q47, q81, q125 gold all use Q3624078; Q6256 over-includes by a few. | High |
| 8 | **Measurement/quantity VALUES via the normalized statement path** `p:Pxxx/psn:Pxxx/wikibase:quantityAmount`, not `wdt:Pxxx`. Use for the answer value and for `ORDER BY` in superlatives; fall back to `wdt:` if it yields nothing. **Conditionally injected** (trigger: `\b(heavy\|mass\|weigh\|weight\|tall\|height\|net worth)\b`, case-insensitive) as of 2026-07-15 — see decision note below. | q229 (mass P2067), q302 (mass), q172 (net worth P2218) gold all use the normalized path; the `wdt:` value mismatches. Training-set trigger coverage: 5 questions. | High (verified: baseball mass `0.146` via normalized path) |
| 9 | **"currently / present-day / still / now" → exclude ended entities** with `MINUS { ?x wdt:P582 ?e }` (end time) and/or `wdt:P576` (dissolved) — **ONLY when the question actually says so**. Do not add it otherwise. **Conditionally injected** (trigger: `\b(currently\|still\|nowadays\|present-day)\b`, case-insensitive) as of 2026-07-15 — see decision note below. | q47 "currently have multiple capitals" needs it; q81 "which countries have a female head of state" does NOT say "currently", and adding the exclusion dropped a valid answer (R=0.95). Training-set trigger coverage: 4 questions. | Medium |
| 10 | **Instance-of completeness**: default `wdt:P31/wdt:P279*`; broaden to `wdt:P31*/wdt:P279*` if incomplete; for organisms/taxa reach members via `(wdt:P31/wdt:P279*)\|(wdt:P171+)`. | q110/q111 gold use `P31*/P279*`; q49 "named after plants" needs the `P171+` taxon path. | Medium |
| 11 | **Do not over-constrain.** Encode only the constraints the question states; do not add an origin/nationality/country/type filter it does not ask for — each extra clause can drop valid gold answers. | q490 "Italian sauces named after Italian cities" gold does NOT require `wdt:P495` (origin=Italy); adding it scored R=0.67. Same failure mode as rule 9 over-application (q81). | High |
| 12 | **Class-vs-instance for "where is X found / which countries have X"**: X is a class, so bind its INSTANCES (`?s wdt:P31/wdt:P279* X`) and reach the place via the location hierarchy (`?s wdt:P131*/wdt:P17 ?country`) — do not read `wdt:P17` off the class node itself. **Conditionally injected** (trigger: `\b(found in\|are there in\|which countries have)\b`, case-insensitive) as of 2026-07-15 — see decision note below. | q301 "in which countries are tepuis found": gold binds instances of tepui (Q828329) + `P131*/P17`; querying `wd:Q828329 wdt:P17` returned the wrong 1 answer. Training-set trigger coverage: 15 questions. | Medium |
| 13 | **Single-variable projection.** `SELECT` exactly ONE variable — the answer variable. Never project helper variables or labels alongside it (e.g. `SELECT ?x ?xLabel` or `SELECT ?x ?cnt`), even when the first variable alone is correct. Always-on. | 416/442 gold `SELECT` queries project a single variable; extra columns are scored as spurious answers. | High |

The normalized-quantity rule (8) requires the `psv:`/`psn:` prefixes, which are now injected by `endpoint.py` (QLever does not predefine them).

### 2026-07-15 decision: conditional injection for rules 8, 9, 12

The 2026-07-03 held-out A/B (seed-99 rand-50, Qwen3.5-397B) found that injecting the
*entire* extended rule bundle (R7-R10, i.e. rules 7-10 and 12 above) unconditionally on
every question **hurt** overall score (0.7311 with only R1-R6 vs 0.6833 with the full
bundle always on). Root cause: each extended rule's evidence is a small, specific slice
of the training set (rule 8: 3 examples; rule 9: 2 examples, one of which is a *negative*
example warning against over-applying it; rule 12: 1 example) — always-on injection adds
that rule's text as noise to the ~90%+ of questions it doesn't apply to, and the model
sometimes over-applies a rule to a question it wasn't meant for (exactly what happened to
rule 9 on q81).

Rather than keep these rules off entirely, `conventions="minimal"` (the default) now
injects rules 8, 9, and 12 **conditionally**: per-question, only when the question text
matches a narrow trigger regex for that rule (see `get_conditional_conventions` in
`ama_kbqa/agents/wikidata_agent/prompts.py`, wired into `_build_analysis_context` in
`agent.py`). This bounds the blast radius that made the full-bundle A/B lose — each rule
only reaches the small number of questions its trigger fires on (5 / 4 / 15 questions in
the training set respectively) — while still recovering the rule's benefit on exactly the
questions gold evidence shows need it. Rules 7, 10, 11 are NOT conditionally injected —
they have no clean textual trigger (e.g. "do not over-constrain" applies to reasoning
generally, not a detectable keyword) — so they remain part of the `conventions="full"`
opt-in only, same as before this change. `conventions="full"` is unchanged: it still
injects the entire `EXTENDED_CONVENTIONS` bundle unconditionally via the system prompt,
for comparison/opt-in use.

### The exact age formula (rule 5)
```sparql
SELECT (
  (YEAR(NOW())-YEAR(?dob))
  - IF(MONTH(NOW())>MONTH(?dob), 0,
       IF(MONTH(NOW())=MONTH(?dob) && DAY(NOW())>=DAY(?dob), 0, 1))
  AS ?age
) WHERE { wd:Q... wdt:P569 ?dob }
```

## Known unwinnable cases
Some gold choices cannot be reliably matched by reasoning alone:
- **Rule 6 inconsistency**: no query can match both q14 (count) and q21 (list) under one policy.
- **Modeling choices**: q24 "EU countries" gold uses `P527` (has part); `P463` (member of)
  gives a near-identical but not identical set (27/28). Either is defensible.

These are benchmark artifacts; we accept a small unavoidable F1 loss on them.

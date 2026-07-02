# WikiKGQA gold-answer conventions

QALD-F1 scores the **exact answer set** against a specific gold query, so a query
that is "correct" but follows a different convention than the gold scores 0. These
are the answer-shape conventions we inferred from the v2026.2 training gold, with
the evidence and our confidence. They are encoded in the Wikidata agent's system
prompt (`ama_kbqa/agents/wikidata_agent/prompts.py`).

This is **challenge-only** guidance. It reflects the benchmark's (sometimes
arbitrary or inconsistent) choices, not a universal "right" way to answer.

## Rules

| # | Rule | Evidence | Confidence |
|---|------|----------|------------|
| 1 | **Return entity URIs, not labels.** Never `SELECT ?xLabel`; never `SERVICE wikibase:label`. | Gold answers are `wd:Q...` URIs; QLever has no label service. Caused the v1 easy-bucket regression. | High (consistent) |
| 2 | **Yes/no questions → `ASK`.** Use `ASK { FILTER NOT EXISTS {...} }` for negatives ("is X alive" = no death date). | q5, q6, q12, q22, q23 gold are all `ASK`. | High |
| 3 | **"Which/what" list questions → `SELECT DISTINCT` of the entity ids.** | q15–17, q20, q24 gold. | High |
| 4 | **Superlatives → return ALL entities tied for the extreme via a `MAX`/`MIN` subquery + `FILTER(?v = ?m)`**, not `ORDER BY DESC(?v) LIMIT 1` (which drops ties). For "most/fewest", the value is a `COUNT` computed per-subject in an inner `GROUP BY` subquery. | q1, q2 (single extreme) match either way, but q327 ("Kubrick movie that won the most Oscars") gold returns **2** tied movies via a MAX-subquery; `LIMIT 1` scored R=0.5. | High (ties are common) |
| 5 | **Age questions → current age via `NOW()` from `wdt:P569`, even for the dead**, with month/day correction. | q9, q10 gold use `YEAR(NOW())-YEAR(?dob) - IF(MONTH...)`. q9 ("what age IS Stan Lee", who is dead) computes as-if-alive. | Medium (n=2; arguably counterintuitive) |
| 6 | **"How many" → default to a `COUNT` scalar.** Corrected from an earlier "prefer list" rule after checking the full corpus. | Of 29 "how many" questions, **21 gold answers use `COUNT`, only 8 return a list** (and 42/497 gold queries use COUNT overall). The earlier list-default came from one example (q21) and was wrong. | Medium — 72% majority, not unanimous (8 list exceptions remain) |
| 7 | **"Country/countries" = `Q3624078` (sovereign state)**, with `wdt:P31/wdt:P279*`, NOT `Q6256` (country). | q47, q81, q125 gold all use Q3624078; Q6256 over-includes by a few. | High |
| 8 | **Measurement/quantity VALUES via the normalized statement path** `p:Pxxx/psn:Pxxx/wikibase:quantityAmount`, not `wdt:Pxxx`. Use for the answer value and for `ORDER BY` in superlatives; fall back to `wdt:` if it yields nothing. | q229 (mass P2067), q302 (mass), q172 (net worth P2218) gold all use the normalized path; the `wdt:` value mismatches. | High (verified: baseball mass `0.146` via normalized path) |
| 9 | **"currently / present-day / still / now" → exclude ended entities** with `MINUS { ?x wdt:P582 ?e }` (end time) and/or `wdt:P576` (dissolved) — **ONLY when the question actually says so**. Do not add it otherwise. | q47 "currently have multiple capitals" needs it; q81 "which countries have a female head of state" does NOT say "currently", and adding the exclusion dropped a valid answer (R=0.95). | Medium |
| 10 | **Instance-of completeness**: default `wdt:P31/wdt:P279*`; broaden to `wdt:P31*/wdt:P279*` if incomplete; for organisms/taxa reach members via `(wdt:P31/wdt:P279*)\|(wdt:P171+)`. | q110/q111 gold use `P31*/P279*`; q49 "named after plants" needs the `P171+` taxon path. | Medium |
| 11 | **Do not over-constrain.** Encode only the constraints the question states; do not add an origin/nationality/country/type filter it does not ask for — each extra clause can drop valid gold answers. | q490 "Italian sauces named after Italian cities" gold does NOT require `wdt:P495` (origin=Italy); adding it scored R=0.67. Same failure mode as rule 9 over-application (q81). | High |
| 12 | **Class-vs-instance for "where is X found / which countries have X"**: X is a class, so bind its INSTANCES (`?s wdt:P31/wdt:P279* X`) and reach the place via the location hierarchy (`?s wdt:P131*/wdt:P17 ?country`) — do not read `wdt:P17` off the class node itself. | q301 "in which countries are tepuis found": gold binds instances of tepui (Q828329) + `P131*/P17`; querying `wd:Q828329 wdt:P17` returned the wrong 1 answer. | Medium |

The normalized-quantity rule (8) requires the `psv:`/`psn:` prefixes, which are now injected by `endpoint.py` (QLever does not predefine them).

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

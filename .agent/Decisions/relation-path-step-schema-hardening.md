# ADR: RelationPathStep Schema Hardening (Named-Key Validation for relation_path)

**Date:** 2026-07-25
**Commit:** `195ce31` (pushed to `origin/dev`, **not yet deployed** — see Deployment status below)

## Related Docs
- [Abstract Operation Contract](./abstract-operation-contract.md) — the wider ADR this one narrows: Decision 4 there notes KG-flavored tool names are preserved and mechanically validated, but did not cover per-parameter shape validation within a tool call; this ADR is the parameter-level companion for the one op (`follow_path`) both KGs bind to these tools
- [Project Architecture](../System/project_architecture.md) — `FindEntitiesByRelationPath` (KQAPro) and `FollowRelationPath` (SciQA) tool descriptions

---

## Context

Both MCP servers typed the `relation_path` parameter of a multi-hop tool as a plain `list[Dict[str, str]]`. Pydantic validated only that each list item was a dict — not that it contained the required relation-name key. A malformed step passed validation and reached the query builder, where subscripting (`step["relation"]` / `step["predicate"]`) raised a raw `KeyError`.

Discovered live during the seed-42 benchmark, from actual log evidence:
1. Agent called `FindEntitiesByRelationPath` with `relation_path: ["country_of_origin"]` (list of strings).
2. Pydantic returned only `Input should be a valid dictionary [type=dict_type]` — never naming the expected keys.
3. Agent reasoned correctly ("needs to be a list of dicts, not strings") and retried with `[{"predicate": "country_of_origin"}]`.
4. That is a dict, so it validated, then `step["relation"]` raised `KeyError: 'relation'`.

Measured cost: tool-validation errors on ~16% of questions with minimax-m2.7 (0.160/question vs 0.000/question for gemma-4-31b) — roughly one wasted agent turn per six questions. It fired twice in a single 500-question run. Non-fatal: the wrapper at `kqapro_server.py:526` catches it, so it cost tokens and latency, not failed questions.

`sciqa_server.py`'s `FollowRelationPath` had the identical `step["predicate"]` bug, found by sweeping for the bug class rather than from an independent incident report.

---

## Decision

Replace the untyped `list[Dict[str, str]]` with a `RelationPathStep` Pydantic model in both servers, and change the tool signature to `list[RelationPathStep]`.

**`ama_kbqa/server/kqapro_server.py`** (~lines 268-324):
- Module-level constant `_RELATION_STEP_KEY_ALIASES = ("relation", "predicate", "relation_name", "property")` — the single source of truth for accepted keys, so the `AliasChoices` binding and the validator's error-message text cannot drift apart.
- `relation: str` field, required, bound via `validation_alias=AliasChoices(*_RELATION_STEP_KEY_ALIASES)`.
- `direction: Literal["forward", "backward"]`, default `"forward"`.
- `@model_validator(mode="before")` rejects non-dict items and dicts missing every alias key, with a message that names the accepted keys.

**`ama_kbqa/server/sciqa_server.py`** (~lines 289-342): the same model mirrored for `FollowRelationPath`. Primary field name is `predicate` (not `relation`) to match that server's existing vocabulary; alias tuple order is `("predicate", "relation", "relation_name", "property")`.

Both servers project the validated model back to a plain-dict shape (`relation_path_dicts`) before building response and journal payloads, so **the JSON returned to the agent is unchanged** — this is what keeps the existing test suite green without touching downstream consumers.

### Rejected alternatives
- *Catch `KeyError` at the call site and return a generic error string*: would have papered over the message-quality defect without fixing it — the model still wouldn't learn the accepted key names, just get a different-shaped generic error one layer later.
- *Reject unrecognized keys outright (single canonical key name only)*: rejected because the model's `predicate` guess in the observed trace was reasonable; forcing one exact spelling would keep costing a wasted turn on every plausible-but-wrong guess instead of just the first one.

---

## Consequences worth flagging

1. **Behavior change, not just error-message polish:** an unrecognized `direction` (e.g. `"reverse"`) previously fell through to forward traversal *silently* and returned confidently wrong results. It now raises an explicit validation error. This is a new failure path that did not exist before — deliberate, on the grounds that silent-wrong is worse than loud-wrong for a benchmarked system.
2. Field **descriptions are load-bearing, not decorative**: they are what the LLM reads when deciding how to call the tool. The original defect was fundamentally an error-message problem — the model had no way to learn the expected key names from a bare `dict_type` validation error.
3. Accepting aliases (`relation`/`predicate`/`relation_name`/`property`) is a deliberate design stance, not laziness: the model's own alternate-key guess was reasonable, so making it succeed removes the wasted turn entirely rather than just erroring more politely.
4. `QueryComparisonRows`'s `filters` param in `sciqa_server.py` (~line 3397) was examined for the same bug class and deliberately **left unchanged** — it uses `.get()` throughout, so no `KeyError` is possible there, only a less-helpful generic error. Not a gap; a scoped decision not to touch a param that can't crash.

---

## Verification

- `uv run pytest tests/server/test_relation_path_step_schema.py` → 15 passed (306-line new test file).
- Full suite `uv run pytest -q` → 494 passed.
- Both runs executed and confirmed directly (not subagent-reported).

---

## Deployment status

Committed and pushed to `origin/dev` (`MaxKlat29/AMAKBQA`) as `195ce31`, but **deliberately not deployed**. The Hetzner host is mid-run on a ~25h benchmark at commit `921a04d` and must not be updated until that run finishes. A future session should not assume the running Hetzner system has this fix — check the deployed commit before attributing behavior to it.

"""SPARQL generators for WikiKGQA.

A generator turns a :class:`WikiKGQAQuestion` into a SPARQL query (and, since it
executes during self-correction, its result too). The with-mentions baseline
here is deliberately lean: it does NOT run the full agentic tool-loop, because
the resolved QIDs/PIDs remove the need for exploration. It prompts the chat
model to assemble a query, executes it against the endpoint, and repairs once or
twice on execution errors or empty results.

The agentic generator (re-pointing the KQAPro exploration tools at live SPARQL)
is a later, heavier alternative that plugs into the same `Generator` protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ama_kbqa.wikikgqa.dataset import WikiKGQAQuestion
from ama_kbqa.wikikgqa.endpoint import SparqlResult, execute
from ama_kbqa.wikikgqa.prompts import (
    REPAIR_EMPTY_RESULT,
    REPAIR_EXECUTION_ERROR,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    format_mentions_block,
)

_EMPTY_SELECT = {"head": {"vars": []}, "results": {"bindings": []}}
_FENCE_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class GeneratedQuery:
    """A generator's output: the chosen query plus its execution result."""

    qid: int
    sparql: str | None
    result: SparqlResult | None
    attempts: int = 1

    @property
    def answer(self) -> dict:
        if self.result and self.result.ok and self.result.json is not None:
            return self.result.json
        return dict(_EMPTY_SELECT)

    @property
    def ok(self) -> bool:
        return bool(self.result and self.result.ok)


class Generator(Protocol):
    """Anything that can answer a WikiKGQA question with a SPARQL query."""

    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery: ...


def strip_sparql(text: str) -> str:
    """Pull a bare SPARQL query out of a model response (strip fences/prose).

    Returns "" when the text contains no SPARQL at all (e.g. an agent-loop error
    string like "Error: Agent reached maximum iteration limit."), so callers can
    tell "no query" apart from a query and fall back to journal recovery instead
    of executing prose.
    """
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # No fence: drop any leading prose before the first SPARQL keyword.
    upper = text.upper()
    for kw in ("PREFIX ", "SELECT ", "ASK ", "ASK{", "CONSTRUCT ", "DESCRIBE "):
        idx = upper.find(kw)
        if idx != -1:
            return text[idx:].strip()
    return ""


def _has_rows(result: SparqlResult) -> bool:
    if not (result.ok and result.json):
        return False
    if "boolean" in result.json:
        return True
    return bool(result.json.get("results", {}).get("bindings"))


# --- class-closure expansion repair (ANSWER_CONVENTIONS.md rule 10) ---
#
# Dominant remaining failure class: the committed query constrains class
# membership with bare `?var wdt:P31 wd:QX .`, but gold uses the transitive
# closure `wdt:P31/wdt:P279*` (or even `wdt:P31*/wdt:P279*`) so subclass
# instances count too. That scores precision=1.0/recall~=0. This used to be a
# prompt rule (R10); a held-out A/B showed prompt-space R7+ didn't help
# overall, so it's now applied mechanically at commit time instead: widen the
# membership pattern textually and adopt only if the wider answer set is a
# STRICT SUPERSET of the committed one (never a same-size or unrelated set).

_ASK_OR_AGGREGATE_RE = re.compile(
    r"\bASK\b"                                 # ASK query form
    r"|\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\("       # aggregate function calls
    r"|\bGROUP\s+BY\b"                         # GROUP BY
    r"|\bHAVING\b",                            # HAVING
    re.IGNORECASE,
)

_STRING_LITERAL_RE = re.compile(
    r"'''(?:\\.|[^\\])*?'''"
    r'|"""(?:\\.|[^\\])*?"""'
    r"|'(?:\\.|[^'\\])*'"
    r'|"(?:\\.|[^"\\])*"',
    re.DOTALL,
)

# Class-membership property-path patterns, most-specific alternative first so
# an already-escalated occurrence is never re-matched as a lower level (regex
# alternation tries alternatives left-to-right at each start position).
_MEMBERSHIP_RE = re.compile(
    r"wdt:P31\*/wdt:P279\*"   # level 2: already at the escalation ceiling
    r"|wdt:P31/wdt:P279\*"    # level 1: one escalation applied
    r"|wdt:P31(?!\*)"         # level 0: bare instance-of
)

_MEMBERSHIP_ESCALATION = {
    0: "wdt:P31/wdt:P279*",
    1: "wdt:P31*/wdt:P279*",
}


def _mask_string_literals(sparql: str) -> str:
    """Blank string-literal contents so the membership regex never matches inside one.

    Length-preserving (replaces each literal with same-length 'x' filler), so
    match spans found in the masked text are valid offsets into the original
    query text too.
    """
    return _STRING_LITERAL_RE.sub(lambda m: "x" * len(m.group(0)), sparql)


def _membership_pattern_level(text: str) -> int:
    if text == "wdt:P31*/wdt:P279*":
        return 2
    if text == "wdt:P31/wdt:P279*":
        return 1
    return 0


def _closure_expansion_eligible(sparql: str, result_json: dict | None) -> bool:
    """Guardrails: SELECT-of-entities only, never ASK or an aggregate query.

    Expanding an ASK or an aggregate (COUNT/SUM/AVG/MIN/MAX/GROUP BY/HAVING)
    would silently change a scalar instead of growing a row set, which the
    strict-superset check can't validate -- so those are refused outright
    rather than risking a wrong "adoption".
    """
    if _ASK_OR_AGGREGATE_RE.search(_mask_string_literals(sparql)):
        return False
    if not result_json or "boolean" in result_json:
        return False
    bindings = result_json.get("results", {}).get("bindings", [])
    if not bindings:
        return False  # 0-row committed results are already handled upstream
    head = result_json.get("head", {}).get("vars", [])
    var = head[0] if head else next(iter(bindings[0]), None)
    # Class membership is a URI concept; only escalate when the projected
    # column is entity/URI-typed (never literal-valued SELECTs).
    return all(row.get(var, {}).get("type") == "uri" for row in bindings if var in row)


def _next_closure_escalation(sparql: str) -> str | None:
    """Escalate the query's lowest-level membership pattern(s) by one step.

    Returns the rewritten query text, or None when there is nothing left to
    escalate: no `wdt:P31` membership pattern at all, or every occurrence is
    already at the `wdt:P31*/wdt:P279*` ceiling. Only the property-path text
    of matching triples is touched -- variable names, object ids, whitespace,
    and string literals all pass through unchanged.
    """
    matches = list(_MEMBERSHIP_RE.finditer(_mask_string_literals(sparql)))
    if not matches:
        return None
    levels = [_membership_pattern_level(m.group(0)) for m in matches]
    min_level = min(levels)
    if min_level >= 2:
        return None  # everything already at the ceiling
    replacement = _MEMBERSHIP_ESCALATION[min_level]
    out: list[str] = []
    last = 0
    for m, level in zip(matches, levels):
        if level != min_level:
            continue
        out.append(sparql[last:m.start()])
        out.append(replacement)
        last = m.end()
    out.append(sparql[last:])
    return "".join(out)


def _answer_set(result_json: dict | None) -> frozenset:
    if not result_json:
        return frozenset()
    from ama_kbqa.wikikgqa.submission import to_codabench_answers

    return frozenset(str(v) for v in to_codabench_answers(result_json))


def _is_strict_superset(candidate_json: dict | None, committed_json: dict | None) -> bool:
    """True iff candidate's answer set properly contains every committed answer."""
    committed = _answer_set(committed_json)
    candidate = _answer_set(candidate_json)
    return candidate.issuperset(committed) and candidate != committed


# --- answer-sanity guard ---
#
# Every gold answer in this benchmark is a clean Q/P entity id, a literal, or a
# boolean -- never a statement node, blank node, or Special:EntityData URL. A
# committed query can return rows and still be known-wrong when its bindings are
# one of those junk shapes (e.g. q113: a 50-row result of `statement/Q...-UUID`
# ids that can never match gold). _result_is_sane names that failure mode so it
# can be treated the same way a 0-row result already is: known-wrong, worth a
# journal-alternate recovery attempt.
def _result_is_sane(result_json: dict | None) -> bool:
    """False when any binding is a blank node, a statement-node URI, a
    Special:EntityData URL, or (catch-all) still contains "wikidata.org/"
    (case-insensitive) after to_codabench_answers normalization -- i.e.
    unmapped Wikidata-URI junk.

    Rule (c) was checked against the gold dataset (data/wikikgqa/wikikgqa.json,
    497 questions): no gold answer value contains "wikidata.org/", so the
    catch-all does not need weakening for THAT substring. It's scoped to
    Wikidata URIs specifically (rather than any "/") because the benchmark also
    has legitimate URL-literal answers (e.g. official-website values like
    "https://www.louvre.fr/") that must not be flagged just for containing a
    slash.

    ASK (boolean) results are always sane -- there are no bindings to inspect.
    """
    if not result_json:
        return False
    if "boolean" in result_json:
        return True
    bindings = result_json.get("results", {}).get("bindings", [])
    for row in bindings:
        for cell in row.values():
            if not isinstance(cell, dict):
                continue
            if cell.get("type") == "bnode":
                return False
            value = cell.get("value", "")
            if "/entity/statement/" in value or "Special:EntityData" in value:
                return False
    from ama_kbqa.wikikgqa.submission import to_codabench_answers

    return not any(
        "wikidata.org/" in v.lower() for v in to_codabench_answers(result_json) if isinstance(v, str)
    )


# --- projection trim (commit-time SELECT column pruning) ---
#
# 416/442 (94%) of gold SELECT queries project exactly one variable. The
# submission answer-flattening (to_codabench_answers) treats EVERY projected
# column as an answer value, so a committed multi-column SELECT is almost
# always precision poison from spurious extra columns (observed: q25 test run
# warned "3 projected vars, gold is single-column"). _trim_projection rewrites
# such a query to keep only its first projected variable; the caller executes
# the trimmed query and adopts it only if it still runs, has rows, and is sane
# -- the first column's own value set is unchanged by construction (same WHERE
# clause, same first variable), so unlike class-closure expansion there is no
# superset check to make.

_SELECT_CLAUSE_RE = re.compile(
    r"\bSELECT\b\s*(?P<mod>DISTINCT|REDUCED)?\s*(?P<vars>[^{]*?)\s*(?=\{|\bWHERE\b)",
    re.IGNORECASE,
)


def _trim_projection(sparql: str) -> str | None:
    """Rewrite a multi-variable non-aggregate SELECT to project only its first variable.

    Returns None (leave the query untouched) when: the query is an ASK or an
    aggregate query (COUNT/SUM/AVG/MIN/MAX/GROUP BY/HAVING -- same guard as
    class-closure expansion, since none of those are a "just extra columns"
    shape); the projection is `SELECT *`; the projection contains an `(...)`
    expression (e.g. `(?x + 1 AS ?y)`); or the query already projects a single
    variable. Detection runs on string-literal-masked text (see
    _mask_string_literals) so a literal that happens to contain "SELECT" or a
    brace never confuses the SELECT-clause match; match offsets are valid into
    the original (unmasked) text because masking is length-preserving.
    """
    masked = _mask_string_literals(sparql)
    if _ASK_OR_AGGREGATE_RE.search(masked):
        return None
    m = _SELECT_CLAUSE_RE.search(masked)
    if not m:
        return None
    vars_text = m.group("vars").strip()
    if not vars_text or "*" in vars_text or "(" in vars_text:
        return None
    tokens = vars_text.split()
    if len(tokens) <= 1:
        return None
    mod = m.group("mod")
    replacement = "SELECT " + (f"{mod} " if mod else "") + tokens[0] + " "
    return sparql[: m.start()] + replacement + sparql[m.end() :]


# --- yes/no ASK repair (commit-time) ---
#
# Gold-scan of data/wikikgqa/wikikgqa.json: all 35/35 yes/no-form questions
# (by surface form) have an ASK gold query, zero exceptions. Failure case
# q403/q405 ("Has France won the Eurovision at least twice?") committed a
# COUNT scalar SELECT instead and scored 0. When the question surface form is
# yes/no and the committed result is not a boolean, re-run generation exactly
# once with an explicit instruction to emit ASK, and adopt the re-run's result
# only if it IS a boolean.

_YESNO_QUESTION_RE = re.compile(
    r"^\s*(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would)\s+",
    re.IGNORECASE,
)

_ASK_REPAIR_INSTRUCTION = (
    "\n\nIMPORTANT: This is a yes/no question. The final SPARQL MUST be an ASK "
    "query returning a boolean. For 'at least N times' questions use "
    "ASK { { SELECT (COUNT(...) AS ?cnt) {...} } FILTER(?cnt >= N) }."
)


def _is_yesno_question(text: str) -> bool:
    return bool(_YESNO_QUESTION_RE.match(text or ""))


def _is_boolean_result(result_json: dict | None) -> bool:
    return bool(result_json and "boolean" in result_json)


class MentionSparqlGenerator:
    """Baseline with-mentions generator: prompt -> SPARQL -> execute -> repair.

    Uses the project's configured chat client. Pass an explicit ``complete``
    callable (``(system, user) -> str``) to unit-test without network/LLM.
    """

    def __init__(
        self,
        complete=None,
        endpoint: str | None = None,
        language: str = "en",
        max_repairs: int = 2,
        timeout: int = 120,
        provider: str | None = None,
        model: str | None = None,
    ):
        self.endpoint = endpoint
        self.language = language
        self.max_repairs = max_repairs
        self.timeout = timeout
        # Per-run provider/model override (e.g. KIT gemma) without editing config.
        self.provider = provider
        self.model = model
        self._complete = complete or self._default_complete

    # --- LLM plumbing (mirrors base_agent's chat.completions call) ---
    def _default_complete(self, system: str, user: str) -> str:
        from ama_kbqa.config import (
            _create_client,
            get_chat_client,
            get_chat_model_name,
            get_provider_preferences,
        )

        if self.provider:
            client = _create_client(self.provider, "chat")
        else:
            client = get_chat_client()
        params = {
            "model": self.model or get_chat_model_name(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        # OpenRouter-only provider routing preferences; None for KIT/local.
        if not self.provider or self.provider == "openrouter":
            prefs = get_provider_preferences()
            if prefs:
                params["extra_body"] = {"provider": prefs}
        resp = client.chat.completions.create(**params)
        # Providers occasionally return a response with no choices; don't crash.
        if not resp or not getattr(resp, "choices", None):
            return ""
        return resp.choices[0].message.content or ""

    # --- generation ---
    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery:
        mentions = question.mentions_for(self.language) or question.mentions
        user = USER_TEMPLATE.format(
            question=question.question(self.language),
            mentions_block=format_mentions_block(mentions),
        )

        sparql = strip_sparql(self._complete(SYSTEM_PROMPT, user))
        result = execute(sparql, endpoint=self.endpoint, timeout=self.timeout)

        attempt = 1
        while attempt <= self.max_repairs:
            if result.ok and _has_rows(result):
                break
            # Build a repair instruction from the failure mode.
            if not result.ok:
                repair = REPAIR_EXECUTION_ERROR.format(
                    error=(result.error or "unknown error")[:500],
                    previous_query=sparql,
                )
            else:
                repair = REPAIR_EMPTY_RESULT.format(previous_query=sparql)
            sparql = strip_sparql(self._complete(SYSTEM_PROMPT, user + "\n\n" + repair))
            result = execute(sparql, endpoint=self.endpoint, timeout=self.timeout)
            attempt += 1

        return GeneratedQuery(qid=question.id, sparql=sparql, result=result, attempts=attempt)


def _augment_question(question: WikiKGQAQuestion, language: str, include_mentions: bool = True) -> str:
    """Render the question for the agent, optionally with its resolved mentions.

    ``include_mentions=False`` simulates the without-mentions track: the given
    QIDs/PIDs are withheld so the agent must link names itself via SearchEntities.
    Used to score the live-API linking path against the (with-mentions) gold set.
    """
    if not include_mentions:
        return f"QUESTION: {question.question(language)}"
    lines = [f"QUESTION: {question.question(language)}", "", "Linked mentions (already resolved for you):"]
    mentions = question.mentions_for(language) or question.mentions
    seen: set[tuple] = set()
    for m in mentions:
        key = (m.entity, m.property, m.inverse)
        if key in seen:
            continue
        seen.add(key)
        parts = [f'  "{m.string}"']
        if m.entity:
            parts.append(f"entity {m.entity}")
        if m.property:
            parts.append(f"property {m.property}" + (" (inverse)" if m.inverse else ""))
        lines.append(" -> ".join(parts) if len(parts) > 1 else parts[0])
    if not mentions:
        lines.append("  (none)")
    return "\n".join(lines)


def _best_query_from_snapshots(agent) -> str | None:
    """Salvage the latest validated non-empty SPARQL query the agent ran.

    Every RunSPARQL is recorded into the journal's ``found_values`` as
    ``sparql_result_N`` with its ``result_count``, and the agent snapshots the
    journal locally after each tool call. So even when the agent is cancelled by
    the wall-clock (or emits nothing at synthesis), we can return the last query
    that actually returned rows instead of an empty answer. Delegates to the
    framework's shared implementation (also used by the agent's own synthesis
    fallback) so the two salvage paths can never diverge.
    """
    from ama_kbqa.framework.base_agent import best_snapshot_query

    return best_snapshot_query(getattr(agent, "journal_snapshots", None) or [])


class AgentSparqlGenerator:
    """Agentic generator: the BaseKBQAAgent tool-loop, emitting a SPARQL query.

    Unlike :class:`MentionSparqlGenerator` (a blind single LLM call), this lets
    the agent EXPLORE the graph to discover properties/paths not present in the
    mentions, which is the dominant failure mode on this challenge. Each question
    gets a fresh agent (and a fresh MCP server process, so the journal resets).
    """

    def __init__(
        self,
        language: str = "en",
        endpoint: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: int = 120,
        agent_timeout: float | None = None,
        conventions: str = "minimal",
        entity_search: bool = False,
        tool_budget: int = 20,
        votes: int = 1,
        closure_expansion: bool = True,
        projection_trim: bool = True,
        ask_repair: bool = True,
    ):
        self.language = language
        self.endpoint = endpoint
        self.model = model
        self.provider = provider
        self.timeout = timeout
        # Without-mentions track: let the agent link names to ids via SearchEntities.
        self.entity_search = entity_search
        # Per-question tool-call budget (raise for the linking-heavy without-mentions track).
        self.tool_budget = tool_budget
        # No wall-clock by default (quality over speed): the tool-call budget bounds
        # the run and forces synthesis, and every LLM/tool call is individually timed
        # and retried with stepped backoff, so a slow-but-healthy endpoint is waited
        # out instead of cancelled mid-query. Set a float only to force a hard cap
        # (e.g. tests exercising the journal-recovery fallback).
        self.agent_timeout = agent_timeout
        # "minimal" = R1-R6 (default; held-out A/B found no R7-R10 benefit);
        # "full" = R1-R10 (opt-in).
        self.conventions = conventions
        # Self-consistency voting over EXECUTED ANSWER SETS (all question types):
        # run the full generation up to `votes` times, stop early as soon as two
        # runs agree on the same answer, and return the modal answer (ties keep
        # the first run's). Run-to-run trajectory divergence, not prompt rules,
        # dominates the residual error (seed-99 A/B, 2026-07-03), which is what
        # this attacks. 1 = off (default).
        self.votes = max(1, int(votes))
        # Commit-time class-closure repair (ANSWER_CONVENTIONS.md rule 10, applied
        # mechanically instead of via prompt): see _next_closure_escalation. On by
        # default; the benchmark CLI flag to disable it is wired separately.
        self.closure_expansion = closure_expansion
        # Commit-time projection trim: see _trim_projection. On by default.
        self.projection_trim = projection_trim
        # Commit-time yes/no ASK repair: see _is_yesno_question / _ASK_REPAIR_INSTRUCTION.
        # On by default.
        self.ask_repair = ask_repair

    def _generate_once(
        self, question: WikiKGQAQuestion, _repair_instruction: str | None = None
    ) -> GeneratedQuery:
        import asyncio

        from ama_kbqa.agents.wikidata_agent.agent import WikidataAgent

        # In entity_search (without-mentions) mode, withhold the given mentions so the
        # agent is forced to link names itself — exercising the live-API linking path.
        augmented = _augment_question(question, self.language, include_mentions=not self.entity_search)
        # ``_repair_instruction`` is set ONLY by the yes/no ASK-repair re-run below
        # (see the block after class-closure expansion): it is never set on the
        # outward-facing call, so this appended instruction can never itself
        # trigger another repair -- see the `_repair_instruction is None` guard
        # further down, which is what actually prevents recursion.
        if _repair_instruction:
            augmented += _repair_instruction

        async def _run() -> tuple[str, str | None]:
            agent = WikidataAgent(
                model=self.model,
                provider=self.provider,
                conventions=self.conventions,
                entity_search=self.entity_search,
                tool_budget=self.tool_budget,
            )
            raw = ""
            try:
                if self.agent_timeout:  # optional hard cap; default is None (no wall-clock)
                    raw = await asyncio.wait_for(agent.ask(augmented), timeout=self.agent_timeout)
                else:
                    raw = await agent.ask(augmented)
            except Exception:  # timeout / agent / MCP failure: fall through to recovery
                raw = ""
            # Read the journal snapshots (local, survives cancellation) BEFORE closing,
            # so a cancelled or empty-synthesis run can still emit its best query.
            recovered = _best_query_from_snapshots(agent)
            try:
                await agent.close()
            except Exception:
                pass
            return raw, recovered

        try:
            raw, recovered = asyncio.run(_run())
        except Exception:  # event-loop level failure -> nothing to salvage
            raw, recovered = "", None

        sparql = strip_sparql(raw)
        used_recovery = False
        if not sparql and recovered:  # synthesis produced nothing: fall back to the journal
            sparql = strip_sparql(recovered)
            used_recovery = bool(sparql)
        if not sparql:
            return GeneratedQuery(question.id, None, None, attempts=1)
        result = execute(sparql, endpoint=self.endpoint, timeout=self.timeout)
        # Non-empty prior: every gold answer in this benchmark is non-empty, so a
        # committed query that fails or returns 0 rows is known-wrong. If the journal
        # holds a different query that returned rows during exploration, prefer it.
        if not _has_rows(result) and recovered and not used_recovery:
            alt = strip_sparql(recovered)
            if alt and alt != sparql:
                alt_result = execute(alt, endpoint=self.endpoint, timeout=self.timeout)
                if _has_rows(alt_result):
                    sparql, result, used_recovery = alt, alt_result, True
        # Projection trim: 94% of gold SELECT queries project exactly one
        # variable, so a committed multi-column SELECT is almost always
        # precision poison from spurious extra columns (to_codabench_answers
        # treats every projected column as an answer value). Adopt the
        # first-variable-only rewrite when it still executes, has rows, and is
        # sane; keep the original on any failure. Runs before the sanity guard
        # (a trimmed result can drop junk that only lived in a dropped column)
        # and before class-closure expansion (which expects a single-column
        # query anyway).
        if self.projection_trim and _has_rows(result):
            trimmed = _trim_projection(sparql)
            if trimmed and trimmed != sparql:
                trimmed_result = execute(trimmed, endpoint=self.endpoint, timeout=self.timeout)
                if _has_rows(trimmed_result) and _result_is_sane(trimmed_result.json):
                    sparql, result = trimmed, trimmed_result
        # Answer-sanity guard: a committed result WITH rows can still be
        # known-wrong when those rows are unsane (blank nodes / statement-node
        # URIs / Special:EntityData URLs / unmapped URI junk -- see
        # _result_is_sane). Try the same journal-alternate recovery as the
        # 0-row path above; adopt the alternate only if it both has rows AND is
        # sane. If no sane alternate exists, keep the original result -- an
        # unsane answer is not worse than an empty one (both score 0), and the
        # empty submission fallback is guaranteed 0 anyway.
        if _has_rows(result) and not _result_is_sane(result.json) and recovered and not used_recovery:
            alt = strip_sparql(recovered)
            if alt and alt != sparql:
                alt_result = execute(alt, endpoint=self.endpoint, timeout=self.timeout)
                if _has_rows(alt_result) and _result_is_sane(alt_result.json):
                    sparql, result, used_recovery = alt, alt_result, True
        # Class-closure expansion: widen bare wdt:P31 membership patterns to the
        # transitive closure and adopt only if the wider answer set is a strict
        # superset of the committed one (never a same-size or unrelated set), so
        # a wrong broadening can never be adopted. Runs on whatever query/result
        # ended up committed above (original or non-empty-prior swap).
        if self.closure_expansion and _has_rows(result) and _closure_expansion_eligible(sparql, result.json):
            # `base` walks the escalation ladder; (sparql, result) is only
            # re-pointed when the answer set strictly grows. An equal-set
            # escalation must pass through rather than stop the loop (verified
            # live on q110: P31 -> P31/P279* keeps the same 1 row, and only
            # P31*/P279* reaches gold's 564) — but it is not adopted either, so
            # a no-op ladder leaves the committed query text untouched.
            # Anything that loses or swaps rows still stops the loop.
            base = sparql
            for _ in range(2):  # at most two escalations: bare -> P279* -> *P279*
                candidate = _next_closure_escalation(base)
                if candidate is None or candidate == base:
                    break
                candidate_result = execute(candidate, endpoint=self.endpoint, timeout=self.timeout)
                if not candidate_result.ok:
                    break
                candidate_answers = _answer_set(candidate_result.json)
                committed_answers = _answer_set(result.json)
                if not candidate_answers.issuperset(committed_answers):
                    break
                # A strictly-growing candidate is only adopted when it's also
                # sane; an unsane candidate (e.g. the widened pattern now pulls
                # in blank/statement nodes) must not replace a sane committed
                # result, even though its answer set is nominally a superset.
                if candidate_answers != committed_answers and _result_is_sane(candidate_result.json):
                    sparql, result = candidate, candidate_result
                base = candidate
        # Yes/no ASK repair: every yes/no-form question in gold has an ASK
        # query, so a committed non-boolean result on a yes/no question is
        # known-wrong. Re-run generation exactly once with an explicit
        # instruction to emit ASK, and adopt the re-run's result only if it IS
        # a boolean -- keep the original (non-boolean) result otherwise, since
        # a failed repair attempt is not worse than the status quo.
        # `_repair_instruction is None` both identifies this as the OUTER call
        # (not a nested repair re-run) and is what stops the recursion: the
        # nested `_generate_once` call below is invoked WITH a
        # `_repair_instruction`, so its own copy of this same condition is
        # False and it can never trigger a further repair.
        ask_repair_used = False
        if (
            self.ask_repair
            and _repair_instruction is None
            and _is_yesno_question(question.question(self.language))
            and not _is_boolean_result(result.json)
        ):
            repaired = self._generate_once(question, _repair_instruction=_ASK_REPAIR_INSTRUCTION)
            if _is_boolean_result(repaired.result.json if repaired.result else None):
                sparql, result = repaired.sparql, repaired.result
                ask_repair_used = True
        attempts = (2 if used_recovery else 1) + (1 if ask_repair_used else 0)
        return GeneratedQuery(qid=question.id, sparql=sparql, result=result, attempts=attempts)

    @staticmethod
    def _answer_key(gq: GeneratedQuery):
        """Comparable identity of an executed answer.

        Booleans key on their value; SELECT results key on the frozen set of
        codabench-reduced values (bare ids / literal strings) so two different
        queries with the same answer set vote together. None (= empty or
        failed) can never win a vote: the benchmark's answers are non-empty.
        """
        if not (gq.result and gq.result.ok and gq.result.json):
            return None
        j = gq.result.json
        if "boolean" in j:
            return ("bool", bool(j["boolean"]))
        from ama_kbqa.wikikgqa.submission import to_codabench_answers

        vals = to_codabench_answers(j)
        if not vals:
            return None
        return ("set", frozenset(str(v) for v in vals))

    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery:
        from collections import Counter

        first = self._generate_once(question)
        if self.votes <= 1:
            return first
        runs = [first]
        while len(runs) < self.votes:
            keys = [k for k in (self._answer_key(r) for r in runs) if k is not None]
            if keys and Counter(keys).most_common(1)[0][1] >= 2:
                break  # consensus: two runs already agree on the answer
            runs.append(self._generate_once(question))
        counts = Counter(k for k in (self._answer_key(r) for r in runs) if k is not None)
        if not counts:
            return first  # every run came back empty/failed
        # Counter preserves insertion order on equal counts, so a tie resolves
        # to the earliest-seen answer (the first run's, when it produced one) --
        # UNLESS one of the tied answer sets is unsane (blank/statement nodes,
        # see _result_is_sane) and another tied one is sane, in which case the
        # sane answer wins the tie: an unsane answer set can never match gold,
        # so it should never beat an equally-voted sane alternative.
        entries = counts.most_common()
        top_count = entries[0][1]
        tied_keys = [k for k, c in entries if c == top_count]
        if len(tied_keys) == 1:
            best_key = tied_keys[0]
        else:
            def _key_is_sane(key) -> bool:
                return any(
                    self._answer_key(r) == key
                    and r.result is not None
                    and r.result.json is not None
                    and _result_is_sane(r.result.json)
                    for r in runs
                )

            sane_tied = [k for k in tied_keys if _key_is_sane(k)]
            best_key = sane_tied[0] if sane_tied else tied_keys[0]
        for r in runs:
            if self._answer_key(r) == best_key:
                return r
        return first

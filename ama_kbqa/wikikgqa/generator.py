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
    """Pull a bare SPARQL query out of a model response (strip fences/prose)."""
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
    return text


def _has_rows(result: SparqlResult) -> bool:
    if not (result.ok and result.json):
        return False
    if "boolean" in result.json:
        return True
    return bool(result.json.get("results", {}).get("bindings"))


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
    that actually returned rows instead of an empty answer. Picks the
    highest-numbered (most recent) query with a positive result_count — the best
    proxy for what the agent would have committed.
    """
    snaps = getattr(agent, "journal_snapshots", None) or []
    best_idx, best_query = -1, None
    for snap in snaps:
        found = (snap.get("state") or {}).get("found_values") or {}
        for key, val in found.items():
            if not (isinstance(val, dict) and key.startswith("sparql_result")):
                continue
            query = val.get("query")
            if not query or (val.get("result_count") or 0) <= 0:
                continue
            try:
                idx = int(key.rsplit("_", 1)[-1])
            except ValueError:
                idx = 0
            if idx > best_idx:
                best_idx, best_query = idx, query
    return best_query


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
        conventions: str = "full",
        entity_search: bool = False,
        tool_budget: int = 20,
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
        # "full" = R1-R10; "minimal" = R1-R6 (for held-out A/B of the R7-R10 conventions).
        self.conventions = conventions

    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery:
        import asyncio

        from ama_kbqa.agents.wikidata_agent.agent import WikidataAgent

        # In entity_search (without-mentions) mode, withhold the given mentions so the
        # agent is forced to link names itself — exercising the live-API linking path.
        augmented = _augment_question(question, self.language, include_mentions=not self.entity_search)

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
        return GeneratedQuery(
            qid=question.id, sparql=sparql, result=result, attempts=2 if used_recovery else 1
        )

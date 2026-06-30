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


def _augment_question(question: WikiKGQAQuestion, language: str) -> str:
    """Render the question plus its resolved mentions for the agent."""
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
        agent_timeout: float = 280.0,
        conventions: str = "full",
    ):
        self.language = language
        self.endpoint = endpoint
        self.model = model
        self.provider = provider
        self.timeout = timeout
        # Wall-clock SAFETY NET only. The binding limit is the agent's tool-call
        # budget (max_tool_calls), which forces synthesis and emits a query. This
        # cap just catches a genuinely stuck run; it should rarely fire now.
        self.agent_timeout = agent_timeout
        # "full" = R1-R10; "minimal" = R1-R6 (for held-out A/B of the R7-R10 conventions).
        self.conventions = conventions

    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery:
        import asyncio

        from ama_kbqa.agents.wikidata_agent.agent import WikidataAgent

        augmented = _augment_question(question, self.language)

        async def _run() -> str:
            agent = WikidataAgent(
                model=self.model, provider=self.provider, conventions=self.conventions
            )
            try:
                return await asyncio.wait_for(agent.ask(augmented), timeout=self.agent_timeout)
            finally:
                try:
                    await agent.close()
                except Exception:
                    pass

        try:
            raw = asyncio.run(_run())
        except Exception:  # timeout / agent / MCP failure -> empty answer, keep going
            return GeneratedQuery(question.id, None, None, attempts=1)

        sparql = strip_sparql(raw)
        result = execute(sparql, endpoint=self.endpoint, timeout=self.timeout)
        return GeneratedQuery(qid=question.id, sparql=sparql, result=result, attempts=1)

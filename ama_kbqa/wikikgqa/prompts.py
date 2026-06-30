"""Prompts for WikiKGQA SPARQL generation (with-mentions track).

The with-mentions task hands us the entity QIDs and property PIDs already
resolved, so the model's job is narrowed to *assembling* a correct Wikidata
SPARQL query: choosing direction, qualifiers (p:/ps:/pq:), aggregation, and
ASK-vs-SELECT shape. These prompts lean on the provided mentions and keep the
output to a single bare SPARQL query so it can be executed verbatim.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an expert at writing SPARQL queries for Wikidata.

You will be given a natural-language question and a list of RESOLVED MENTIONS:
the Wikidata entity IDs (Q...) and property IDs (P...) that appear in the
question, already linked for you. Use these IDs directly — do not guess other
IDs.

Output requirements:
- Return EXACTLY ONE SPARQL query and NOTHING else: no prose, no markdown
  fences, no comments.
- Use the standard Wikidata prefixes (wd:, wdt:, p:, ps:, pq:, rdfs:, etc.);
  do not redeclare them.
- Prefer `wdt:` for simple direct statements. Use `p:`/`ps:`/`pq:` only when the
  question needs a qualifier (e.g. a role, a point in time, a rank).
- For "how many" questions use COUNT. For yes/no questions use ASK.
- Return entity IDs as the answer where the question asks "who/what/which";
  only return labels when the question explicitly asks for a name/title, and
  even then prefer returning the entity unless a label is clearly required.
- A `mention` with "inverse": true means the relation runs from object to
  subject — swap the triple direction accordingly.
- Keep the query minimal and executable against the Wikidata SPARQL endpoint."""

USER_TEMPLATE = """QUESTION: {question}

RESOLVED MENTIONS:
{mentions_block}

Write the single SPARQL query that answers the question."""

REPAIR_EXECUTION_ERROR = """The previous query failed to execute with this error:
{error}

Previous query:
{previous_query}

Return a corrected single SPARQL query."""

REPAIR_EMPTY_RESULT = """The previous query executed but returned NO results, which is likely wrong
for this question. Reconsider the triple direction, the choice of
wdt:/p:/ps:/pq:, and whether a qualifier is needed.

Previous query:
{previous_query}

Return a corrected single SPARQL query."""


def format_mentions_block(mentions) -> str:
    """Render resolved mentions as compact lines for the prompt.

    Each line: surface string -> entity/property id (+ inverse flag).
    """
    lines: list[str] = []
    for m in mentions:
        parts = [f'"{m.string}"']
        if m.entity:
            parts.append(f"entity={m.entity}")
        if m.property:
            tag = f"property={m.property}"
            if m.inverse:
                tag += " (inverse)"
            parts.append(tag)
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines) if lines else "  (none provided)"

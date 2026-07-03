"""MCP server exposing Wikidata exploration tools for the WikiKGQA agent.

This is the Wikidata counterpart of ``kqapro_server.py``. The crucial difference
is that there is NO local index: the graph is far too large to download/embed, so
every tool issues live SPARQL against the challenge endpoint (the endpoint *is*
the index). That live exploration is exactly what addresses the dominant
structural blocker measured on this challenge: ~62% of gold queries need a
property that is NOT in the given mentions, so the agent must DISCOVER properties
and paths by probing the graph rather than guessing them.

Tool groups:
  * Journal/state (reused framework JournalState): ManageJournal,
    GetJournalStateJSON, GetJournalSummary.
  * Execution: RunSPARQL (truncated; exploratory use, not the final answer).
  * Discovery (the blocker-#1 tools): GetEntityProperties, GetRelationBetween,
    GetPropertyInfo, GetLabels.

Entity/property *linking* (fuzzy search) is gated behind a config flag. For the
with-mentions track the QIDs/PIDs are given and no linker is needed. For the
without-mentions track, set ``WIKIKGQA_ENTITY_SEARCH=1`` (surfaced by the
``entity_search`` config parameter on WikidataAgent/AgentSparqlGenerator/the
benchmark CLI) to register the ``SearchEntities`` tool, which resolves names to
Wikidata ids via the Wikidata search API. It is off by default so the with-mentions
tool list stays lean.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Literal

from fastmcp import Context, FastMCP

from ama_kbqa.framework.state import JournalState
from ama_kbqa.wikikgqa.endpoint import USER_AGENT, execute

mcp = FastMCP("Wikidata-KG-Server")

# Per-process journal. The agent restarts the server per question, so module
# state resets automatically between questions (same contract as kqapro_server).
session_journal = JournalState(kg_name="Wikidata")

# Exploratory SPARQL is truncated to protect the agent's context window. The
# FINAL answer query is executed separately (in the generator/submission path)
# with no truncation, preserving the "answer query must not truncate" invariant.
_MAX_ROWS = 30
# Per-tool SPARQL timeout: an expensive/too-broad query returns a descriptive
# TIMEOUT message to the agent (not a hard hang), so it can narrow and retry.
_TOOL_SPARQL_TIMEOUT = 30

# Tool-call budget surfaced live to the agent. Matches the agent's max_tool_calls
# so the running "[tool call N/BUDGET]" tag lines up with when the framework forces
# synthesis. Resets per question (the server process restarts each question).
_TOOL_BUDGET = int(os.environ.get("WIKIKGQA_TOOL_BUDGET", "20"))
_tool_call_count = 0

# Entity linking (without-mentions track) is gated: the SearchEntities tool is only
# registered when WIKIKGQA_ENTITY_SEARCH is truthy, which the `entity_search` config
# parameter sets before the agent spawns this server. Keeps with-mentions runs lean.
_ENTITY_SEARCH_ENABLED = os.environ.get("WIKIKGQA_ENTITY_SEARCH", "").strip().lower() in {
    "1", "true", "yes", "on",
}
# The frozen challenge endpoint has no search API, so linking hits the LIVE Wikidata
# search API. QIDs are stable identifiers, so an id found here resolves in the frozen
# dump too (and the agent validates every path with RunSPARQL against the endpoint).
_SEARCH_API = os.environ.get("WIKIDATA_SEARCH_API", "https://www.wikidata.org/w/api.php")
_SEARCH_LIMIT = 7


def _budgeted(fn):
    """Append a live '[tool call N/BUDGET]' tag to a tool's string result so the
    agent knows how much budget it has used and commits before running out."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _tool_call_count
        _tool_call_count += 1
        out = fn(*args, **kwargs)
        if not isinstance(out, str) or out.lstrip()[:1] in "{[":
            return out  # don't corrupt machine-readable (JSON) results
        tag = f"\n\n[tool call {_tool_call_count}/{_TOOL_BUDGET}]"
        if _tool_call_count >= _TOOL_BUDGET - 4:
            tag += " near budget: stop exploring and submit your best validated query."
        return out + tag

    return wrapper

_ENTITY = "http://www.wikidata.org/entity/"
_WDT = "http://www.wikidata.org/prop/direct/"


def _short(uri: str) -> str:
    """Abbreviate a Wikidata URI to its bare id / value for compact display."""
    if uri.startswith(_ENTITY):
        return uri[len(_ENTITY):]
    if uri.startswith(_WDT):
        return uri[len(_WDT):]
    return uri


def _cell(binding: dict) -> str:
    """Render one SPARQL-JSON binding cell as a short string."""
    return _short(binding.get("value", ""))


def _valid_id(value: str, prefix: str) -> bool:
    return bool(re.fullmatch(prefix + r"\d+", value.strip()))


# --------------------------------------------------------------------------- #
# Journal tools (reuse the framework JournalState; mirror kqapro_server)
# --------------------------------------------------------------------------- #
@mcp.tool
@_budgeted
def ManageJournal(
    action: Literal[
        "update_plan", "set_question", "set_qtype", "set_target",
        "set_partial_answer", "add_fact", "read", "clear",
    ],
    content: str,
    context: Context,
) -> str:
    """Track progress so you do not loop. Returns the full journal afterwards.

    Actions: update_plan, set_question, set_qtype, set_target,
    set_partial_answer, add_fact (subject|predicate|object), read, clear.
    """
    global session_journal
    if action == "update_plan":
        session_journal.current_plan = [s.strip() for s in content.split("\n") if s.strip()]
    elif action == "set_question":
        session_journal.question_text = content
    elif action == "set_qtype":
        session_journal.question_type = content
    elif action == "set_target":
        if content and content not in session_journal.target_entities:
            session_journal.target_entities.append(content)
    elif action == "set_partial_answer":
        session_journal.partial_answer = content
    elif action == "add_fact":
        parts = [p.strip() for p in content.split("|")]
        if len(parts) >= 3:
            session_journal.verified_facts.append(
                {"subject": parts[0], "relation": parts[1], "related_id": parts[2], "source": "agent"}
            )
    elif action == "clear":
        session_journal = JournalState(kg_name="Wikidata")
    return session_journal.to_str()


@mcp.tool
def GetJournalStateJSON(context: Context) -> str:
    """Return the structured journal state as JSON (used by the live graph view)."""
    return json.dumps(session_journal.model_dump(), default=str, ensure_ascii=False)


@mcp.tool
def GetJournalSummary(context: Context) -> str:
    """Formatted summary of everything discovered. Use before the final answer."""
    return session_journal.to_summary_str()


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
@mcp.tool
@_budgeted
def RunSPARQL(query: str, context: Context) -> str:
    """Execute a SPARQL query against the Wikidata endpoint (exploratory).

    Standard Wikidata prefixes (wd:, wdt:, p:, ps:, pq:, rdfs:, ...) are
    available without declaration. Results are TRUNCATED for context safety, so
    use this to inspect/validate, not to fetch a full large answer set.
    """
    # Exploration must fail fast: cap unbounded SELECTs and use a short timeout so
    # an accidental cross-product can't stall the agent (the final answer query is
    # executed separately, without these caps).
    probe = query
    upper = query.upper()
    if upper.lstrip().startswith("SELECT") and "LIMIT" not in upper:
        probe = query.rstrip().rstrip(".") + "\nLIMIT 1000"
    result = execute(probe, timeout=_TOOL_SPARQL_TIMEOUT, retries=1)
    if not result.ok or result.json is None:
        err = result.error or "query failed"
        session_journal.add_failed_attempt(f"SPARQL error: {err[:120]}")
        if "timeout" in err.lower() or "timed out" in err.lower():
            return (
                f"TIMEOUT: this query did not return within {_TOOL_SPARQL_TIMEOUT}s. It is "
                "probably too expensive or matches too many rows. Make it MORE SPECIFIC: "
                "add a tighter type constraint or FILTER, restrict the entity set, or add a "
                "LIMIT for inspection, then retry. Do not re-run the same broad query."
            )
        return f"ERROR: {err[:300]}"

    if "boolean" in result.json:
        return f"ASK result: {result.json['boolean']}"

    head = result.json.get("head", {}).get("vars", [])
    rows = result.json.get("results", {}).get("bindings", [])
    total = len(rows)
    shown = rows[:_MAX_ROWS]
    lines = [f"vars: {head}  | rows: {total}" + (f" (showing {_MAX_ROWS})" if total > _MAX_ROWS else "")]
    for r in shown:
        lines.append("  " + " | ".join(f"{v}={_cell(r[v])}" for v in head if v in r))
    if total == 0:
        # Every answer in this benchmark is non-empty, so 0 rows means the PATH is
        # wrong (not that the answer is empty). Point the agent at graph evidence
        # rather than letting it finalize an empty query.
        lines.append(
            "NOTE: 0 rows. In this benchmark every answer is non-empty, so an empty "
            "result means the query path is WRONG. Do not finalize this. Verify the "
            "correct predicate/direction with GetEntityProperties or "
            "GetIncomingRelations on the relevant entity, then fix the query and re-run."
        )
    # Record into the journal so synthesis can see what was retrieved.
    key = f"sparql_result_{len(session_journal.found_values) + 1}"
    session_journal.found_values[key] = {
        # Full query (not truncated): synthesis emits the validated query from here.
        "query": query,
        "vars": head,
        "rows": [{v: _cell(r[v]) for v in head if v in r} for r in shown],
        "result_count": total,
    }
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Discovery tools — the blocker-#1 machinery
# --------------------------------------------------------------------------- #
@mcp.tool
@_budgeted
def GetEntityProperties(entity_id: str, context: Context) -> str:
    """List the direct (wdt:) properties an entity HAS, with labels + a sample value.

    This is the core discovery tool: given a known entity (e.g. a QID from the
    mentions), see which relations actually exist on it so you can pick the right
    property for the query instead of guessing. Returns property id, label, a
    sample object, and how many objects that property has.
    """
    entity_id = entity_id.strip()
    if not _valid_id(entity_id, "Q"):
        return f"ERROR: '{entity_id}' is not a valid entity id (expected Q<number>)."
    q = (
        "SELECT ?p (SAMPLE(?o) AS ?sample) (COUNT(?o) AS ?cnt) WHERE { "
        f"wd:{entity_id} ?p ?o . "
        f'FILTER(STRSTARTS(STR(?p), "{_WDT}")) '
        "} GROUP BY ?p ORDER BY DESC(?cnt)"
    )
    r = execute(q, timeout=90)
    if not r.ok or r.json is None:
        return f"ERROR: {(r.error or 'query failed')[:200]}"
    rows = r.json.get("results", {}).get("bindings", [])
    if not rows:
        return f"{entity_id} has no direct properties (or does not exist)."
    pids = [_short(b["p"]["value"]) for b in rows if "p" in b]
    labels = _labels_for(pids)
    out = [f"{entity_id} has {len(rows)} direct properties:"]
    for b in rows[:_MAX_ROWS]:
        pid = _short(b["p"]["value"])
        sample = _cell(b["sample"]) if "sample" in b else "?"
        cnt = b.get("cnt", {}).get("value", "?")
        out.append(f"  {pid} ({labels.get(pid, '?')})  e.g. {sample}  [{cnt} value(s)]")
    session_journal.visited_nodes.setdefault(entity_id, labels.get(entity_id, entity_id))
    return "\n".join(out)


def _summarize_relations(entity_id: str, incoming: bool) -> tuple[str, str | None]:
    """Bounded probe of an entity's direct (wdt:) relations in one direction.

    Uses a LIMITed sample (early termination) instead of an unbounded
    GROUP BY/COUNT, so probing a heavily-referenced node can't blow up. Returns
    (formatted_text, error).
    """
    if incoming:
        var, pattern, vkey = "?s", f"?s ?p wd:{entity_id}", "s"
        extra = ""
    else:
        var, pattern, vkey = "?o", f"wd:{entity_id} ?p ?o", "o"
        extra = f' FILTER(isIRI(?o) && STRSTARTS(STR(?o), "{_ENTITY}"))'  # entity objects only
    q = (
        f"SELECT ?p {var} WHERE {{ {pattern} . "
        f'FILTER(STRSTARTS(STR(?p), "{_WDT}")){extra} }} LIMIT 600'
    )
    r = execute(q, timeout=40, retries=1)
    if not r.ok or r.json is None:
        return "", (r.error or "query failed")[:200]
    rows = r.json.get("results", {}).get("bindings", [])
    by_p: dict[str, list[str]] = {}
    for b in rows:
        if "p" not in b or vkey not in b:
            continue
        by_p.setdefault(_short(b["p"]["value"]), []).append(_short(b[vkey]["value"]))
    if not by_p:
        return "", None
    labels = _labels_for(list(by_p))
    ordered = sorted(by_p.items(), key=lambda kv: len(kv[1]), reverse=True)
    arrow = "<-" if incoming else "->"
    out = []
    for pid, examples in ordered[:_MAX_ROWS]:
        n = len(examples)
        out.append(
            f"  {pid} ({labels.get(pid, '?')}) {arrow} e.g. {examples[0]}"
            f"  [{n}{'+' if n >= 600 else ''} in sample]"
        )
    return "\n".join(out), None


@mcp.tool
@_budgeted
def GetIncomingRelations(entity_id: str, context: Context) -> str:
    """Entities that POINT TO this one, by property (backward traversal).

    The key tool for "reverse" structure: e.g. tournament editions that are
    `P3450` (sports season of) a competition, or items whose `P1346` (winner) is
    this entity. Use this when an entity has no useful outgoing edge for the
    question (the answer lives on entities that reference it).
    """
    entity_id = entity_id.strip()
    if not _valid_id(entity_id, "Q"):
        return f"ERROR: '{entity_id}' is not a valid entity id (expected Q<number>)."
    text, err = _summarize_relations(entity_id, incoming=True)
    if err:
        return f"ERROR: {err}"
    if not text:
        return f"Nothing points to {entity_id} via a direct property (in the sample)."
    return f"Entities pointing TO {entity_id} (incoming):\n{text}"


@mcp.tool
@_budgeted
def GetOutgoingRelations(entity_id: str, context: Context) -> str:
    """Entities this one POINTS TO, by property (forward traversal, entity objects only).

    Use to follow the graph forward to other entities (e.g. an edition -> its
    winner -> the winner's country). For literal attributes (population, dates)
    use GetEntityProperties instead.
    """
    entity_id = entity_id.strip()
    if not _valid_id(entity_id, "Q"):
        return f"ERROR: '{entity_id}' is not a valid entity id (expected Q<number>)."
    text, err = _summarize_relations(entity_id, incoming=False)
    if err:
        return f"ERROR: {err}"
    if not text:
        return f"{entity_id} has no outgoing relations to other entities (in the sample)."
    return f"Entities {entity_id} points TO (outgoing):\n{text}"


@mcp.tool
@_budgeted
def GetRelationBetween(entity_a: str, entity_b: str, context: Context) -> str:
    """Find how two entities are connected: direct properties in either direction.

    Use this to discover the predicate linking two known entities (e.g. which
    property connects a film to its director), including via the qualifier/
    statement layer when there is no direct edge.
    """
    a, b = entity_a.strip(), entity_b.strip()
    if not (_valid_id(a, "Q") and _valid_id(b, "Q")):
        return "ERROR: both arguments must be entity ids (Q<number>)."
    q = (
        "SELECT ?p ?dir WHERE { "
        f'{{ wd:{a} ?p wd:{b} . BIND("forward" AS ?dir) }} UNION '
        f'{{ wd:{b} ?p wd:{a} . BIND("reverse" AS ?dir) }} '
        f'FILTER(STRSTARTS(STR(?p), "{_WDT}")) }}'
    )
    r = execute(q, timeout=90)
    if not r.ok or r.json is None:
        return f"ERROR: {(r.error or 'query failed')[:200]}"
    rows = r.json.get("results", {}).get("bindings", [])
    if not rows:
        return (
            f"No direct edge between {a} and {b}. They may be connected via an "
            "intermediate node or the statement/qualifier layer; try exploring "
            f"GetEntityProperties({a}) and inspecting promising statements."
        )
    pids = [_short(b_["p"]["value"]) for b_ in rows]
    labels = _labels_for(pids)
    out = [f"Connections between {a} and {b}:"]
    for b_ in rows:
        pid = _short(b_["p"]["value"])
        direction = b_.get("dir", {}).get("value", "")
        arrow = f"{a} -{pid}-> {b}" if direction == "forward" else f"{b} -{pid}-> {a}"
        out.append(f"  {pid} ({labels.get(pid, '?')})  [{direction}]  {arrow}")
    return "\n".join(out)


@mcp.tool
@_budgeted
def GetPropertyInfo(property_id: str, context: Context) -> str:
    """Get a property's label, description and aliases to confirm it is the right one."""
    pid = property_id.strip()
    if not _valid_id(pid, "P"):
        return f"ERROR: '{pid}' is not a valid property id (expected P<number>)."
    q = (
        "SELECT ?l ?d (GROUP_CONCAT(DISTINCT ?a; SEPARATOR=\" | \") AS ?aliases) WHERE { "
        f'OPTIONAL {{ wd:{pid} rdfs:label ?l . FILTER(LANG(?l)="en") }} '
        f'OPTIONAL {{ wd:{pid} schema:description ?d . FILTER(LANG(?d)="en") }} '
        f'OPTIONAL {{ wd:{pid} skos:altLabel ?a . FILTER(LANG(?a)="en") }} '
        "} GROUP BY ?l ?d"
    )
    r = execute(q, timeout=60)
    if not r.ok or r.json is None:
        return f"ERROR: {(r.error or 'query failed')[:200]}"
    rows = r.json.get("results", {}).get("bindings", [])
    if not rows:
        return f"No info found for {pid}."
    b = rows[0]
    label = b.get("l", {}).get("value", "?")
    desc = b.get("d", {}).get("value", "")
    aliases = b.get("aliases", {}).get("value", "")
    return f"{pid}: {label}\n  description: {desc}\n  aliases: {aliases}"


_QUALIFIER_PREFIX = "http://www.wikidata.org/prop/qualifier/"


@mcp.tool
@_budgeted
def GetStatementQualifiers(entity_id: str, property_id: str, context: Context) -> str:
    """Show an entity's statements for a property INCLUDING their qualifiers (the p:/ps:/pq: layer).

    GetEntityProperties only shows a property's direct (wdt:) value. Use THIS when the
    answer lives on a qualifier of a statement — e.g. a person's middle name is the
    p:P735 statement whose pq:P1545 (series ordinal) is "2"; a review score is the
    p:P444 statement qualified by pq:P459 (determination method). It lists each
    statement's main value and every qualifier (id, label, value) so you can pick the
    right pq: predicate for your query instead of guessing.
    """
    e, p = entity_id.strip(), property_id.strip()
    if not _valid_id(e, "Q"):
        return f"ERROR: '{e}' is not a valid entity id (expected Q<number>)."
    if not _valid_id(p, "P"):
        return f"ERROR: '{p}' is not a valid property id (expected P<number>)."
    # Two steps so a slow qualifier scan can't cost us the statement values. The
    # main-value query is cheap; the qualifier query uses an unbound predicate
    # (?stmt ?q ?qv), which QLever can be slow on, so it gets a tight timeout and
    # degrades gracefully to values-only rather than hanging.
    base = f"SELECT ?stmt ?val WHERE {{ wd:{e} p:{p} ?stmt . ?stmt ps:{p} ?val . }} LIMIT 300"
    rb = execute(base, timeout=30, retries=1)
    if not rb.ok or rb.json is None:
        return f"ERROR: {(rb.error or 'query failed')[:200]}"
    base_rows = rb.json.get("results", {}).get("bindings", [])
    if not base_rows:
        return f"{e} has no p:{p} statements (or the property is not present)."
    stmts: dict[str, dict] = {}
    for b in base_rows:
        stmts.setdefault(b["stmt"]["value"], {"val": _cell(b["val"]), "quals": []})

    qq = (
        f"SELECT ?stmt ?q ?qv WHERE {{ wd:{e} p:{p} ?stmt . ?stmt ?q ?qv . "
        f'FILTER(STRSTARTS(STR(?q), "{_QUALIFIER_PREFIX}")) }} LIMIT 300'
    )
    rq = execute(qq, timeout=25, retries=0)
    qual_note = ""
    if rq.ok and rq.json is not None:
        for b in rq.json.get("results", {}).get("bindings", []):
            d = stmts.get(b["stmt"]["value"])
            if d is not None and "q" in b and "qv" in b:
                d["quals"].append((b["q"]["value"].rsplit("/", 1)[-1], _cell(b["qv"])))
    else:
        qual_note = "  (qualifier lookup timed out; showing statement values only)"
    # Resolve labels for the property, all qualifier P-ids, and any entity-valued ids.
    label_ids = [p]
    for d in stmts.values():
        if _valid_id(d["val"], "Q"):
            label_ids.append(d["val"])
        for qpid, qv in d["quals"]:
            label_ids.append(qpid)
            if _valid_id(qv, "Q"):
                label_ids.append(qv)
    labels = _labels_for(label_ids)

    def _lbl(x: str) -> str:
        return f"{x} ({labels[x]})" if x in labels and labels[x] else x

    out = [f"{e} p:{p} ({labels.get(p, '?')}) statements ({len(stmts)}):{qual_note}"]
    for d in list(stmts.values())[:_MAX_ROWS]:
        quals = ", ".join(f"pq:{_lbl(qpid)}={_lbl(qv)}" for qpid, qv in d["quals"]) or "(no qualifiers)"
        out.append(f"  value {_lbl(d['val'])}  [{quals}]")
    return "\n".join(out)


@mcp.tool
@_budgeted
def SelectExtreme(
    value_property: str,
    context: Context,
    mode: str = "max",
    concept: str = "",
    by_count: bool = False,
    filter_property: str = "",
    filter_value: str = "",
) -> str:
    """Return ALL entities tied for the extreme (max/min) of a property — superlatives done right.

    Writing a superlative by hand as `ORDER BY DESC(?v) LIMIT 1` silently drops ties; the
    gold answer keeps every tied entity. This builds the correct MAX/MIN-subquery for you.
    Args:
      value_property: the P-id whose value to extremize (e.g. P2046 area, P1082 population).
      mode: "max" (default) for largest/highest/longest/most, "min" for smallest/least.
      concept: optional Q-id to restrict subjects to its instances (wdt:P31/wdt:P279*).
      by_count: True for "most/fewest <things>" — extremize by the COUNT of value_property
        per subject instead of by its value.
      filter_property/filter_value: optional extra subject constraint (P-id = Q-id),
        e.g. P57=Q2001 to restrict to films directed by Kubrick.
    Returns the tied entity ids plus the SPARQL used, so you can submit that as the answer.
    """
    vp = value_property.strip()
    if not _valid_id(vp, "P"):
        return f"ERROR: '{vp}' is not a valid property id (expected P<number>)."
    agg = "MIN" if str(mode).lower().startswith("min") else "MAX"
    restr = ""
    if concept.strip() and _valid_id(concept.strip(), "Q"):
        restr += f"?s wdt:P31/wdt:P279* wd:{concept.strip()} . "
    if (
        filter_property.strip() and filter_value.strip()
        and _valid_id(filter_property.strip(), "P") and _valid_id(filter_value.strip(), "Q")
    ):
        restr += f"?s wdt:{filter_property.strip()} wd:{filter_value.strip()} . "
    if by_count:
        inner = f"SELECT ?s (COUNT(DISTINCT ?x) AS ?v) WHERE {{ {restr}?s wdt:{vp} ?x . }} GROUP BY ?s"
    else:
        inner = f"SELECT ?s ?v WHERE {{ {restr}?s wdt:{vp} ?v . }}"
    query = (
        f"SELECT DISTINCT ?s WHERE {{ "
        f"{{ SELECT ({agg}(?v) AS ?m) WHERE {{ {{ {inner} }} }} }} "
        f"{{ {inner} }} FILTER(?v = ?m) }}"
    )
    r = execute(query, timeout=_TOOL_SPARQL_TIMEOUT, retries=1)
    if not r.ok or r.json is None:
        err = (r.error or "query failed")[:200]
        if "timeout" in err.lower():
            return (
                "TIMEOUT: the extreme query was too broad. Add a `concept` restriction or a "
                "filter_property/filter_value to shrink the candidate set, then retry."
            )
        return f"ERROR: {err}"
    rows = r.json.get("results", {}).get("bindings", [])
    ids = [_short(b["s"]["value"]) for b in rows if "s" in b]
    if not ids:
        return (
            "0 rows. Check the value_property exists on these subjects (try GetEntityProperties), "
            "or relax the concept/filter."
        )
    # Record into the journal so synthesis can reuse this exact (tie-safe) query as the answer.
    key = f"sparql_result_{len(session_journal.found_values) + 1}"
    session_journal.found_values[key] = {
        "query": query, "vars": ["s"],
        "rows": [{"s": i} for i in ids[:_MAX_ROWS]], "result_count": len(ids),
    }
    labels = _labels_for(ids[:_MAX_ROWS])
    shown = "  ".join(f"{i}({labels[i]})" if labels.get(i) else i for i in ids[:_MAX_ROWS])
    return (
        f"{agg} of {vp}: {len(ids)} entity(ies) tied for the extreme:\n  {shown}\n"
        f"Use this query as your answer (it keeps all ties):\n{query}"
    )


@mcp.tool
@_budgeted
def GetLabels(ids: str, context: Context) -> str:
    """Resolve a comma-separated list of Q/P ids to English labels."""
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    valid = [x for x in id_list if _valid_id(x, "Q") or _valid_id(x, "P")]
    if not valid:
        return "ERROR: provide a comma-separated list of Q/P ids."
    labels = _labels_for(valid)
    return "\n".join(f"  {i}: {labels.get(i, '?')}" for i in valid)


def _labels_for(ids: list[str]) -> dict[str, str]:
    """Batch-resolve Q/P ids to English labels via a single VALUES query."""
    ids = [i for i in dict.fromkeys(ids) if _valid_id(i, "Q") or _valid_id(i, "P")]
    if not ids:
        return {}
    values = " ".join(f"wd:{i}" for i in ids)
    q = (
        "SELECT ?e ?l WHERE { "
        f"VALUES ?e {{ {values} }} "
        '?e rdfs:label ?l . FILTER(LANG(?l)="en") }'
    )
    r = execute(q, timeout=60)
    out: dict[str, str] = {}
    if r.ok and r.json is not None:
        for b in r.json.get("results", {}).get("bindings", []):
            out[_short(b["e"]["value"])] = b.get("l", {}).get("value", "")
    return out


# --------------------------------------------------------------------------- #
# Entity/property linking (without-mentions track) — gated behind config
# --------------------------------------------------------------------------- #
def _wbsearch(term: str, kind: str, language: str) -> list[dict]:
    """Query the Wikidata ``wbsearchentities`` API; return the raw candidate list.

    Retries with backoff: the anonymous API rate-limits by IP (HTTP 429 /
    "too many requests"), and parallel benchmark shards can trip it. The
    backoff both rides out the window and self-throttles our request rate.
    """
    params = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "search": term,
            "language": language,
            "uselang": language,
            "type": kind,
            "format": "json",
            "limit": _SEARCH_LIMIT,
        }
    )
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0, 2, 6)):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(
                f"{_SEARCH_API}?{params}", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            return data.get("search", []) or []
        except Exception as exc:  # HTTPError(429), URLError, timeout, bad JSON
            last_exc = exc
    raise last_exc if last_exc else RuntimeError("wbsearchentities failed")


def SearchEntities(
    query: str,
    context: Context,
    type: Literal["item", "property"] = "item",
    language: str = "en",
) -> str:
    """Link a NAME to Wikidata ids via the Wikidata search API (without-mentions track).

    Use this FIRST when you were given no Q/P ids and must derive them from the
    question text. ``type="item"`` finds entities (people, places, works, classes);
    ``type="property"`` finds relations. ``language`` should match the question
    ("en"/"es"). Returns ranked candidates with label + description; pick the id
    whose description matches the question's intended sense (e.g. "Mercury" -> the
    planet, not the element), then confirm with GetEntityProperties / RunSPARQL
    before committing.
    """
    term = query.strip()
    if not term:
        return "ERROR: provide a name to search for."
    kind = "property" if str(type).lower().startswith("prop") else "item"
    lang = (language or "en").strip() or "en"
    try:
        hits = _wbsearch(term, kind, lang)
    except Exception as exc:
        # NB: the `type` PARAMETER shadows the builtin in this scope — use
        # __class__ (a bare type(exc) here crashes with "'str' object is not
        # callable" and masks the real error).
        return f"ERROR: entity search failed: {exc.__class__.__name__}: {exc}"[:300]
    if not hits and lang != "en":
        # A non-English label may be missing; fall back to an English search.
        try:
            hits = _wbsearch(term, kind, "en")
        except Exception:
            hits = []
    if not hits:
        return (
            f"No {kind} candidates found for '{term}'. Try a shorter form, a synonym, "
            "or search for a related class instead."
        )
    out = [f"Candidates for '{term}' ({kind}):"]
    for h in hits:
        cid = h.get("id", "?")
        label = h.get("label", "")
        desc = h.get("description", "")
        out.append(f"  {cid}  {label}" + (f" — {desc}" if desc else ""))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Property linking via embedded dense/hybrid retrieval (without-mentions track)
# --------------------------------------------------------------------------- #
# Unlike SearchEntities (live Wikidata API), relations are linked against a LOCAL
# Qdrant collection of all ~13k properties (label+description+aliases), embedded with
# the project's shared model. Semantic matching beats the live API's label-only match
# for oddly-phrased relations, and it costs no endpoint round-trip per search.
_WIKIDATA_PROPERTIES_COLLECTION = "wikidata-properties"
_qdrant_client = None
_embed_client = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        from ama_kbqa.config import get_qdrant_host, get_qdrant_port

        _qdrant_client = QdrantClient(host=get_qdrant_host(), port=get_qdrant_port(), timeout=30)
    return _qdrant_client


def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from ama_kbqa.config import get_embedding_client

        _embed_client = get_embedding_client()
    return _embed_client


def SearchProperties(query: str, context: Context, limit: int = 8) -> str:
    """Semantic search over ALL Wikidata properties to find the P-id for a relation.

    The relation linker for the without-mentions track. Prefer this over guessing
    property ids: it matches MEANING (label + description + aliases), so a relation
    phrased differently than its label still resolves ("who governs" -> P6 head of
    government; "cause of death" -> P509). Returns ranked candidates (P-id, label,
    description); confirm the winner with GetPropertyInfo or RunSPARQL before committing.
    """
    term = query.strip()
    if not term:
        return "ERROR: provide a relation phrase to search for."
    try:
        from ama_kbqa import retrieval

        qc = _get_qdrant()
        if not qc.collection_exists(_WIKIDATA_PROPERTIES_COLLECTION):
            return (
                f"ERROR: the '{_WIKIDATA_PROPERTIES_COLLECTION}' index is not populated. "
                "Use SearchEntities(type=property) as a fallback."
            )
        vec = retrieval.embed_query(_get_embed_client(), term)
        hits = retrieval.search(
            qc,
            _WIKIDATA_PROPERTIES_COLLECTION,
            query_text=term,
            query_vector=vec,
            # allow_rerank=False: dense+BM25 fusion is enough for linking and avoids the
            # optional sentence-transformers dep (install `uv sync --extra rerank` for a
            # cross-encoder boost later).
            params=retrieval.build_retrieval_params(
                limit=max(1, min(int(limit), 15)), score_threshold=None, allow_rerank=False
            ),
        )
    except Exception as exc:
        return f"ERROR: property search failed: {type(exc).__name__}: {exc}"[:300]
    if not hits:
        return f"No property candidates for '{term}'. Try a synonym or simpler phrasing."
    out = [f"Property candidates for '{term}':"]
    for h in hits:
        p = h.payload or {}
        pid, label, desc = p.get("pid", "?"), p.get("label", ""), p.get("description", "")
        out.append(f"  {pid}  {label}" + (f" — {desc[:70]}" if desc else "") + f"  (score {h.score:.2f})")
    return "\n".join(out)


# Register only when the config flag is set, so these are absent (and invisible to
# the agent's tool catalog) on with-mentions runs.
if _ENTITY_SEARCH_ENABLED:
    mcp.tool(_budgeted(SearchEntities))
    mcp.tool(_budgeted(SearchProperties))


if __name__ == "__main__":
    mcp.run(transport="stdio")

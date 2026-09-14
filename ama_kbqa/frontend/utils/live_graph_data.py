"""Normalise agent output into the subgraph the live graph panel draws.

Why this module exists: the demo has two partial views of what an agent
discovered, and neither alone is enough for a useful picture.

1. The journal (``JournalState``) knows which entities were visited and which
   values were found, but several tools drop their neighbours on the floor
   (``GetNodeSummary`` journals attributes only, ``FindNode`` journals
   nothing), and the verified-fact entries come in at least seven different
   shapes written by different tools.
2. The raw tool results carry the missing neighbours and the search
   candidates, but only for the calls the agent actually made.

Building the graph from both sources, merged by node id, is what makes the
panel show search candidates, the chosen entity, its neighbours and its
literal values at once. Both the live view (deltas parsed off the worker
thread) and the frozen view (a completed trace) go through the same
normalisers here so they can never disagree.

Deliberately Streamlit-free and side-effect-free: every builder is a pure
function and every builder is tolerant. Agent output is LLM-driven and
server-shaped, so a malformed fact, a non-JSON tool result or a ``None``
where a string was expected is expected traffic, not an error. Bad input is
skipped; nothing here raises.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Optional


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
# All caps are per PRD §3.5 / §5.1. They exist because a single tool call can
# legitimately return hundreds of rows (a SPARQL query, a hub entity's
# relations) and the panel is a glanceable overview, not a KG browser.

MAX_NODES = 300
MAX_EDGES = 600
MAX_HIGHLIGHT = 50

LABEL_CHARS = 40
EDGE_LABEL_CHARS = 30
TOOLTIP_VALUE_CHARS = 120
MAX_TOOLTIP_LINES = 12

MAX_LITERALS_PER_ATTR = 8
MAX_OBJECTS_PER_FACT = 5
MAX_CANDIDATES = 10
MAX_RELATION_TARGETS = 10
MAX_ATTR_VALUES = 5
MAX_SPARQL_NODES = 25


NodeKind = Literal["entity", "literal", "candidate"]

# Ranking used when the same node id arrives from two sources: a node the
# agent actually visited outranks one that was only a search candidate.
_KIND_RANK: dict[str, int] = {"candidate": 0, "literal": 1, "entity": 2}

# Labels the servers write while a real label is still being resolved. Such a
# label carries no information, so it must never beat a real one on merge.
_PENDING_LABELS = frozenset({"", "(resolving label)", "(pending)", "(unknown)"})

_ID_PATTERN = re.compile(r"^[QPLRC]\d+$")
_URI_PREFIXES = ("http://", "https://")
_CURIE_PREFIXES = ("orkgr:", "orkgp:", "orkgc:")

# Verified-fact `direction` values that mean "the stored triple points at the
# subject, not away from it". The servers write three different spellings:
# "backward" (GetRelationDetails' SPARQL BIND), "inverse_only"
# (GetRelationBetween) and "reverse" (rendered by _format_verified_fact).
_REVERSED_DIRECTIONS = frozenset({"reverse", "inverse_only", "backward"})

# Tools whose results are worth extracting nodes/edges from. Anything not
# listed returns an empty graph, so an unrelated tool (VerifyFact,
# GetJournalSummary, ...) costs nothing.
GRAPH_TOOLS: frozenset[str] = frozenset({
    # KQAPro search
    "FindNode",
    "LookupEntityByName",
    "FindByAttribute",
    # SciQA search
    "FindResource",
    "LookupResourceByLabel",
    # KQAPro exploration
    "GetNodeSummary",
    "GetRelationDetails",
    # SciQA exploration
    "GetRelationTargets",
    "GetResourceDetails",
    # Raw SPARQL (nodes only, no edges)
    "RunSPARQL",
    "RunORKGSPARQL",
})

_SEARCH_TOOLS = frozenset({
    "FindNode", "LookupEntityByName", "FindByAttribute",
    "FindResource", "LookupResourceByLabel",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphNode:
    """One node of the explored subgraph.

    `id` is the KG identifier for entities and candidates ("Q937", "R12345")
    and a synthetic ``lit:<entity>:<attr>:<n>`` key for literals, so two
    entities that happen to share a value do not collapse into one hub node.
    """

    id: str
    label: str
    kind: NodeKind = "entity"
    source: str = ""
    title: str = ""
    order: int = 0


@dataclass(frozen=True)
class GraphEdge:
    """One predicate edge. `id` is deterministic so re-deriving the same
    graph from the same input never duplicates an edge."""

    id: str
    src: str
    dst: str
    label: str = ""
    title: str = ""


@dataclass
class GraphData:
    """A subgraph, insertion-ordered on both dicts (plain dicts keep order)."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    truncated_nodes: int = 0

    def merge(self, other: "GraphData") -> "GraphData":
        """Union by id, self's ordering first.

        Node conflicts resolve field by field (see `_merge_nodes`): the
        visited entity beats the search candidate, a real label beats an
        id-as-label, a known source beats an unknown one. Returns a new
        object; neither input is mutated.
        """
        merged = GraphData(
            nodes=dict(self.nodes),
            edges=dict(self.edges),
            truncated_nodes=self.truncated_nodes + getattr(other, "truncated_nodes", 0),
        )
        if not isinstance(other, GraphData):
            return merged
        for node in other.nodes.values():
            _put_node(merged, node)
        for edge in other.edges.values():
            merged.edges.setdefault(edge.id, edge)
        return merged

    def stats(self) -> dict:
        """Counts for the panel's metric row."""
        counts = {"entities": 0, "literals": 0, "candidates": 0}
        for node in self.nodes.values():
            if node.kind == "literal":
                counts["literals"] += 1
            elif node.kind == "candidate":
                counts["candidates"] += 1
            else:
                counts["entities"] += 1
        counts["edges"] = len(self.edges)
        return counts

    def is_empty(self) -> bool:
        return not self.nodes and not self.edges


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _looks_like_id(value: Any) -> bool:
    """Heuristic: does this string name a KG node rather than a value?

    Covers Wikidata/KQAPro style ids (Q/P/L), ORKG ids (R/P/C), http(s) URIs
    and the ORKG CURIE prefixes. Extended from the original in graph_html.py,
    which only knew Q/P/L and URIs.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if _ID_PATTERN.match(text):
        return True
    if text.startswith(_URI_PREFIXES):
        return True
    return text.startswith(_CURIE_PREFIXES)


def _split_id(value: Any) -> tuple[str, str]:
    """Return ``(node_id, full_form)`` for an id-shaped string.

    URIs are shortened to their last path segment so the graph shows "Q937"
    rather than the full http URI; the full form stays for the tooltip.
    """
    text = str(value).strip()
    if text.startswith(_URI_PREFIXES):
        tail = text.rstrip("/").rsplit("/", 1)[-1]
        if "#" in tail:
            tail = tail.rsplit("#", 1)[-1]
        return (tail or text), text
    for prefix in _CURIE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):] or text, text
    return text, text


def _is_pending_label(label: Any, node_id: str) -> bool:
    """True when `label` carries no information beyond the id itself."""
    if not isinstance(label, str):
        return True
    text = label.strip()
    if text in _PENDING_LABELS or text == node_id:
        return True
    # The servers write placeholders such as "(resolving label)" and check
    # for them with startswith("("); mirror that so any parenthesised
    # placeholder counts as pending.
    return text.startswith("(") and text.endswith(")")


def _label_rank(label: str, node_id: str) -> int:
    return 0 if _is_pending_label(label, node_id) else len(label)


_PENDING_LINE = "<i>label pending</i>"


def _title_lines(title: str) -> list[str]:
    return [line for line in title.split("<br>") if line]


def _body_lines(title: str) -> list[str]:
    """Tooltip lines that are not part of the identity header.

    The header (bold label, id, "label pending") is recomputed on every merge
    from the winning label, so carrying the old one along would leave a node
    that gained a real label still claiming its label is pending.
    """
    return [
        line for line in _title_lines(title)
        if not line.startswith("<b>") and line != _PENDING_LINE
    ]


def _make_title(lines: Iterable[str]) -> str:
    seen: list[str] = []
    for line in lines:
        if line and line not in seen:
            seen.append(line)
    return "<br>".join(seen[:MAX_TOOLTIP_LINES])


def _compose_entity_title(node_id: str, label: str, body: Iterable[str]) -> str:
    lines = [f"<b>{_esc(label)}</b>"]
    if label == node_id:
        lines.append(_PENDING_LINE)
    else:
        lines.append(_esc(node_id))
    lines.extend(body)
    return _make_title(lines)


def _merge_nodes(existing: GraphNode, incoming: GraphNode) -> GraphNode:
    """Field-wise precedence for two sightings of the same node id."""
    kind = existing.kind
    if _KIND_RANK.get(incoming.kind, 0) > _KIND_RANK.get(existing.kind, 0):
        kind = incoming.kind
    label = existing.label
    if _label_rank(incoming.label, incoming.id) > _label_rank(existing.label, existing.id):
        label = incoming.label
    body = _body_lines(existing.title) + _body_lines(incoming.title)
    title = (
        _make_title(body) if kind == "literal"
        else _compose_entity_title(existing.id, label, body)
    )
    return GraphNode(
        id=existing.id,
        label=label,
        kind=kind,
        source=existing.source or incoming.source,
        title=title,
        order=existing.order,
    )


def _put_node(graph: GraphData, node: GraphNode) -> None:
    """Insert or merge `node`. Insertion order is assigned here, so a node's
    `order` is stable for the lifetime of the graph it first appeared in."""
    if not node.id:
        return
    existing = graph.nodes.get(node.id)
    if existing is None:
        graph.nodes[node.id] = replace(node, order=len(graph.nodes))
        return
    graph.nodes[node.id] = _merge_nodes(existing, node)


def _put_edge(
    graph: GraphData,
    src: str,
    dst: str,
    predicate: str,
    *,
    provenance: str = "",
    reverse: bool = False,
) -> None:
    if not src or not dst or src == dst:
        return
    if reverse:
        src, dst = dst, src
    predicate = str(predicate or "")
    edge_id = f"{src}|{predicate}|{dst}"
    if edge_id in graph.edges:
        return
    title = _esc(predicate) if predicate else ""
    if provenance:
        title = f"{title}<br><i>{_esc(provenance)}</i>" if title else f"<i>{_esc(provenance)}</i>"
    graph.edges[edge_id] = GraphEdge(
        id=edge_id,
        src=src,
        dst=dst,
        label=_truncate(predicate, EDGE_LABEL_CHARS),
        title=title,
    )


def _entity_node(
    raw_id: Any,
    label: Any = None,
    *,
    source: str = "",
    kind: NodeKind = "entity",
    extra_lines: Iterable[str] = (),
) -> Optional[GraphNode]:
    """Build an entity/candidate node, or None when the id is unusable."""
    node_id, full = _split_id(raw_id)
    if not node_id:
        return None
    pending = _is_pending_label(label, node_id)
    full_label = node_id if pending else str(label).strip()
    display = _truncate(full_label, LABEL_CHARS)
    # The graph label is truncated to stay readable; the tooltip keeps the
    # full text (PRD 3.5), and the full URI when the id came from one.
    body = [] if display == full_label else [_esc(_truncate(full_label, TOOLTIP_VALUE_CHARS))]
    if full != node_id:
        body.append(_esc(full))
    body.extend(extra_lines)
    return GraphNode(
        id=node_id,
        label=display,
        kind=kind,
        source=source,
        title=_compose_entity_title(node_id, display, body),
    )


def _literal_node(
    literal_id: str,
    value: Any,
    *,
    source: str = "",
    unit: Any = None,
    extra_lines: Iterable[str] = (),
) -> Optional[GraphNode]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    if unit:
        text = f"{text} {unit}".strip()
    lines = [_esc(_truncate(text, TOOLTIP_VALUE_CHARS)), *extra_lines]
    return GraphNode(
        id=literal_id,
        label=_truncate(text, LABEL_CHARS),
        kind="literal",
        source=source,
        title=_make_title(lines),
    )


def source_for_agent(agent_or_name: Any) -> str:
    """Map an agent object, agent name or KG name to a graph source.

    Accepts every spelling the codebase uses for the same specialist:
    ``kqapro_agent`` (delegate span attribute), ``KQAProAgent`` (class),
    ``KQAPro`` (picker entry and ``JournalState.kg_name``), and the ORKG
    aliases on the SciQA side. Anything else is unknown ("") rather than
    guessed, so an orchestrator-level or unrecognised caller does not paint
    its nodes with the wrong graph's colour.
    """
    if agent_or_name is None:
        return ""
    if isinstance(agent_or_name, str):
        text = agent_or_name
    else:
        text = (
            getattr(agent_or_name, "name", None)
            or getattr(agent_or_name, "kg_name", None)
            or type(agent_or_name).__name__
        )
    text = str(text).lower()
    if "kqapro" in text or "kqa_pro" in text:
        return "kqapro"
    if "sciqa" in text or "orkg" in text:
        return "sciqa"
    return ""


# ---------------------------------------------------------------------------
# Journal state
# ---------------------------------------------------------------------------

def graph_from_journal_state(state: dict, *, source: str) -> GraphData:
    """Build the subgraph the journal alone can support.

    `visited_nodes` gives entities, `found_values` gives literal leaves (plus
    entity edges where a value is really an id), and `verified_facts` gives
    edges once the seven-odd shapes the servers write are normalised. Unknown
    shapes are dropped rather than rendered as "?" placeholders.
    """
    graph = GraphData()
    if not isinstance(state, dict):
        return graph

    visited = state.get("visited_nodes")
    if isinstance(visited, dict):
        for node_id, label in visited.items():
            node = _entity_node(node_id, label, source=source)
            if node is not None:
                _put_node(graph, node)

    found_values = state.get("found_values")
    if isinstance(found_values, dict):
        for entity_id, attrs in found_values.items():
            _absorb_found_values(graph, entity_id, attrs, source=source)

    facts = state.get("verified_facts")
    if isinstance(facts, list):
        for fact in facts:
            _absorb_verified_fact(graph, fact, source=source)

    return graph


def _absorb_found_values(
    graph: GraphData,
    entity_id: Any,
    attrs: Any,
    *,
    source: str,
) -> None:
    """One `found_values[entity_id]` bucket.

    RunSPARQL parks its result rows under synthetic keys such as
    "sparql_result_1", which are not entities at all; those are skipped by
    the id heuristic instead of being drawn as a node.
    """
    if not isinstance(attrs, dict):
        return
    node_id, _full = _split_id(entity_id)
    if not _looks_like_id(node_id):
        return
    subject = _entity_node(node_id, None, source=source)
    if subject is None:
        return
    _put_node(graph, subject)

    for attr, value in attrs.items():
        attr_name = str(attr)
        if attr_name.startswith("relation_to_"):
            # GetRelationBetween stores the *predicate name* under
            # "relation_to_<object id>" (kqapro_server.py:5072). That is an
            # edge label, not a value; the edge itself arrives through the
            # matching verified_facts entry, so drawing this as a literal
            # leaf would show a predicate name hanging off the entity.
            continue
        if attr_name.startswith("_"):
            # SciQA's GetResourceDetails stores the RDF types under "_type".
            # Type information is tooltip material, not a node.
            _put_node(graph, replace(
                subject,
                title=_make_title([f"<i>{_esc(attr_name)}</i>: {_esc(_short_repr(value))}"]),
            ))
            continue
        items = value if isinstance(value, list) else [value]
        for index, item in enumerate(items[:MAX_LITERALS_PER_ATTR]):
            _absorb_value_item(
                graph, node_id, attr_name, index, item, source=source,
            )


def _absorb_value_item(
    graph: GraphData,
    subject_id: str,
    attr: str,
    index: int,
    item: Any,
    *,
    source: str,
) -> None:
    """One value of one attribute: either a neighbour entity or a literal."""
    related: Any = None
    text: Any = item
    unit: Any = None
    extra: list[str] = []

    if isinstance(item, dict):
        for key in ("related_id", "entity_id", "id"):
            candidate = item.get(key)
            if candidate and _looks_like_id(str(candidate)):
                related = candidate
                break
        text = item.get("value", item.get("label"))
        unit = item.get("unit")
        qualifier = item.get("matched_qualifier") or item.get("qualifier")
        if qualifier:
            extra.append(f"<i>qualifier</i>: {_esc(qualifier)}")
    elif isinstance(item, (list, tuple, set)):
        # A nested collection carries no attribute semantics we can trust.
        return

    if related is None and _looks_like_id(str(text) if text is not None else ""):
        related = text

    if related is not None:
        label = text if (text is not None and not _looks_like_id(str(text))) else None
        node = _entity_node(related, label, source=source, extra_lines=extra)
        if node is None:
            return
        _put_node(graph, node)
        _put_edge(graph, subject_id, node.id, attr, provenance="journal: found_values")
        return

    literal = _literal_node(
        f"lit:{subject_id}:{attr}:{index}",
        text,
        source=source,
        unit=unit,
        extra_lines=extra,
    )
    if literal is None:
        return
    _put_node(graph, literal)
    _put_edge(graph, subject_id, literal.id, attr, provenance="journal: found_values")


def _absorb_verified_fact(graph: GraphData, fact: Any, *, source: str) -> None:
    """Normalise one `verified_facts` entry into zero or more edges.

    The shape checks run in the same order as the server's own
    `_format_verified_fact` (kqapro_server.py:1804), so a fact is read the
    same way here and in `GetJournalSummary`.
    """
    if not isinstance(fact, dict):
        return
    provenance = f"journal: {fact.get('source')}" if fact.get("source") else "journal"

    fact_type = fact.get("type")
    if fact_type:
        _absorb_qualifier_fact(graph, fact, fact_type, source=source, provenance=provenance)
        return

    subject_raw = fact.get("subject") or fact.get("node")
    subject_id = ""
    if subject_raw:
        subject_node = _entity_node(subject_raw, None, source=source)
        if subject_node is not None:
            _put_node(graph, subject_node)
            subject_id = subject_node.id
    if not subject_id:
        return

    # Shape: subject -relation-> related_id (GetRelationDetails/GetRelationBetween)
    if "relation" in fact and "related_id" in fact:
        related = fact.get("related_id")
        if not related:
            return
        node = _entity_node(related, None, source=source)
        if node is None:
            return
        _put_node(graph, node)
        _put_edge(
            graph, subject_id, node.id, fact.get("relation") or "",
            provenance=provenance,
            reverse=str(fact.get("direction") or "") in _REVERSED_DIRECTIONS,
        )
        return

    # Shape: subject.attribute = value [unit] (GetAttributeDetails)
    if "attribute" in fact and "value" in fact:
        attr = str(fact.get("attribute") or "")
        literal = _literal_node(
            f"lit:{subject_id}:{attr}:0",
            fact.get("value"),
            source=source,
            unit=fact.get("unit"),
        )
        if literal is None:
            return
        _put_node(graph, literal)
        _put_edge(graph, subject_id, literal.id, attr, provenance=provenance)
        return

    # Shape: subject -predicate-> objects[] (ExploreNeighborhood)
    if "predicate" in fact and "objects" in fact:
        objects = fact.get("objects")
        if not isinstance(objects, list):
            return
        predicate = str(fact.get("predicate") or "")
        for index, obj in enumerate(objects[:MAX_OBJECTS_PER_FACT]):
            _absorb_value_item(
                graph, subject_id, predicate, index, obj, source=source,
            )
        return

    # Shape: subject -predicate-> object (the classic triple)
    if "predicate" in fact and "object" in fact:
        predicate = str(fact.get("predicate") or "")
        _absorb_value_item(graph, subject_id, predicate, 0, fact.get("object"), source=source)
        return

    # Everything else (free-text "fact"/"raw" entries, RunSPARQL summaries)
    # carries no reliable structure: ignored on purpose.


def _absorb_qualifier_fact(
    graph: GraphData,
    fact: dict,
    fact_type: Any,
    *,
    source: str,
    provenance: str,
) -> None:
    """The `type`-tagged qualifier entries.

    Only `relation_qualifiers` names a real statement (node -relation->
    target) and becomes an edge. `edge_qualifiers` and `qualifier_value`
    describe an existing statement, so they add a tooltip line to the subject
    and nothing else.
    """
    node_raw = fact.get("node") or fact.get("subject")
    if not node_raw:
        return
    subject = _entity_node(node_raw, None, source=source)
    if subject is None:
        return
    quals = fact.get("qualifiers")
    qual_names = ", ".join(str(q) for q in quals) if isinstance(quals, list) else ""
    if not qual_names:
        qual_names = str(fact.get("matched_qualifier") or fact.get("qualifier") or "")
    if qual_names:
        subject = replace(subject, title=_make_title([
            *_title_lines(subject.title),
            f"<i>{_esc(fact_type)}</i>: {_esc(qual_names)}",
        ]))
    _put_node(graph, subject)

    if fact_type != "relation_qualifiers":
        return
    target = fact.get("target")
    if not target or not _looks_like_id(str(target)):
        return
    node = _entity_node(target, fact.get("target_label"), source=source)
    if node is None:
        return
    _put_node(graph, node)
    _put_edge(graph, subject.id, node.id, fact.get("relation") or "", provenance=provenance)


def _short_repr(value: Any, limit: int = 80) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value[:3])[:limit]
    if isinstance(value, dict):
        return "{" + ", ".join(list(value.keys())[:3]) + "}"
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

def graph_from_tool_result(
    tool_name: str,
    arguments: dict,
    result_text: str,
    *,
    source: str,
) -> GraphData:
    """Extract nodes/edges from one allow-listed tool call.

    `result_text` is the tool's raw string result as recorded on the
    `tool_call` span's payload. Anything that is not allow-listed, not JSON,
    or not shaped as the tool's response model yields an empty graph.
    """
    graph = GraphData()
    if tool_name not in GRAPH_TOOLS:
        return graph
    if not isinstance(result_text, str) or not result_text.strip():
        return graph
    try:
        payload = json.loads(result_text)
    except (ValueError, TypeError):
        return graph
    if not isinstance(payload, dict):
        return graph
    if not isinstance(arguments, dict):
        arguments = {}

    try:
        if tool_name in _SEARCH_TOOLS:
            _extract_search_matches(graph, payload, source=source)
        elif tool_name == "GetNodeSummary":
            _extract_node_summary(graph, payload, source=source)
        elif tool_name == "GetRelationDetails":
            _extract_relation_details(graph, payload, arguments, source=source)
        elif tool_name == "GetRelationTargets":
            _extract_relation_targets(graph, payload, arguments, source=source)
        elif tool_name == "GetResourceDetails":
            _extract_resource_details(graph, payload, source=source)
        elif tool_name in ("RunSPARQL", "RunORKGSPARQL"):
            _extract_sparql_bindings(graph, payload, source=source)
    except Exception:  # noqa: BLE001 - a malformed result must never break the panel
        return graph
    return graph


def _extract_search_matches(graph: GraphData, payload: dict, *, source: str) -> None:
    """SearchResponse (kqapro_server.py:146, sciqa_server.py:176).

    Matches are drawn as *candidates*: the agent has seen them, but has not
    committed to any of them yet. The journal upgrades the one it visits.
    """
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return
    for match in matches[:MAX_CANDIDATES]:
        if not isinstance(match, dict):
            continue
        node = _entity_node(
            match.get("original_id"),
            match.get("name"),
            source=source,
            kind="candidate",
            extra_lines=(
                [f"<i>score</i>: {_esc(match.get('relevance_score'))}"]
                if match.get("relevance_score") is not None else []
            ),
        )
        if node is not None:
            _put_node(graph, node)


def _extract_node_summary(graph: GraphData, payload: dict, *, source: str) -> None:
    """GetNodeSummary's plain dict (kqapro_server.py:2451).

    This is the tool the journal loses the most from: it journals the
    attributes and drops the whole `relations` dict, so without this
    extractor a journal-only graph shows a visited entity with literal leaves
    and no neighbours at all.
    """
    node_id_raw = payload.get("node_id")
    if not node_id_raw:
        return
    base = _entity_node(node_id_raw, payload.get("name"), source=source)
    if base is None:
        return
    _put_node(graph, base)

    relations = payload.get("relations")
    if isinstance(relations, dict):
        for predicate, targets in relations.items():
            if not isinstance(targets, list):
                continue
            for target in targets[:MAX_RELATION_TARGETS]:
                node = _entity_node(target, None, source=source)
                if node is None or not _looks_like_id(node.id):
                    continue
                _put_node(graph, node)
                _put_edge(
                    graph, base.id, node.id, str(predicate),
                    provenance="GetNodeSummary",
                )

    attributes = payload.get("attributes")
    if isinstance(attributes, dict):
        for attr, values in attributes.items():
            if not isinstance(values, list):
                values = [values]
            for index, value in enumerate(values[:MAX_ATTR_VALUES]):
                literal = _literal_node(
                    f"lit:{base.id}:{attr}:{index}", value, source=source,
                )
                if literal is None:
                    continue
                _put_node(graph, literal)
                _put_edge(
                    graph, base.id, literal.id, str(attr),
                    provenance="GetNodeSummary",
                )


def _extract_relation_details(
    graph: GraphData, payload: dict, arguments: dict, *, source: str,
) -> None:
    """RelationDetailsResponse (kqapro_server.py:201).

    The triples this tool returns are NOT (subject, predicate, object)
    triples: each entry is `{related_id, related_uri, direction}` with the
    subject implied by `node_id` and the predicate by `relation_name`. The
    endpoint keys are read defensively so a differently-shaped triple (from
    another tool reusing this extractor) still lands.
    """
    base_raw = payload.get("node_id") or arguments.get("base_node_id")
    predicate = payload.get("relation_name") or arguments.get("relation_name") or ""
    triples = payload.get("triples")
    if not base_raw or not isinstance(triples, list):
        return
    base = _entity_node(base_raw, None, source=source)
    if base is None:
        return
    _put_node(graph, base)

    for triple in triples[:MAX_RELATION_TARGETS]:
        if not isinstance(triple, dict):
            continue
        target_raw = None
        for key in ("related_id", "object", "target", "entity_id", "id", "related_uri"):
            value = triple.get(key)
            if value and _looks_like_id(str(value)):
                target_raw = value
                break
        if target_raw is None:
            continue
        node = _entity_node(target_raw, triple.get("label"), source=source)
        if node is None:
            continue
        _put_node(graph, node)
        subject_raw = triple.get("subject")
        subject_id = base.id
        if subject_raw and _looks_like_id(str(subject_raw)):
            subject_node = _entity_node(subject_raw, None, source=source)
            if subject_node is not None:
                _put_node(graph, subject_node)
                subject_id = subject_node.id
        _put_edge(
            graph, subject_id, node.id,
            str(triple.get("predicate") or predicate),
            provenance="GetRelationDetails",
            reverse=str(triple.get("direction") or "") in _REVERSED_DIRECTIONS,
        )


def _extract_relation_targets(
    graph: GraphData, payload: dict, arguments: dict, *, source: str,
) -> None:
    """GetRelationTargets' JSON dict (sciqa_server.py:1864).

    `targets[]` entries are either `{id, label?}` (a resource) or `{value}`
    (a literal); both shapes appear in the same list.
    """
    base_raw = payload.get("resource_id") or arguments.get("resource_id")
    predicate = str(payload.get("predicate") or arguments.get("predicate") or "")
    targets = payload.get("targets")
    if not base_raw or not isinstance(targets, list):
        return
    base = _entity_node(base_raw, None, source=source)
    if base is None:
        return
    _put_node(graph, base)
    for index, target in enumerate(targets[:MAX_RELATION_TARGETS]):
        _absorb_target_entry(graph, base.id, predicate, index, target, source=source,
                             provenance="GetRelationTargets")


def _extract_resource_details(graph: GraphData, payload: dict, *, source: str) -> None:
    """GetResourceDetails' JSON dict (sciqa_server.py:1749).

    `relations` is `{predicate: [{id, label} | {value}]}`; the `types` list is
    tooltip material rather than nodes.
    """
    base_raw = payload.get("resource_id")
    if not base_raw:
        return
    types = payload.get("types")
    extra = (
        [f"<i>type</i>: {_esc(_short_repr(types))}"]
        if isinstance(types, list) and types else []
    )
    base = _entity_node(base_raw, payload.get("label"), source=source, extra_lines=extra)
    if base is None:
        return
    _put_node(graph, base)

    relations = payload.get("relations")
    if not isinstance(relations, dict):
        return
    for predicate, targets in relations.items():
        if not isinstance(targets, list):
            continue
        for index, target in enumerate(targets[:MAX_RELATION_TARGETS]):
            _absorb_target_entry(
                graph, base.id, str(predicate), index, target,
                source=source, provenance="GetResourceDetails",
            )


def _absorb_target_entry(
    graph: GraphData,
    subject_id: str,
    predicate: str,
    index: int,
    target: Any,
    *,
    source: str,
    provenance: str,
) -> None:
    """One ORKG `{id, label} | {value}` target entry."""
    if isinstance(target, dict):
        target_id = target.get("id")
        if target_id:
            node = _entity_node(target_id, target.get("label"), source=source)
            if node is None:
                return
            _put_node(graph, node)
            _put_edge(graph, subject_id, node.id, predicate, provenance=provenance)
            return
        value = target.get("value")
    else:
        value = target
    if value is not None and _looks_like_id(str(value)):
        node = _entity_node(value, None, source=source)
        if node is not None:
            _put_node(graph, node)
            _put_edge(graph, subject_id, node.id, predicate, provenance=provenance)
        return
    literal = _literal_node(f"lit:{subject_id}:{predicate}:{index}", value, source=source)
    if literal is None:
        return
    _put_node(graph, literal)
    _put_edge(graph, subject_id, literal.id, predicate, provenance=provenance)


def _extract_sparql_bindings(graph: GraphData, payload: dict, *, source: str) -> None:
    """SPARQLResponse (kqapro_server.py:212, sciqa_server.py:206).

    `bindings` rows are already flattened to `{var: value_string}` by
    `compact_sparql_select_results`, so a row is scanned for id-shaped values
    and a sibling `<var>Label` binding supplies the label. Nodes only: a
    SELECT projection says nothing reliable about which variable relates to
    which, so inventing edges here would be fiction.
    """
    bindings = payload.get("bindings")
    if not isinstance(bindings, list):
        return
    added = 0
    for row in bindings:
        if not isinstance(row, dict):
            continue
        for var, value in row.items():
            if added >= MAX_SPARQL_NODES:
                return
            if var.endswith("Label") or not _looks_like_id(str(value)):
                continue
            label = row.get(f"{var}Label")
            node = _entity_node(value, label, source=source)
            if node is None or node.id in graph.nodes:
                continue
            _put_node(graph, node)
            added += 1


# ---------------------------------------------------------------------------
# Trace (frozen view)
# ---------------------------------------------------------------------------

def graph_from_trace(trace: dict) -> GraphData:
    """Rebuild the full subgraph of a completed run.

    Tool calls are attributed to the specialist that made them by walking
    `parent_span_id` up to the enclosing `delegate` span, exactly as
    `lifecycle_runner.reconstruct_orchestrator` does, so a federated run
    colours each specialist's nodes with its own graph. For journal
    snapshots, only the LAST snapshot per `source_agent` is merged: snapshots
    are cumulative, so earlier ones add nothing but work.
    """
    graph = GraphData()
    if not isinstance(trace, dict):
        return graph
    events = trace.get("events") or []
    if not isinstance(events, list):
        events = []
    default_source = source_for_agent(trace.get("agent") or "")

    span_parent: dict[str, Optional[str]] = {}
    delegate_owner: dict[str, str] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        span_id = event.get("span_id") or ""
        span_parent[span_id] = event.get("parent_span_id")
        if event.get("kind") == "delegate":
            delegate_owner[span_id] = (event.get("attributes") or {}).get("sub_agent") or ""

    def owner_of(span_id: Optional[str]) -> Optional[str]:
        current, seen = span_id, 0
        while current is not None and seen < 256:
            if current in delegate_owner:
                return delegate_owner[current]
            current = span_parent.get(current)
            seen += 1
        return None

    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "tool_call":
            continue
        attributes = event.get("attributes") or {}
        name = event.get("name") or attributes.get("tool_name") or ""
        if name not in GRAPH_TOOLS:
            continue
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        owner = owner_of(event.get("span_id") or "")
        source = source_for_agent(owner) if owner else default_source
        graph = graph.merge(graph_from_tool_result(
            name,
            payload.get("arguments") or {},
            payload.get("result") or "",
            source=source,
        ))

    snapshots = trace.get("journal_snapshots") or []
    if isinstance(snapshots, list):
        last_per_source: dict[Optional[str], dict] = {}
        for snapshot in snapshots:
            if isinstance(snapshot, dict):
                last_per_source[snapshot.get("source_agent")] = snapshot
        for tag, snapshot in last_per_source.items():
            state = snapshot.get("state")
            source = source_for_agent(tag) if tag else default_source
            if not source:
                source = source_for_agent((state or {}).get("kg_name") or "")
            graph = graph.merge(graph_from_journal_state(state or {}, source=source))

    return graph


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def answer_node_ids(graph: GraphData, answer: Optional[str]) -> set[str]:
    """Which nodes the final answer actually mentions.

    Three ways to match, cheapest first: the id verbatim, the label on word
    boundaries (case-insensitive, 3+ chars so "US" or "a" cannot light up
    half the graph), or, for literals, the value as a plain substring (a
    number in a sentence rarely sits on word boundaries).
    """
    if not answer or not isinstance(graph, GraphData):
        return set()
    text = str(answer)
    lowered = text.lower()
    matched: set[str] = set()
    for node in graph.nodes.values():
        if len(matched) >= MAX_HIGHLIGHT:
            break
        # Ids match on word boundaries too: "Q1" must not light up because
        # the answer mentions "Q123".
        if (
            node.kind != "literal"
            and node.id
            and re.search(rf"(?<!\w){re.escape(node.id)}(?!\w)", text)
        ):
            matched.add(node.id)
            continue
        label = node.label.rstrip("…").strip()
        if len(label) < 3:
            continue
        if node.kind == "literal":
            if label.lower() in lowered:
                matched.add(node.id)
            continue
        if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text, re.IGNORECASE):
            matched.add(node.id)
    return matched


def to_vis_payload(
    graph: GraphData,
    *,
    highlight: set[str],
    new_ids: set[str],
) -> dict:
    """Render the graph into the vis-network node/edge dicts the template eats."""
    highlight = highlight or set()
    new_ids = new_ids or set()
    nodes = []
    for node in graph.nodes.values():
        nodes.append({
            "id": node.id,
            "label": node.label,
            "title": node.title,
            "group": _vis_group(node),
            "highlighted": node.id in highlight,
            "new": node.id in new_ids,
        })
    edges = [
        {
            "id": edge.id,
            "from": edge.src,
            "to": edge.dst,
            "label": edge.label,
            "title": edge.title,
        }
        for edge in graph.edges.values()
    ]
    return {"nodes": nodes, "edges": edges}


def _vis_group(node: GraphNode) -> str:
    """Group name driving colour/shape in the template.

    Entities are grouped by source graph (KQAPro blue, SciQA green); an
    entity whose source is unknown falls back to a neutral "entity" group
    rather than being painted with one graph's colour by accident.
    """
    if node.kind == "literal":
        return "literal"
    if node.kind == "candidate":
        return "candidate"
    if node.source in ("kqapro", "sciqa"):
        return node.source
    return "entity"


def apply_caps(
    graph: GraphData,
    *,
    max_nodes: int = MAX_NODES,
    max_edges: int = MAX_EDGES,
) -> GraphData:
    """Trim an oversized graph, keeping the most informative nodes.

    Entities first, then literals, then search-only candidates: a candidate
    the agent never visited is the least interesting thing on the canvas.
    Edges whose endpoints were dropped go with them.
    """
    if len(graph.nodes) <= max_nodes and len(graph.edges) <= max_edges:
        return graph

    buckets: dict[str, list[GraphNode]] = {"entity": [], "literal": [], "candidate": []}
    for node in graph.nodes.values():
        buckets.setdefault(node.kind, []).append(node)
    ordered = buckets["entity"] + buckets["literal"] + buckets["candidate"]
    kept = ordered[:max_nodes]
    kept_ids = {node.id for node in kept}

    nodes = {
        node.id: node
        for node in graph.nodes.values()
        if node.id in kept_ids
    }
    edges: dict[str, GraphEdge] = {}
    for edge in graph.edges.values():
        if len(edges) >= max_edges:
            break
        if edge.src in kept_ids and edge.dst in kept_ids:
            edges[edge.id] = edge
    return GraphData(
        nodes=nodes,
        edges=edges,
        truncated_nodes=graph.truncated_nodes + (len(graph.nodes) - len(nodes)),
    )

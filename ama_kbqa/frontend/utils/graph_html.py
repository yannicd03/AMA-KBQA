"""Build the standalone HTML for an embedded vis-network graph.

We avoid adding a Python graph-viz dependency and instead ship a small HTML
template that loads vis-network from a CDN. The template is parameterised by:

- nodes: list of {id, label, group?, title?}
- edges: list of {from, to, label?}
- view_state_key: localStorage key for persisting pan/zoom across rerenders
"""

from __future__ import annotations

import html
import json
from typing import Any, Optional


_VIS_NETWORK_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def journal_to_graph(state: dict) -> tuple[list[dict], list[dict]]:
    """Convert a JournalState dict to vis-network nodes + edges.

    - nodes come from `visited_nodes: {id: label}`. `found_values` adds a
      tooltip with attribute → value pairs.
    - edges come from `verified_facts: list[{subject, predicate, object,
      source}]`. We try to resolve subject/object back to node ids; when the
      target node hasn't been visited yet we still draw it (orphan node) so
      the user sees what's been observed.
    """
    visited: dict[str, str] = state.get("visited_nodes") or {}
    found_values: dict[str, dict[str, Any]] = state.get("found_values") or {}
    verified_facts: list[dict] = state.get("verified_facts") or []

    # 1) Nodes from visited_nodes.
    nodes_by_id: dict[str, dict] = {}
    for nid, label in visited.items():
        title_lines = [f"<b>{html.escape(label)}</b>", html.escape(nid)]
        attrs = found_values.get(nid)
        if attrs:
            title_lines.append("<br>")
            for k, v in list(attrs.items())[:8]:
                v_repr = _short_repr(v)
                title_lines.append(f"<i>{html.escape(str(k))}</i>: {html.escape(v_repr)}")
        nodes_by_id[nid] = {
            "id": nid,
            "label": label or nid,
            "title": "<br>".join(title_lines),
            "group": "visited",
        }

    # 2) Edges from verified_facts. Subject and object may be node ids or
    # plain strings (e.g. literal values). When neither is in visited_nodes
    # we still create lightweight pseudo-nodes so the edge has endpoints.
    edges: list[dict] = []
    for i, fact in enumerate(verified_facts):
        subj = str(fact.get("subject", "")).strip()
        pred = str(fact.get("predicate", "")).strip()
        obj_raw = fact.get("object", "")
        obj = str(obj_raw).strip() if obj_raw is not None else ""
        if not subj or not obj:
            continue
        for endpoint in (subj, obj):
            if endpoint not in nodes_by_id:
                # Pseudo-node for unseen endpoint (literal value or
                # not-yet-visited entity).
                nodes_by_id[endpoint] = {
                    "id": endpoint,
                    "label": _truncate(endpoint, 40),
                    "title": html.escape(endpoint),
                    "group": "literal" if not _looks_like_id(endpoint) else "unvisited",
                }
        edges.append({
            "from": subj,
            "to": obj,
            "label": _truncate(pred, 30),
            "title": html.escape(f"{pred}\n(source: {fact.get('source', '?')})"),
            "id": f"e{i}",
        })

    return list(nodes_by_id.values()), edges


def _short_repr(v: Any, limit: int = 80) -> str:
    if isinstance(v, list):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return "{" + ", ".join(list(v.keys())[:3]) + "...}"
    s = str(v)
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _looks_like_id(s: str) -> bool:
    """Heuristic: KQAPro/Wikidata IDs look like 'Q123' or have URI shape."""
    if not s:
        return False
    if s[0] in "QPL" and s[1:].isdigit():
        return True
    if s.startswith("http://") or s.startswith("https://"):
        return True
    if s.startswith("orkgr:") or s.startswith("orkgp:"):
        return True
    return False


def build_graph_html(
    nodes: list[dict],
    edges: list[dict],
    *,
    view_state_key: str = "ama_kbqa_graph_view",
    height_px: int = 620,
    highlight_node_ids: Optional[list[str]] = None,
) -> str:
    """Return a complete HTML document embedding the vis-network graph.

    `view_state_key` is used inside the iframe's localStorage so pan/zoom
    persists across Streamlit reruns (which remount the iframe).
    """
    payload = {
        "nodes": nodes,
        "edges": edges,
        "highlight": highlight_node_ids or [],
        "view_state_key": view_state_key,
    }
    payload_json = json.dumps(payload, default=str)
    return _TEMPLATE.replace("__PAYLOAD__", payload_json).replace(
        "__HEIGHT__", str(height_px)
    ).replace("__CDN__", _VIS_NETWORK_CDN)


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body { margin: 0; padding: 0; background: #0d1117; }
  #graph {
    width: 100%;
    height: __HEIGHT__px;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
  }
  .legend {
    position: absolute;
    top: 8px;
    right: 12px;
    background: rgba(13,17,23,0.85);
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: monospace;
    font-size: 11px;
    color: #c9d1d9;
    z-index: 10;
  }
  .legend .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }
</style>
</head>
<body>
<div id="graph"></div>
<div class="legend">
  <div><span class="swatch" style="background:#58a6ff"></span>visited</div>
  <div><span class="swatch" style="background:#a371f7"></span>unvisited</div>
  <div><span class="swatch" style="background:#3fb950"></span>literal</div>
  <div><span class="swatch" style="background:#f78166"></span>highlighted</div>
</div>
<script src="__CDN__"></script>
<script>
(function() {
  const payload = __PAYLOAD__;
  const highlightSet = new Set(payload.highlight || []);

  const groupColours = {
    visited:    { background: "#1f3960", border: "#58a6ff" },
    unvisited:  { background: "#3a2e60", border: "#a371f7" },
    literal:    { background: "#1d3a2a", border: "#3fb950" },
  };

  const nodes = new vis.DataSet(payload.nodes.map(n => ({
    id: n.id,
    label: n.label,
    title: n.title,
    color: highlightSet.has(n.id)
      ? { background: "#5a2a1d", border: "#f78166" }
      : (groupColours[n.group] || groupColours.visited),
    font: { color: "#e6edf3", size: 13, face: "monospace" },
    borderWidth: highlightSet.has(n.id) ? 3 : 1,
    shape: n.group === "literal" ? "box" : "dot",
    size: 14,
  })));
  const edges = new vis.DataSet(payload.edges.map(e => ({
    id: e.id,
    from: e.from,
    to: e.to,
    label: e.label,
    title: e.title,
    arrows: "to",
    color: { color: "#8b949e", highlight: "#58a6ff" },
    font: { color: "#8b949e", size: 10, face: "monospace", strokeWidth: 0, align: "middle" },
    smooth: { type: "continuous" },
  })));

  const container = document.getElementById("graph");
  const data = { nodes, edges };
  const options = {
    physics: {
      stabilization: { iterations: 200, fit: true },
      barnesHut: { gravitationalConstant: -8000, springLength: 110 },
    },
    interaction: { hover: true, navigationButtons: false, keyboard: true },
    edges: { width: 1.2 },
  };
  const network = new vis.Network(container, data, options);

  // Persist view state across reruns. The iframe remounts on every
  // Streamlit script run, so we save pan/zoom in localStorage and restore
  // on the next mount under the same key.
  const stateKey = payload.view_state_key;
  network.once("stabilized", function() {
    try {
      const saved = JSON.parse(localStorage.getItem(stateKey));
      if (saved && saved.position && typeof saved.scale === "number") {
        network.moveTo({ position: saved.position, scale: saved.scale, animation: false });
      }
    } catch (e) { /* ignore */ }
  });
  network.on("dragEnd", saveViewState);
  network.on("zoom", saveViewState);
  function saveViewState() {
    try {
      const v = { position: network.getViewPosition(), scale: network.getScale() };
      localStorage.setItem(stateKey, JSON.stringify(v));
    } catch (e) { /* ignore */ }
  }
})();
</script>
</body>
</html>
"""

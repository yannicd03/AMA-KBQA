"""Build the standalone HTML for an embedded vis-network graph.

We avoid adding a Python graph-viz dependency and instead ship a small HTML
template that loads vis-network from a CDN. The template is parameterised by:

- nodes: list of {id, label, title?, group?, highlighted?, new?} as produced by
  ``live_graph_data.to_vis_payload``
- edges: list of {id, from, to, label?, title?}
- view_state_key: localStorage key for persisting pan/zoom and node positions

Two things the original post-hoc version could not do, and the live panel
needs:

1. **Positions persist.** Streamlit remounts the iframe on every rerun, so a
   graph that re-lays-out from scratch on every tick looks like it is
   exploding rather than growing. Node positions are written to
   ``<view_state_key>:pos`` and seeded back on the next mount, and physics
   only re-fits the viewport when there was nothing saved.
2. **The palette matches the demo.** The demo's Streamlit theme is light
   (``frontend/.streamlit/config.toml``: background #fafafa, text #3f3f46,
   primary #60a5fa, Inter). The old dark-on-#0d1117 template sat in the page
   like a hole; it is replaced rather than kept as an option.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ama_kbqa.frontend.utils.live_graph_data import (
    graph_from_journal_state,
    source_for_agent,
    to_vis_payload,
)


_VIS_NETWORK_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def journal_to_graph(state: dict) -> tuple[list[dict], list[dict]]:
    """Convert a JournalState dict to vis-network nodes + edges.

    Compatibility wrapper over ``live_graph_data.graph_from_journal_state``,
    kept because ``graph_panel.py`` (the post-hoc scrubber, still used on the
    ``dev`` line) consumes the old list-of-dicts shape. The normalisation
    itself lives in one place so the scrubber and the live panel can never
    disagree about what a journal means.

    Returns ``(nodes, edges)`` where nodes carry ``id/label/title/group`` and
    edges carry ``id/from/to/label/title``.
    """
    if not isinstance(state, dict):
        state = {}
    graph = graph_from_journal_state(
        state, source=source_for_agent(state.get("kg_name") or "")
    )
    payload = to_vis_payload(graph, highlight=set(), new_ids=set())
    nodes = [
        {"id": n["id"], "label": n["label"], "title": n["title"], "group": n["group"]}
        for n in payload["nodes"]
    ]
    edges = [
        {
            "from": e["from"],
            "to": e["to"],
            "label": e["label"],
            "title": e["title"],
            "id": e["id"],
        }
        for e in payload["edges"]
    ]
    return nodes, edges


def build_graph_html(
    nodes: list[dict],
    edges: list[dict],
    *,
    view_state_key: str = "ama_kbqa_graph_view",
    height_px: int = 620,
    highlight_node_ids: Optional[list[str]] = None,
    new_node_ids: Optional[Iterable[str]] = None,
    reset: bool = False,
) -> str:
    """Return a complete HTML document embedding the vis-network graph.

    ``view_state_key`` keys both localStorage entries the iframe writes: the
    viewport (pan/zoom) under the key itself and the node positions under
    ``<key>:pos``. Pass ``reset=True`` on the first render of a new trace to
    drop both, so a new question starts from a clean layout instead of
    inheriting the previous graph's coordinates.
    """
    payload: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "highlight": list(highlight_node_ids or []),
        "new": list(new_node_ids or []),
        "view_state_key": view_state_key,
        "reset": bool(reset),
    }
    # The payload is spliced into an inline <script>, where an HTML parser
    # ends the script at the first literal "</" sequence even inside a JS
    # string. KG labels really do contain markup-ish text, so escape it as
    # "<\/", which JSON parses back to the original characters.
    payload_json = json.dumps(payload, default=str).replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("__PAYLOAD__", payload_json)
        .replace("__HEIGHT__", str(height_px))
        .replace("__CDN__", _VIS_NETWORK_CDN)
    )


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 0;
    padding: 0;
    background: #fafafa;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
  }
  #graph {
    width: 100%;
    height: __HEIGHT__px;
    background: #ffffff;
    border: 1px solid #eeeef0;
    border-radius: 8px;
  }
  .legend {
    position: absolute;
    top: 10px;
    right: 14px;
    background: rgba(255,255,255,0.92);
    border: 1px solid #eeeef0;
    border-radius: 6px;
    padding: 7px 10px;
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
    font-size: 11px;
    color: #3f3f46;
    line-height: 1.6;
    z-index: 10;
  }
  .legend .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    box-sizing: border-box;
  }
  .legend .swatch.box { border-radius: 2px; }
  .legend .swatch.dashed { border-style: dashed; background: #ffffff; }
</style>
</head>
<body>
<div id="graph"></div>
<div class="legend">
  <div><span class="swatch" style="background:#dbeafe;border:2px solid #3b82f6"></span>KQAPro</div>
  <div><span class="swatch" style="background:#dcfce7;border:2px solid #22c55e"></span>SciQA</div>
  <div><span class="swatch box" style="background:#f4f4f5;border:1px solid #d4d4d8"></span>literal</div>
  <div><span class="swatch dashed" style="border:2px dashed #93c5fd"></span>candidate</div>
  <div><span class="swatch" style="background:#ffedd5;border:3px solid #f97316"></span>in answer</div>
</div>
<script src="__CDN__"></script>
<script>
(function() {
  const payload = __PAYLOAD__;
  const highlightSet = new Set(payload.highlight || []);
  const newSet = new Set(payload["new"] || []);

  // Light-theme palette, matching frontend/.streamlit/config.toml.
  const groupColours = {
    kqapro:    { background: "#dbeafe", border: "#3b82f6" },
    sciqa:     { background: "#dcfce7", border: "#22c55e" },
    entity:    { background: "#e4e4e7", border: "#a1a1aa" },
    literal:   { background: "#f4f4f5", border: "#d4d4d8" },
    candidate: { background: "#ffffff", border: "#93c5fd" },
  };
  const ANSWER_COLOUR = { background: "#ffedd5", border: "#f97316" };
  const NEW_BORDER = "#f59e0b";

  const stateKey = payload.view_state_key;
  const posKey = stateKey + ":pos";

  // A new trace starts from a clean canvas: the previous question's
  // coordinates and viewport would otherwise be seeded into an unrelated
  // graph.
  if (payload.reset) {
    try { localStorage.removeItem(stateKey); } catch (e) { /* ignore */ }
    try { localStorage.removeItem(posKey); } catch (e) { /* ignore */ }
  }

  let savedPositions = {};
  try {
    savedPositions = JSON.parse(localStorage.getItem(posKey)) || {};
  } catch (e) { savedPositions = {}; }

  let hasSavedPositions = false;
  const nodes = new vis.DataSet((payload.nodes || []).map(function(n) {
    const group = n.group || "entity";
    const isAnswer = n.highlighted || highlightSet.has(n.id);
    const isNew = n.new || newSet.has(n.id);
    const base = groupColours[group] || groupColours.entity;
    const colour = isAnswer ? ANSWER_COLOUR : base;
    const node = {
      id: n.id,
      label: n.label,
      title: n.title,
      group: group,
      color: {
        background: colour.background,
        border: isNew ? NEW_BORDER : colour.border,
        highlight: { background: colour.background, border: colour.border },
      },
      font: {
        color: group === "literal" ? "#52525b" : "#3f3f46",
        size: group === "literal" ? 11 : 13,
        face: "Inter, Segoe UI, system-ui, sans-serif",
      },
      borderWidth: isAnswer ? 4 : (isNew ? 3 : 2),
      shape: group === "literal" ? "box" : "dot",
      size: group === "literal" ? 8 : 14,
      shapeProperties: { borderDashes: group === "candidate" ? [4, 3] : false },
    };
    const pos = savedPositions[n.id];
    if (pos && typeof pos.x === "number" && typeof pos.y === "number") {
      node.x = pos.x;
      node.y = pos.y;
      hasSavedPositions = true;
    }
    return node;
  }));

  const edges = new vis.DataSet((payload.edges || []).map(function(e) {
    return {
      id: e.id,
      from: e.from,
      to: e.to,
      label: e.label,
      title: e.title,
      arrows: "to",
      color: { color: "#a1a1aa", highlight: "#60a5fa" },
      font: {
        color: "#71717a",
        size: 10,
        face: "Inter, Segoe UI, system-ui, sans-serif",
        strokeWidth: 3,
        strokeColor: "#ffffff",
        align: "middle",
      },
      smooth: { type: "continuous" },
    };
  }));

  const container = document.getElementById("graph");
  const data = { nodes: nodes, edges: edges };
  const options = {
    physics: {
      // Few iterations: the graph is re-rendered as it grows, so a long
      // stabilisation run would just burn CPU between ticks. Only fit the
      // viewport when there were no saved positions to honour.
      stabilization: { iterations: 80, fit: !hasSavedPositions },
      barnesHut: { gravitationalConstant: -8000, springLength: 110 },
    },
    interaction: { hover: true, navigationButtons: false, keyboard: true },
    edges: { width: 1.2 },
  };
  const network = new vis.Network(container, data, options);

  function savePositions() {
    try {
      localStorage.setItem(posKey, JSON.stringify(network.getPositions()));
    } catch (e) { /* ignore */ }
  }
  function saveViewState() {
    try {
      const v = { position: network.getViewPosition(), scale: network.getScale() };
      localStorage.setItem(stateKey, JSON.stringify(v));
    } catch (e) { /* ignore */ }
  }

  network.once("stabilized", function() {
    savePositions();
    try {
      const saved = JSON.parse(localStorage.getItem(stateKey));
      if (saved && saved.position && typeof saved.scale === "number") {
        network.moveTo({ position: saved.position, scale: saved.scale, animation: false });
      }
    } catch (e) { /* ignore */ }
  });
  network.on("dragEnd", function() {
    savePositions();
    saveViewState();
  });
  network.on("zoom", saveViewState);
})();
</script>
</body>
</html>
"""

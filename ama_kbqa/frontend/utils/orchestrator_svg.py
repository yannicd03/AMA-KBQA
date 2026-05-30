"""Inline SVG of the multi-agent orchestrator figure (`fig:agent_flow_swarm`).

The layout mirrors the paper's orchestrator variant: a left-hand **User Query**
feeding an **Orchestrator** column (Datasource Probing → Async Dispatch →
Answer Combination), which dispatches to a fixed set of specialist sub-agents.
Each sub-agent is drawn as a container with three Pre / Main / Post mini-boxes
echoing the single-agent lifecycle (Fig. 1). The two specialists are the real
registered agents — **KQAPro** and **SciQA**.

Lighting is driven by the live trace: the routed specialist's container lights
while its ``delegate`` span is open, and its Pre/Main/Post minis track that
sub-agent's internal progress. The specialist that is not dispatched stays idle
(dimmed), reproducing the "inactive sub-agent" look of the paper figure.

Like ``lifecycle_svg``, the renderer is pure-Python and Streamlit-free. The SVG
is tagged ``lifecycle-svg orchestrator-svg`` so it reuses the lifecycle CSS in
``styling.py`` (node states, edges, arrowheads) plus a few orchestrator-only
rules. Coordinates are authored directly in SVG space (px, y grows downward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional
from xml.sax.saxutils import escape


Shape = Literal["box", "container", "mini"]


@dataclass(frozen=True)
class ONode:
    id: str
    label: str
    shape: Shape
    cx: float
    cy: float
    w: float
    h: float
    # Sub-agent container caption rendered above the box (containers only).
    caption: Optional[str] = None


# ---------------------------------------------------------------------------
# Layout (px)
# ---------------------------------------------------------------------------

CANVAS_W = 780
CANVAS_H = 380

_CONTAINER_CX = 600.0
_KQAPRO_CY = 120.0
_SCIQA_CY = 250.0
_MINI_DX = 75.0  # horizontal offset of side minis from the container centre


def _mini_row(prefix: str, cy: float) -> list[ONode]:
    return [
        ONode(f"{prefix}_pre",  "Pre",  "mini", _CONTAINER_CX - _MINI_DX, cy, 58, 36),
        ONode(f"{prefix}_main", "Main", "mini", _CONTAINER_CX,            cy, 58, 36),
        ONode(f"{prefix}_post", "Post", "mini", _CONTAINER_CX + _MINI_DX, cy, 58, 36),
    ]


ORCH_NODES: list[ONode] = [
    ONode("orch_user",     "User\nQuery",          "box", 80,  70,  110, 52),
    ONode("orch_probe",    "Datasource\nProbing",  "box", 290, 70,  138, 52),
    ONode("orch_dispatch", "Async\nDispatch",      "box", 290, 185, 138, 52),
    ONode("orch_combine",  "Answer\nCombination",  "box", 290, 300, 138, 52),
    # Sub-agent containers (boxes are drawn; minis sit inside).
    ONode("sub_kqapro", "", "container", _CONTAINER_CX, _KQAPRO_CY, 256, 70, caption="KQAPro"),
    ONode("sub_sciqa",  "", "container", _CONTAINER_CX, _SCIQA_CY,  256, 70, caption="SciQA"),
    *_mini_row("sub_kqapro", _KQAPRO_CY),
    *_mini_row("sub_sciqa", _SCIQA_CY),
]

ORCH_NODES_BY_ID: dict[str, ONode] = {n.id: n for n in ORCH_NODES}


# Orchestrator phase band behind the three orchestrator boxes.
# (x, y, w, h, title)
ORCH_BAND: tuple[float, float, float, float, str] = (218, 18, 144, 326, "Orchestrator")


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OEdge:
    # Absolute polyline: a start point, optional waypoints, and an end point.
    points: tuple[tuple[float, float], ...]
    id: Optional[str] = None
    style: Literal["solid", "dashed"] = "solid"


def _x(node_id: str) -> ONode:
    return ORCH_NODES_BY_ID[node_id]


def _side(n: ONode, s: str) -> tuple[float, float]:
    hw, hh = n.w / 2, n.h / 2
    return {
        "n": (n.cx, n.cy - hh),
        "s": (n.cx, n.cy + hh),
        "e": (n.cx + hw, n.cy),
        "w": (n.cx - hw, n.cy),
    }[s]


# Dispatch branch point and return rail, in absolute coords.
_BRANCH_X = 448.0
_RAIL_X = 752.0


ORCH_EDGES: list[OEdge] = [
    # User Query → Datasource Probing
    OEdge((_side(_x("orch_user"), "e"), _side(_x("orch_probe"), "w"))),
    # Orchestrator column: probe → dispatch
    OEdge((_side(_x("orch_probe"), "s"), _side(_x("orch_dispatch"), "n"))),
    # Async dispatch trunk → branch → each sub-agent container (left side)
    OEdge((_side(_x("orch_dispatch"), "e"), (_BRANCH_X, 185),
           (_BRANCH_X, _KQAPRO_CY), _side(_x("sub_kqapro"), "w")),
          id="dispatch_kqapro"),
    OEdge((_side(_x("orch_dispatch"), "e"), (_BRANCH_X, 185),
           (_BRANCH_X, _SCIQA_CY), _side(_x("sub_sciqa"), "w")),
          id="dispatch_sciqa"),
    # Returns: each container's right edge → shared rail → up into Answer Combination
    OEdge((_side(_x("sub_kqapro"), "e"), (_RAIL_X, _KQAPRO_CY),
           (_RAIL_X, 352), (308, 352), (308, _side(_x("orch_combine"), "s")[1])),
          id="return_kqapro"),
    OEdge((_side(_x("sub_sciqa"), "e"), (_RAIL_X - 16, _SCIQA_CY),
           (_RAIL_X - 16, 344), (272, 344), (272, _side(_x("orch_combine"), "s")[1])),
          id="return_sciqa"),
    # Loop back: Answer Combination → User Query (left then up)
    OEdge((_side(_x("orch_combine"), "w"), (40, 300), (40, 70),
           _side(_x("orch_user"), "w")),
          id="loop_user", style="dashed"),
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _state_for(node_id: str, active: set[str], visited: set[str]) -> str:
    if node_id in active:
        return "active"
    if node_id in visited:
        return "visited"
    return "idle"


def _multiline(cx: float, cy: float, label: str, css_class: str) -> str:
    lines = label.split("\n")
    first_dy = -((len(lines) - 1) / 2) * 1.2
    parts = [
        f'<text x="{cx:.1f}" y="{cy:.1f}" class="{css_class}"'
        f' text-anchor="middle" dominant-baseline="middle">'
    ]
    for i, line in enumerate(lines):
        dy = f"{first_dy:.2f}em" if i == 0 else "1.2em"
        parts.append(f'<tspan x="{cx:.1f}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _node_svg(n: ONode, state: str) -> str:
    x0, y0 = n.cx - n.w / 2, n.cy - n.h / 2
    if n.shape == "container":
        rect = (
            f'<rect class="node node-container" x="{x0:.1f}" y="{y0:.1f}"'
            f' width="{n.w:.1f}" height="{n.h:.1f}" rx="9" ry="9" />'
        )
        caption = ""
        if n.caption:
            caption = (
                f'<text class="subagent-label" x="{n.cx:.1f}"'
                f' y="{y0 - 9:.1f}" text-anchor="middle">{escape(n.caption)}</text>'
            )
        return f'<g data-id="{n.id}" data-state="{state}">{rect}{caption}</g>'

    rx = 5 if n.shape == "mini" else 6
    cls = "node node-smallbox" if n.shape == "mini" else "node node-box"
    rect = (
        f'<rect class="{cls}" x="{x0:.1f}" y="{y0:.1f}"'
        f' width="{n.w:.1f}" height="{n.h:.1f}" rx="{rx}" ry="{rx}" />'
    )
    label_cls = "node-label" + (" node-label-small" if n.shape == "mini" else "")
    text = _multiline(n.cx, n.cy, n.label, label_cls)
    return f'<g data-id="{n.id}" data-state="{state}">{rect}{text}</g>'


def _edge_svg(e: OEdge, active_edge_ids: frozenset[str]) -> str:
    eid = e.id or ""
    is_active = bool(eid) and eid in active_edge_ids
    classes = ["edge"]
    if e.style == "dashed":
        classes.append("dashed")
    head = "lifecycle-arrowhead-active" if is_active else "lifecycle-arrowhead"
    state_attr = ' data-state="active"' if is_active else ""
    id_attr = f' data-edge-id="{escape(eid)}"' if eid else ""
    p0 = e.points[0]
    d = f"M {p0[0]:.1f} {p0[1]:.1f}"
    for px, py in e.points[1:]:
        d += f" L {px:.1f} {py:.1f}"
    return (
        f'<path class="{" ".join(classes)}"{id_attr}{state_attr} d="{d}"'
        f' marker-end="url(#{head})" />'
    )


def render_orchestrator_svg(
    active_node_ids: Iterable[str] = (),
    visited_node_ids: Iterable[str] = (),
    current_label: Optional[str] = None,
    *,
    active_edge_ids: Iterable[str] = (),
    width: int = CANVAS_W,
    height: int = CANVAS_H,
) -> str:
    """Render the orchestrator figure as a self-contained SVG string.

    Node ids match ``orchestrator_span_to_node_ids`` / ``ORCH_SUBAGENTS`` in
    ``lifecycle_mapping`` plus the per-container ``*_pre/_main/_post`` minis.
    Nodes carry ``data-state`` (``idle``/``visited``/``active``); the CSS in
    ``styling.py`` colours them.
    """
    active = set(active_node_ids)
    visited = set(visited_node_ids) | active
    active_edges = frozenset(active_edge_ids)

    bx, by, bw, bh, btitle = ORCH_BAND
    band = (
        f'<rect class="phase-bg phase-main" x="{bx}" y="{by}" width="{bw}"'
        f' height="{bh}" rx="14" ry="14" />'
        f'<text class="phase-title" x="{bx + bw / 2:.1f}" y="{by - 8:.1f}"'
        f' text-anchor="middle">{escape(btitle)}</text>'
    )

    defs = (
        '<defs>'
        '<marker id="lifecycle-arrowhead" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" class="edge-arrow" />'
        '</marker>'
        '<marker id="lifecycle-arrowhead-active" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" class="edge-arrow-active" />'
        '</marker>'
        '</defs>'
    )

    edges = [_edge_svg(e, active_edges) for e in ORCH_EDGES]
    nodes = [_node_svg(n, _state_for(n.id, active, visited)) for n in ORCH_NODES]

    caption = ""
    if current_label:
        caption = (
            f'<text class="lifecycle-caption" x="{width / 2:.1f}" y="{height - 6:.1f}"'
            f' text-anchor="middle">{escape(current_label)}</text>'
        )

    svg_open = (
        f'<svg class="lifecycle-svg orchestrator-svg"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 -4 {width} {height}" width="100%"'
        f' role="img" aria-label="Orchestrator multi-agent flow">'
    )
    return (
        svg_open
        + defs
        + band
        + "".join(edges)
        + "".join(nodes)
        + caption
        + "</svg>"
    )

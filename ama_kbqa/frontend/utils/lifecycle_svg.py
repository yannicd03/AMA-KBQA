"""Inline SVG of the agent-lifecycle figure (`fig:agent_flow` in the paper).

The layout mirrors the paper figure: a left-hand *Agent Invocation*, then three
phase bands left-to-right — the deterministic **Pre-Agent Hook** (Question Type
Classification → Entity Extraction → Strategy Injection), the **Main-Agent
Loop** (LLM Reasoning → Tool Call → Scratchpad → Done?, looping on "no" and
exiting on "yes"), and the **Post-Agent Hook** (Answer Synthesis → Trace
Evaluation → Lessons Learned). A dashed feedback edge carries Lessons Learned
back to Strategy Injection.

Coordinates are authored directly in SVG space (px, y grows downward).

The renderer is pure-Python and Streamlit-free so it can be unit-tested in
isolation. Each node carries ``data-id`` and ``data-state`` (``idle`` /
``visited`` / ``active``); the accompanying CSS in ``styling.py`` colours them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional
from xml.sax.saxutils import escape


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

# Box sizes (in px).
BOX_W, BOX_H = 132, 46
SMALL_W, SMALL_H = 96, 34
DIAMOND_W, DIAMOND_H = 112, 66

Shape = Literal["box", "smallbox", "diamond", "textlabel"]
Phase = Literal["pre", "main", "post", "input"]


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    phase: Phase
    shape: Shape
    cx: float
    cy: float
    w: float
    h: float


def _b(id: str, label: str, phase: Phase, cx: float, cy: float) -> Node:
    return Node(id, label, phase, "box", cx, cy, BOX_W, BOX_H)


def _sb(id: str, label: str, phase: Phase, cx: float, cy: float) -> Node:
    return Node(id, label, phase, "smallbox", cx, cy, SMALL_W, SMALL_H)


def _d(id: str, label: str, phase: Phase, cx: float, cy: float, h: float = DIAMOND_H) -> Node:
    return Node(id, label, phase, "diamond", cx, cy, DIAMOND_W, h)


def _tl(id: str, label: str, phase: Phase, cx: float, cy: float) -> Node:
    return Node(id, label, phase, "textlabel", cx, cy, 0, 0)


LIFECYCLE_NODES: list[Node] = [
    _b("agent_invocation",    "Agent\nInvocation",          "input", 70,  58),
    # Pre-Agent Hook (vertical chain)
    _b("pre_classifier",      "Question Type\nClassification", "pre", 266, 58),
    _b("pre_extractor",       "Entity\nExtraction",         "pre",   266, 150),
    _b("pre_strategy_inject", "Strategy\nInjection",        "pre",   266, 242),
    # Main-Agent Loop (vertical chain + decision)
    _b("main_llm_reason",     "LLM\nReasoning",             "main",  520, 50),
    _b("main_tool_call",      "Tool\nCall",                 "main",  520, 118),
    _b("main_scratchpad",     "Scratchpad",                 "main",  520, 184),
    _d("main_done",           "Done?",                      "main",  520, 254),
    # Post-Agent Hook (vertical chain)
    _b("post_synthesis",      "Answer\nSynthesis",          "post",  780, 58),
    _b("post_evaluate",       "Trace\nEvaluation",          "post",  780, 150),
    _b("post_lessons",        "Lessons\nLearned",           "post",  780, 242),
]

NODES_BY_ID: dict[str, Node] = {n.id: n for n in LIFECYCLE_NODES}


# Phase background bands. (x, y, w, h, title, css-class)
PHASE_BANDS: list[tuple[float, float, float, float, str, str]] = [
    (172, 0,   188, 300, "Pre-Agent Hook",  "phase-pre"),
    (420, 0,   200, 300, "Main-Agent Loop", "phase-main"),
    (686, 0,   188, 300, "Post-Agent Hook", "phase-post"),
]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

EdgeStyle = Literal["solid", "dashed", "bidir-dashed"]


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    style: EdgeStyle = "solid"
    # routing="straight" draws a single straight line. "orth" routes
    # orthogonally via the listed waypoints (relative anchor sides). "custom"
    # takes a list of absolute (x, y) waypoints between the two endpoints.
    routing: Literal["straight", "orth", "custom"] = "straight"
    waypoints: tuple[tuple[float, float], ...] = ()
    from_side: Literal["n", "s", "e", "w", "ne", "nw", "se", "sw"] = "e"
    to_side:   Literal["n", "s", "e", "w", "ne", "nw", "se", "sw"] = "w"
    label: Optional[str] = None
    label_pos: Optional[tuple[float, float]] = None


LIFECYCLE_EDGES: list[Edge] = [
    # Agent Invocation → Question Type Classification (straight, into Pre band)
    Edge("agent_invocation", "pre_classifier", from_side="e", to_side="w",
         routing="straight"),

    # Pre-Agent Hook vertical chain
    Edge("pre_classifier", "pre_extractor", from_side="s", to_side="n", routing="straight"),
    Edge("pre_extractor", "pre_strategy_inject", from_side="s", to_side="n", routing="straight"),

    # Strategy Injection → LLM Reasoning (left bracket entering the loop)
    Edge("pre_strategy_inject", "main_llm_reason", from_side="e", to_side="w",
         routing="custom",
         waypoints=((390, 242), (390, 50))),

    # Main-Agent Loop vertical chain
    Edge("main_llm_reason", "main_tool_call", from_side="s", to_side="n", routing="straight"),
    Edge("main_tool_call", "main_scratchpad", from_side="s", to_side="n", routing="straight"),
    Edge("main_scratchpad", "main_done", from_side="s", to_side="n", routing="straight"),

    # Done? "no" → loop back up to LLM Reasoning (left bracket)
    Edge("main_done", "main_llm_reason", from_side="w", to_side="w",
         routing="custom",
         waypoints=((400, 254), (400, 50)),
         label="no", label_pos=(430, 248)),

    # Done? "yes" → Answer Synthesis (right then up, into Post band)
    Edge("main_done", "post_synthesis", from_side="e", to_side="w",
         routing="custom",
         waypoints=((650, 254), (650, 58)),
         label="yes", label_pos=(606, 248)),

    # Post-Agent Hook vertical chain
    Edge("post_synthesis", "post_evaluate", from_side="s", to_side="n", routing="straight"),
    Edge("post_evaluate", "post_lessons", from_side="s", to_side="n", routing="straight"),

    # Lessons Learned → Strategy Injection (dashed feedback along the bottom)
    Edge("post_lessons", "pre_strategy_inject", from_side="s", to_side="s",
         style="dashed", routing="custom",
         waypoints=((780, 330), (266, 330))),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _anchor(n: Node, side: str) -> tuple[float, float]:
    """Return the (x, y) point on a node's border for the named side.

    For text labels the anchor is just the centre (text has no real border).
    """
    if n.shape == "textlabel":
        return (n.cx, n.cy)
    hw, hh = n.w / 2, n.h / 2
    if n.shape == "diamond":
        # Anchor at the diamond's tip points for the four cardinal sides.
        if side == "n":
            return (n.cx, n.cy - hh)
        if side == "s":
            return (n.cx, n.cy + hh)
        if side == "e":
            return (n.cx + hw, n.cy)
        if side == "w":
            return (n.cx - hw, n.cy)
        # Diagonal sides: roughly half-way along each face.
        if side == "ne":
            return (n.cx + hw * 0.5, n.cy - hh * 0.5)
        if side == "nw":
            return (n.cx - hw * 0.5, n.cy - hh * 0.5)
        if side == "se":
            return (n.cx + hw * 0.5, n.cy + hh * 0.5)
        if side == "sw":
            return (n.cx - hw * 0.5, n.cy + hh * 0.5)
    # Rectangles (box / smallbox)
    if side == "n":
        return (n.cx, n.cy - hh)
    if side == "s":
        return (n.cx, n.cy + hh)
    if side == "e":
        return (n.cx + hw, n.cy)
    if side == "w":
        return (n.cx - hw, n.cy)
    if side == "ne":
        return (n.cx + hw, n.cy - hh)
    if side == "nw":
        return (n.cx - hw, n.cy - hh)
    if side == "se":
        return (n.cx + hw, n.cy + hh)
    if side == "sw":
        return (n.cx - hw, n.cy + hh)
    return (n.cx, n.cy)


def _edge_path(e: Edge) -> str:
    a = NODES_BY_ID[e.from_id]
    b = NODES_BY_ID[e.to_id]
    sx, sy = _anchor(a, e.from_side)
    tx, ty = _anchor(b, e.to_side)
    if e.routing == "straight" or not e.waypoints:
        return f"M {sx:.1f} {sy:.1f} L {tx:.1f} {ty:.1f}"
    parts = [f"M {sx:.1f} {sy:.1f}"]
    for (wx, wy) in e.waypoints:
        parts.append(f"L {wx:.1f} {wy:.1f}")
    parts.append(f"L {tx:.1f} {ty:.1f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _state_for(node_id: str, active: set[str], visited: set[str]) -> str:
    if node_id in active:
        return "active"
    if node_id in visited:
        return "visited"
    return "idle"


def _multiline_text(x: float, y: float, label: str, *, css_class: str,
                     anchor: str = "middle", baseline: str = "middle") -> str:
    """Render a multi-line label centred at (x, y).

    Lines are separated by ``\n``. The first <tspan> is dy=0 and subsequent
    spans use dy=1.2em so the block grows downward; we then shift the whole
    block upward so it stays vertically centred on (x, y).
    """
    lines = label.split("\n")
    n = len(lines)
    # Shift block upward by (n-1)/2 lines so the visual centre is at y.
    first_dy = -((n - 1) / 2) * 1.2
    parts = [
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css_class}"'
        f' text-anchor="{anchor}" dominant-baseline="{baseline}">'
    ]
    for i, line in enumerate(lines):
        dy_attr = f'{first_dy:.2f}em' if i == 0 else "1.2em"
        parts.append(f'<tspan x="{x:.1f}" dy="{dy_attr}">{escape(line)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def _node_svg(n: Node, state: str) -> str:
    """Render a single node (shape + label) keyed by state."""
    if n.shape == "textlabel":
        # Bold first line ("Tools A" / "Tools B"), regular rest.
        lines = n.label.split("\n")
        first_dy = -((len(lines) - 1) / 2) * 1.2
        parts = [
            f'<g data-id="{n.id}" data-state="{state}" class="textlabel">',
            f'<text x="{n.cx:.1f}" y="{n.cy:.1f}" class="node-label textlabel-text"'
            f' text-anchor="middle" dominant-baseline="middle">',
        ]
        for i, line in enumerate(lines):
            dy_attr = f'{first_dy:.2f}em' if i == 0 else "1.2em"
            cls = "textlabel-head" if i == 0 else "textlabel-tail"
            parts.append(
                f'<tspan x="{n.cx:.1f}" dy="{dy_attr}" class="{cls}">'
                f'{escape(line)}</tspan>'
            )
        parts.append("</text></g>")
        return "".join(parts)

    if n.shape == "diamond":
        hw, hh = n.w / 2, n.h / 2
        pts = (
            f"{n.cx:.1f},{n.cy - hh:.1f} "
            f"{n.cx + hw:.1f},{n.cy:.1f} "
            f"{n.cx:.1f},{n.cy + hh:.1f} "
            f"{n.cx - hw:.1f},{n.cy:.1f}"
        )
        shape = f'<polygon class="node node-diamond" points="{pts}" />'
    else:
        # Rounded rectangle (box / smallbox)
        x0 = n.cx - n.w / 2
        y0 = n.cy - n.h / 2
        rx = 5 if n.shape == "smallbox" else 6
        cls = "node node-smallbox" if n.shape == "smallbox" else "node node-box"
        shape = (
            f'<rect class="{cls}" x="{x0:.1f}" y="{y0:.1f}"'
            f' width="{n.w:.1f}" height="{n.h:.1f}" rx="{rx}" ry="{rx}" />'
        )

    label_css = "node-label" + (" node-label-small" if n.shape == "smallbox" else "")
    text = _multiline_text(n.cx, n.cy, n.label, css_class=label_css)
    return f'<g data-id="{n.id}" data-state="{state}">{shape}{text}</g>'


def _edge_svg(e: Edge) -> str:
    classes = ["edge"]
    if e.style in ("dashed", "bidir-dashed"):
        classes.append("dashed")
    marker_attr = ' marker-end="url(#lifecycle-arrowhead)"'
    if e.style == "bidir-dashed":
        marker_attr += ' marker-start="url(#lifecycle-arrowhead-rev)"'
    cls = " ".join(classes)
    path_d = _edge_path(e)
    out = f'<path class="{cls}" d="{path_d}"{marker_attr} />'
    if e.label and e.label_pos:
        lx, ly = e.label_pos
        out += (
            f'<text class="edge-label" x="{lx:.1f}" y="{ly:.1f}"'
            f' text-anchor="middle">{escape(e.label)}</text>'
        )
    return out


def render_lifecycle_svg(
    active_node_ids: Iterable[str] = (),
    visited_node_ids: Iterable[str] = (),
    current_label: Optional[str] = None,
    *,
    width: int = 1000,
    height: int = 360,
) -> str:
    """Render the agent-lifecycle figure as a self-contained SVG string.

    Nodes carry ``data-state`` (``idle``/``visited``/``active``) so the CSS
    in ``styling.py`` can highlight whichever stage is currently in flight.
    """
    active = set(active_node_ids)
    visited = set(visited_node_ids) | active

    # Phase background bands (and titles above each band).
    bands_svg: list[str] = []
    for x, y, w, h, title, cls in PHASE_BANDS:
        bands_svg.append(
            f'<rect class="phase-bg {cls}" x="{x}" y="{y}" width="{w}"'
            f' height="{h}" rx="14" ry="14" />'
        )
        bands_svg.append(
            f'<text class="phase-title" x="{x + w / 2:.1f}" y="-12"'
            f' text-anchor="middle">{escape(title)}</text>'
        )

    edges_svg = [_edge_svg(e) for e in LIFECYCLE_EDGES]
    nodes_svg = [
        _node_svg(n, _state_for(n.id, active, visited))
        for n in LIFECYCLE_NODES
    ]

    defs = (
        '<defs>'
        '<marker id="lifecycle-arrowhead" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" class="edge-arrow" />'
        '</marker>'
        '<marker id="lifecycle-arrowhead-rev" viewBox="0 0 10 10" refX="1" refY="5"'
        ' markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 10 0 L 0 5 L 10 10 z" class="edge-arrow" />'
        '</marker>'
        '</defs>'
    )

    caption = ""
    if current_label:
        caption = (
            f'<text class="lifecycle-caption" x="{width / 2:.1f}" y="354"'
            f' text-anchor="middle">{escape(current_label)}</text>'
        )

    svg_open = (
        f'<svg class="lifecycle-svg" xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 -32 {width} {height + 32}" width="100%"'
        f' role="img" aria-label="Agent lifecycle">'
    )
    return (
        svg_open
        + defs
        + "".join(bands_svg)
        + "".join(edges_svg)
        + "".join(nodes_svg)
        + caption
        + "</svg>"
    )

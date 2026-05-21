"""Inline SVG of the agent-lifecycle figure (`fig:agent_flow` in the paper).

Coordinates ported from the TikZ source: ``svg_x = (tikz_x + 5) * 40``,
``svg_y = (4.2 - tikz_y) * 40`` (y flipped because SVG y grows downward).

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

# Box sizes (in px). Match TikZ minimums: box 2.5×1.1cm, smallbox 2.0×0.8cm,
# diamond 2.5×1.8cm at 40 px/cm.
BOX_W, BOX_H = 100, 44
SMALL_W, SMALL_H = 80, 32
DIAMOND_W, DIAMOND_H = 100, 72
DIAMOND_TALL_H = 56  # for "More?" which used minimum_height=1.3cm

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


# Centers come from the TikZ transform documented at the module top.
LIFECYCLE_NODES: list[Node] = [
    _b("user_query",          "User\nQuery",        "input", 72,  168),
    _b("pre_classifier",      "Classifier",         "pre",   200, 96),
    _b("pre_extractor",       "Extractor",          "pre",   200, 240),
    _tl("pre_class_out",      "Count\nQueryAttr\n…","pre",   288, 56),
    _tl("pre_extr_out",       "Entity\nRelation\n…","pre",   288, 272),
    _b("pre_strategy_inject", "Strategy\nInject",   "pre",   380, 168),
    _sb("main_loop_detect",   "Loop\nDetect",       "main",  452, 48),
    _b("main_journal_state",  "Journal\nState",     "main",  580, 48),
    _d("main_llm_reason",     "LLM\nReason",        "main",  580, 168),
    _tl("main_tools_b",       "Tools B\nGetSumm\nVerify",  "main", 720, 128),
    _tl("main_tools_a",       "Tools A\nFindNode\nGetAttr","main", 720, 208),
    _d("main_more",           "More?",              "main",  580, 280, h=DIAMOND_TALL_H),
    _b("post_synthesis",      "Synthesis",          "post",  860, 48),
    _b("post_answer",         "Answer",             "post",  860, 128),
    _b("post_response",       "Response",           "post",  860, 200),
    _b("post_evaluate",       "Evaluate",           "post",  860, 264),
    _sb("post_lessons",       "Lessons\nLearned",   "post",  860, 316),
]

NODES_BY_ID: dict[str, Node] = {n.id: n for n in LIFECYCLE_NODES}


# Phase background bands. (x, y, w, h, title, css-class)
PHASE_BANDS: list[tuple[float, float, float, float, str, str]] = [
    (152, 0,   240, 336, "Pre-Agent Hook",       "phase-pre"),
    (416, 0,   328, 336, "Main-Agent Loop",      "phase-main"),
    (760, 0,   200, 336, "Post-Agent Synthesis", "phase-post"),
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
    # User → Classifier / Extractor (two L-shapes branching from a midpoint)
    Edge("user_query", "pre_classifier", from_side="e", to_side="w",
         routing="custom",
         waypoints=((146, 168), (146, 96))),
    Edge("user_query", "pre_extractor", from_side="e", to_side="w",
         routing="custom",
         waypoints=((146, 168), (146, 240))),

    # Classifier / Extractor → Strategy Inject (L-shapes)
    Edge("pre_classifier", "pre_strategy_inject", from_side="e", to_side="nw",
         routing="custom",
         waypoints=((342, 96), (342, 158))),
    Edge("pre_extractor", "pre_strategy_inject", from_side="e", to_side="sw",
         routing="custom",
         waypoints=((342, 240), (342, 178))),

    # Strategy Inject → LLM Reason
    Edge("pre_strategy_inject", "main_llm_reason", from_side="e", to_side="w",
         routing="straight"),

    # LLM north-west → Loop Detect south-east (dashed, diagonal)
    Edge("main_llm_reason", "main_loop_detect",
         from_side="nw", to_side="se", style="dashed", routing="straight"),

    # Loop Detect → Journal State (straight, east → west)
    Edge("main_loop_detect", "main_journal_state",
         from_side="e", to_side="w", routing="straight"),

    # Journal State → LLM (vertical, south → north)
    Edge("main_journal_state", "main_llm_reason",
         from_side="s", to_side="n", routing="straight"),

    # Tools B ↔ LLM (bidir, dashed, diagonal)
    Edge("main_tools_b", "main_llm_reason",
         from_side="w", to_side="ne", style="bidir-dashed", routing="straight"),

    # Tools A ↔ LLM (bidir, dashed, diagonal)
    Edge("main_tools_a", "main_llm_reason",
         from_side="w", to_side="se", style="bidir-dashed", routing="straight"),

    # LLM south → More? north
    Edge("main_llm_reason", "main_more",
         from_side="s", to_side="n", routing="straight"),

    # More? west → loop back to LLM west (yes branch)
    Edge("main_more", "main_llm_reason", from_side="w", to_side="w",
         routing="custom",
         waypoints=((460, 280), (460, 168)),
         label="yes", label_pos=(442, 295)),

    # More? east → Synthesis west (done branch, going right then up)
    Edge("main_more", "post_synthesis", from_side="e", to_side="w",
         routing="custom",
         waypoints=((760, 280), (760, 48)),
         label="done", label_pos=(640, 295)),

    # Journal State east → Synthesis west (dashed, mostly straight)
    Edge("main_journal_state", "post_synthesis",
         from_side="e", to_side="w", style="dashed", routing="straight"),

    # Synthesis → Answer → Response → Evaluate → Lessons (vertical chain)
    Edge("post_synthesis", "post_answer", from_side="s", to_side="n", routing="straight"),
    Edge("post_answer",    "post_response", from_side="s", to_side="n", routing="straight"),
    Edge("post_response",  "post_evaluate", from_side="s", to_side="n", routing="straight"),
    Edge("post_evaluate",  "post_lessons",  from_side="s", to_side="n", routing="straight"),

    # Lessons → Strategy (dashed feedback loop, goes way left then up)
    Edge("post_lessons", "pre_strategy_inject", from_side="w", to_side="s",
         style="dashed", routing="custom",
         waypoints=((380, 316),)),
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
        if side == "n": return (n.cx, n.cy - hh)
        if side == "s": return (n.cx, n.cy + hh)
        if side == "e": return (n.cx + hw, n.cy)
        if side == "w": return (n.cx - hw, n.cy)
        # Diagonal sides: roughly half-way along each face.
        if side == "ne": return (n.cx + hw * 0.5, n.cy - hh * 0.5)
        if side == "nw": return (n.cx - hw * 0.5, n.cy - hh * 0.5)
        if side == "se": return (n.cx + hw * 0.5, n.cy + hh * 0.5)
        if side == "sw": return (n.cx - hw * 0.5, n.cy + hh * 0.5)
    # Rectangles (box / smallbox)
    if side == "n": return (n.cx, n.cy - hh)
    if side == "s": return (n.cx, n.cy + hh)
    if side == "e": return (n.cx + hw, n.cy)
    if side == "w": return (n.cx - hw, n.cy)
    if side == "ne": return (n.cx + hw, n.cy - hh)
    if side == "nw": return (n.cx - hw, n.cy - hh)
    if side == "se": return (n.cx + hw, n.cy + hh)
    if side == "sw": return (n.cx - hw, n.cy + hh)
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

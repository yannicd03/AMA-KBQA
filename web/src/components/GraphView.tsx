// vis-network canvas for the explored subgraph.
//
// The Network and its two DataSets are created once per mount. Every SSE
// snapshot is applied with DataSet.update/remove, never by rebuilding the
// network, so nodes keep their positions while the graph grows. Positions are
// also remembered per run (module-level, survives the panel being closed), so
// switching runs in the inspector and back restores the layout.
import { useEffect, useRef } from "react";
import { DataSet, Network } from "vis-network/standalone";
import type { Edge as VisEdge, Node as VisNode, Options } from "vis-network/standalone";
import type { GraphEdge, GraphNode } from "../api";
import { DARK_PALETTE, LIGHT_PALETTE, type GraphPalette } from "../lib/graphPalette";
import { titleElement } from "../lib/titleHtml";

type Pos = { x: number; y: number };
const savedPositions = new Map<string, Record<string, Pos>>();

const FONT_FACE = '"Libre Franklin Variable", "Libre Franklin", system-ui, sans-serif';

const tipCache = new Map<string, HTMLElement | undefined>();
function tip(fragment: string): HTMLElement | undefined {
  if (!fragment) return undefined;
  if (tipCache.size > 4000) tipCache.clear();
  if (!tipCache.has(fragment)) tipCache.set(fragment, titleElement(fragment));
  return tipCache.get(fragment);
}

function styleNode(n: GraphNode, pal: GraphPalette, highlight: Set<string>, pos?: Pos): VisNode {
  const group = n.group in pal.groups ? n.group : "entity";
  const isAnswer = n.highlighted || highlight.has(n.id);
  const base = isAnswer ? pal.answer : pal.groups[group];
  const literal = group === "literal";
  const node: VisNode = {
    id: n.id,
    label: n.label,
    title: tip(n.title) as unknown as string,
    color: {
      background: base.background,
      border: n.new && !isAnswer ? pal.newBorder : base.border,
      highlight: { background: base.background, border: pal.accent },
      hover: { background: base.background, border: pal.accent },
    },
    font: {
      color: literal ? pal.fontLiteral : pal.font,
      size: literal ? 11 : 13,
      face: FONT_FACE,
      strokeWidth: literal ? 0 : 3,
      strokeColor: pal.canvas,
    },
    borderWidth: isAnswer ? 4 : n.new ? 3 : 2,
    shape: literal ? "box" : "dot",
    size: literal ? 8 : 14,
    shapeProperties: { borderDashes: group === "candidate" ? [4, 3] : false, borderRadius: 4 },
  };
  if (pos) {
    node.x = pos.x;
    node.y = pos.y;
  }
  return node;
}

function styleEdge(e: GraphEdge, pal: GraphPalette): VisEdge {
  return {
    id: e.id,
    from: e.from,
    to: e.to,
    label: e.label,
    title: tip(e.title) as unknown as string,
    arrows: "to",
    color: { color: pal.edge, highlight: pal.accent, hover: pal.accent },
    font: {
      color: pal.edgeFont,
      size: 10,
      face: FONT_FACE,
      strokeWidth: 3,
      strokeColor: pal.canvas,
      align: "middle",
    },
    smooth: { enabled: true, type: "continuous", roundness: 0.5 },
  };
}

const OPTIONS: Options = {
  autoResize: true,
  width: "100%",
  height: "100%",
  physics: {
    // Few iterations: the graph grows while the agent works, so a long
    // stabilisation run would only burn CPU between snapshots.
    stabilization: { iterations: 80, fit: false },
    barnesHut: { gravitationalConstant: -8000, springLength: 110 },
  },
  interaction: {
    hover: true,
    navigationButtons: false,
    tooltipDelay: 120,
    // Bound to the canvas, not the window: arrow keys must keep working in
    // the question box while the graph is on screen.
    keyboard: { enabled: true, bindToWindow: false },
  },
  edges: { width: 1.2 },
};

export interface GraphViewProps {
  runKey: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  highlight: string[];
  dark: boolean;
  reducedMotion: boolean;
  onReady?: (controls: { fit: () => void }) => void;
}

export default function GraphView({ runKey, nodes, edges, highlight, dark, reducedMotion, onReady }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const netRef = useRef<Network | null>(null);
  const nodesRef = useRef(new DataSet<VisNode>());
  const edgesRef = useRef(new DataSet<VisEdge>());
  const keyRef = useRef<string | null>(null);
  const userMovedRef = useRef(false);
  const motionRef = useRef(reducedMotion);
  motionRef.current = reducedMotion;

  // fit() leaves no margin, so labels on the outermost nodes get clipped.
  // Compute the fitted view, then settle 10% further out.
  const fit = () => {
    const net = netRef.current;
    if (!net) return;
    const from = { position: net.getViewPosition(), scale: net.getScale() };
    net.fit({ animation: false });
    const to = { position: net.getViewPosition(), scale: net.getScale() * 0.9 };
    if (motionRef.current) {
      net.moveTo({ ...to, animation: false });
      return;
    }
    net.moveTo({ ...from, animation: false });
    net.moveTo({ ...to, animation: { duration: 350, easingFunction: "easeInOutQuad" } });
  };

  const remember = () => {
    const net = netRef.current;
    if (net && keyRef.current) savedPositions.set(keyRef.current, net.getPositions() as Record<string, Pos>);
  };

  // Create the network once.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const net = new Network(container, { nodes: nodesRef.current, edges: edgesRef.current }, OPTIONS);
    netRef.current = net;
    const markMoved = () => {
      userMovedRef.current = true;
    };
    net.on("dragEnd", (p: { nodes?: unknown[] }) => {
      markMoved();
      if (p?.nodes?.length) remember();
    });
    net.on("zoom", markMoved);
    net.on("stabilized", () => {
      remember();
      if (!userMovedRef.current) fit();
    });

    // The panel can be hidden (tab switch), resized, or turn into a
    // full-screen sheet on a phone. Redraw at the new size and, unless the
    // reader has panned or zoomed, frame the graph again (debounced, so a
    // window drag does not queue dozens of fits).
    let refitTimer: number | undefined;
    const ro = new ResizeObserver(() => {
      net.redraw();
      if (container.clientWidth === 0 || userMovedRef.current) return;
      window.clearTimeout(refitTimer);
      refitTimer = window.setTimeout(() => {
        if (!userMovedRef.current) fit();
      }, 150);
    });
    ro.observe(container);
    onReady?.({ fit: () => {
      userMovedRef.current = false;
      fit();
    } });

    return () => {
      window.clearTimeout(refitTimer);
      remember();
      ro.disconnect();
      net.destroy();
      netRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply each payload in place.
  useEffect(() => {
    const nodeSet = nodesRef.current;
    const edgeSet = edgesRef.current;
    let switched = false;
    if (keyRef.current !== runKey) {
      remember();
      nodeSet.clear();
      edgeSet.clear();
      keyRef.current = runKey;
      userMovedRef.current = false;
      switched = true;
    }
    const pal = dark ? DARK_PALETTE : LIGHT_PALETTE;
    const hl = new Set(highlight);
    const saved = savedPositions.get(runKey);

    const nodeIds = new Set(nodes.map((n) => n.id));
    const staleNodes = nodeSet.getIds().filter((id) => !nodeIds.has(String(id)));
    if (staleNodes.length) nodeSet.remove(staleNodes);
    nodeSet.update(
      nodes.map((n) => styleNode(n, pal, hl, nodeSet.get(n.id) ? undefined : saved?.[n.id])),
    );

    const edgeIds = new Set(edges.map((e) => e.id));
    const staleEdges = edgeSet.getIds().filter((id) => !edgeIds.has(String(id)));
    if (staleEdges.length) edgeSet.remove(staleEdges);
    edgeSet.update(edges.map((e) => styleEdge(e, pal)));

    // A different run must not inherit the previous run's pan and zoom. With
    // a remembered layout, frame it right away; otherwise centre on the
    // origin, where vis-network spawns unpositioned nodes, and let the
    // "stabilized" handler fit once physics settles.
    const net = netRef.current;
    if (switched && net) {
      if (saved && nodes.length) net.fit({ animation: false });
      else net.moveTo({ position: { x: 0, y: 0 }, scale: 1, animation: false });
    }
  }, [runKey, nodes, edges, highlight, dark]);

  return (
    <div
      ref={containerRef}
      className="graph-canvas__net"
      role="img"
      aria-label={`Explored subgraph: ${nodes.length} nodes and ${edges.length} edges. Drag to pan, scroll to zoom.`}
    />
  );
}

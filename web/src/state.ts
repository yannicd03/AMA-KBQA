// Client-side view of a run, folded from SSE snapshots and the final event.
import type {
  DoneEvent,
  ErrorEvent,
  Figure,
  GraphEdge,
  GraphNode,
  GraphStats,
  Snapshot,
  SubagentPane,
  Tokens,
} from "./api";

export type RunStatus = "loading" | "running" | "done" | "error" | "lost";

export interface RunInfo {
  runId: string;
  question: string;
  agent: string;
  startedAt: number;
}

export interface RunView extends RunInfo {
  status: RunStatus;
  continuation: boolean;
  connected: boolean;
  // live
  stage: string | null;
  elapsedS: number;
  /** client clock when `elapsedS` was received, to keep the timer moving */
  elapsedAt: number;
  spanCount: number;
  figure: Figure | null;
  subagents: SubagentPane[];
  logHtml: string;
  // graph (full node/edge set; each non-null payload replaces the previous)
  hasGraph: boolean;
  graphNodes: GraphNode[];
  graphEdges: GraphEdge[];
  graphStats: GraphStats | null;
  highlight: string[];
  answerNodes: string[];
  // final
  answer: string | null;
  durationS: number | null;
  tokens: Tokens | null;
  cost: string | null;
  model: string | null;
  traceId: string | null;
  errorMessage: string | null;
}

export type Turn =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; runId: string }
  | { kind: "notice"; id: string; text: string };

export function newRunView(info: RunInfo, status: RunStatus, continuation = false): RunView {
  return {
    ...info,
    status,
    continuation,
    connected: true,
    stage: null,
    elapsedS: 0,
    elapsedAt: Date.now(),
    spanCount: 0,
    figure: null,
    subagents: [],
    logHtml: "",
    hasGraph: false,
    graphNodes: [],
    graphEdges: [],
    graphStats: null,
    highlight: [],
    answerNodes: [],
    answer: null,
    durationS: null,
    tokens: null,
    cost: null,
    model: null,
    traceId: null,
    errorMessage: null,
  };
}

export function applySnapshot(run: RunView, s: Snapshot): RunView {
  const next: RunView = {
    ...run,
    status: "running",
    connected: true,
    stage: s.stage,
    elapsedS: s.elapsed_s,
    elapsedAt: Date.now(),
    spanCount: s.span_count,
    figure: s.figure ?? run.figure,
    subagents: s.subagents ?? run.subagents,
  };
  if (s.log_html !== null && s.log_html !== undefined) next.logHtml = s.log_html;
  if (s.graph) {
    next.hasGraph = true;
    next.graphNodes = s.graph.nodes;
    next.graphEdges = s.graph.edges;
    next.graphStats = s.graph.stats;
  }
  return next;
}

export function applyDone(run: RunView, d: DoneEvent): RunView {
  const next: RunView = {
    ...run,
    status: "done",
    connected: true,
    answer: d.answer,
    durationS: d.duration_s,
    tokens: d.tokens,
    cost: d.cost,
    model: d.model,
    traceId: d.trace_id,
    figure: d.figure ?? run.figure,
    subagents: d.subagents ?? run.subagents,
    logHtml: d.log_html ?? run.logHtml,
  };
  if (d.graph) {
    next.hasGraph = true;
    next.graphNodes = d.graph.nodes;
    next.graphEdges = d.graph.edges;
    next.graphStats = d.graph.stats;
    next.highlight = d.graph.highlight ?? [];
    next.answerNodes = d.graph.answer_nodes ?? [];
  } else {
    next.hasGraph = false;
    next.graphNodes = [];
    next.graphEdges = [];
    next.graphStats = null;
  }
  return next;
}

export function applyError(run: RunView, e: ErrorEvent): RunView {
  return {
    ...run,
    status: "error",
    connected: true,
    errorMessage: e.message || "Unknown error",
    logHtml: e.log_html ?? run.logHtml,
  };
}

/** Entity / literal / candidate counts, like GraphData.stats() in Python. */
export function graphCounts(nodes: GraphNode[]): { entities: number; literals: number; candidates: number } {
  let entities = 0;
  let literals = 0;
  let candidates = 0;
  for (const n of nodes) {
    if (n.group === "literal") literals += 1;
    else if (n.group === "candidate") candidates += 1;
    else entities += 1;
  }
  return { entities, literals, candidates };
}

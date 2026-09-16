// Typed client for the AMA-KBQA demo API (FastAPI, ama_kbqa/api/).
//
// The contract is shared with the backend; keep the shapes below in sync with
// it. Every request is same-origin under /api (nginx in the container, the
// Vite proxy in dev). With `VITE_MOCK=1` the dev server swaps in fixtures from
// ./mock; `__MOCK__` is a compile-time constant, so the mock is dropped from
// production builds entirely.

// ── Contract types ──────────────────────────────────────────────────────────

export interface AgentMeta {
  name: string;
  tagline: string;
  description: string;
  databases: string;
  tools: string;
  orchestrator: boolean;
  federated: boolean;
  followups: boolean;
}

export interface Suggestion {
  label: string;
  icon: string | null;
  color: string | null;
  question: string;
}

export interface ModelMeta {
  /** Opaque id (e.g. "kit.mistral-small-4-119b-a8b", "llamacpp:qwen3.6-35b"); sent back unchanged. */
  id: string;
  name: string;
  price: string | null;
  /** Present on provider-aware builds (demo-booth, demo-llamacpp). */
  provider?: string;
}

export interface DemoLimits {
  enabled: boolean;
  max_queries_per_session: number;
  min_seconds_between_queries: number;
}

export interface Meta {
  title: string;
  agents: AgentMeta[];
  default_agent: string;
  suggestions: Record<string, Suggestion[]>;
  models: ModelMeta[];
  default_model: string;
  default_temperature: number;
  live_graph: boolean;
  demo: DemoLimits;
  /** e.g. "The local llama.cpp server is not reachable". */
  model_notices?: string[];
}

export interface CreateRunBody {
  session_id: string;
  question: string;
  agent: string;
  model: string;
  temperature: number;
}

export interface CreateRunResponse {
  run_id: string;
  continuation: boolean;
}

export interface Figure {
  kind: "orchestrator" | "lifecycle";
  svg: string;
}

export interface SubagentPane {
  id: string;
  display: string;
  status: string;
  svg: string;
}

export type NodeGroup = "kqapro" | "sciqa" | "entity" | "literal" | "candidate";

export interface GraphNode {
  id: string;
  label: string;
  title: string;
  group: NodeGroup;
  highlighted: boolean;
  new: boolean;
}

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  label: string;
  title: string;
}

export interface GraphStats {
  nodes: number;
  edges: number;
  truncated: number;
  // The backend also sends GraphData.stats() counts; optional in the contract.
  entities?: number;
  literals?: number;
  candidates?: number;
}

export interface LiveGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  new_ids: string[];
  stats: GraphStats;
}

export interface FinalGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: GraphStats;
  highlight: string[];
  answer_nodes: string[];
}

export interface Snapshot {
  seq: number;
  status: string;
  stage: string | null;
  elapsed_s: number;
  span_count: number;
  figure: Figure | null;
  subagents: SubagentPane[];
  /** null = unchanged since the previous snapshot on this connection. */
  log_html: string | null;
  /** null = live graph disabled, or unchanged on this connection. */
  graph: LiveGraph | null;
}

export interface Tokens {
  prompt: number;
  completion: number;
  total: number;
}

export interface DoneEvent {
  run_id: string;
  status: "done";
  agent: string;
  question: string;
  answer: string;
  duration_s: number;
  tokens: Tokens | null;
  cost: string | null;
  model: string | null;
  trace_id: string | null;
  log_html: string | null;
  figure: Figure | null;
  subagents: SubagentPane[];
  graph: FinalGraph | null;
}

export interface ErrorEvent {
  run_id: string;
  status: "error";
  message: string;
  log_html: string | null;
}

export type RunRecord = {
  run_id: string;
  session_id: string;
  agent: string;
  question: string;
  status: "running" | "done" | "error";
  started_at: number | string;
} & Partial<Omit<DoneEvent, "status" | "run_id" | "agent" | "question">> &
  Partial<Pick<ErrorEvent, "message">>;

export interface TraceEvent {
  trace_id: string;
  span_id: string;
  parent_span_id: string | null;
  kind: string;
  name: string;
  start_time_unix_nano: number;
  end_time_unix_nano: number;
  duration_ms: number;
  status: string;
  is_event: boolean;
  attributes: Record<string, unknown>;
  payload: Record<string, unknown>;
  error?: string | null;
}

export interface TraceSummary {
  total_duration_ms: number;
  total_tokens: number;
  n_llm_calls: number;
  n_tool_calls: number;
  n_errors: number;
  kind_counts: Record<string, number>;
}

export interface TraceResponse {
  trace_id: string | null;
  summary: TraceSummary;
  events: TraceEvent[];
}

// ── Errors ──────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// ── Streaming ───────────────────────────────────────────────────────────────

export interface RunHandlers {
  onSnapshot: (s: Snapshot) => void;
  onDone: (d: DoneEvent) => void;
  onError: (e: ErrorEvent) => void;
  /** The run is unknown to the server (evicted, or the server restarted). */
  onLost: (reason: string) => void;
  /** false while EventSource is reconnecting after a transport error. */
  onConnection?: (connected: boolean) => void;
}

export interface ApiImpl {
  getMeta(): Promise<Meta>;
  createRun(body: CreateRunBody): Promise<CreateRunResponse>;
  resetSession(sessionId: string): Promise<void>;
  getRun(runId: string): Promise<RunRecord>;
  getTrace(runId: string): Promise<TraceResponse>;
  /** Subscribe to a run's events; returns an unsubscribe function. */
  subscribeRun(runId: string, handlers: RunHandlers): () => void;
}

// ── Real implementation ─────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}) },
    });
  } catch {
    throw new ApiError(0, "The demo server can't be reached. Check your connection and try again.");
  }
  if (res.status === 204) return undefined as T;
  let body: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body && typeof (body as { detail: unknown }).detail === "string"
        ? (body as { detail: string }).detail
        : `Request failed (HTTP ${res.status}).`;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

function recordToDone(r: RunRecord): DoneEvent {
  return {
    run_id: r.run_id,
    status: "done",
    agent: r.agent,
    question: r.question,
    answer: r.answer ?? "",
    duration_s: r.duration_s ?? 0,
    tokens: r.tokens ?? null,
    cost: r.cost ?? null,
    model: r.model ?? null,
    trace_id: r.trace_id ?? null,
    log_html: r.log_html ?? null,
    figure: r.figure ?? null,
    subagents: r.subagents ?? [],
    graph: r.graph ?? null,
  };
}

function recordToError(r: RunRecord): ErrorEvent {
  return { run_id: r.run_id, status: "error", message: r.message ?? "Unknown error", log_html: r.log_html ?? null };
}

const realApi: ApiImpl = {
  getMeta: () => request<Meta>("/api/meta"),
  createRun: (body) => request<CreateRunResponse>("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  resetSession: (sessionId) =>
    request<void>(`/api/sessions/${encodeURIComponent(sessionId)}/reset`, { method: "POST" }),
  getRun: (runId) => request<RunRecord>(`/api/runs/${encodeURIComponent(runId)}`),
  getTrace: (runId) => request<TraceResponse>(`/api/runs/${encodeURIComponent(runId)}/trace`),

  subscribeRun(runId, h) {
    let finished = false;
    let es: EventSource | null = null;
    let retryTimer: number | undefined;

    const finish = () => {
      finished = true;
      es?.close();
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    };

    // EventSource gave up (a non-200 answer on reconnect, e.g. the run was
    // evicted). Ask the run record what happened instead of guessing.
    const recover = async () => {
      try {
        const rec = await realApi.getRun(runId);
        if (finished) return;
        if (rec.status === "done") {
          finish();
          h.onDone(recordToDone(rec));
        } else if (rec.status === "error") {
          finish();
          h.onError(recordToError(rec));
        } else {
          retryTimer = window.setTimeout(connect, 2000);
        }
      } catch (err) {
        if (finished) return;
        if (err instanceof ApiError && err.status === 404) {
          finish();
          h.onLost("The server no longer has this run. It may have restarted.");
        } else {
          retryTimer = window.setTimeout(connect, 3000);
        }
      }
    };

    const connect = () => {
      if (finished) return;
      es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
      es.onopen = () => h.onConnection?.(true);
      es.addEventListener("snapshot", (e) => {
        h.onConnection?.(true);
        h.onSnapshot(JSON.parse((e as MessageEvent<string>).data) as Snapshot);
      });
      es.addEventListener("done", (e) => {
        const data = JSON.parse((e as MessageEvent<string>).data) as DoneEvent;
        finish();
        h.onDone(data);
      });
      // The server's named `error` event shares its type with EventSource's
      // own transport-error event. Only the server's carries data.
      es.addEventListener("error", (e) => {
        if (e instanceof MessageEvent && typeof e.data === "string" && e.data) {
          const data = JSON.parse(e.data) as ErrorEvent;
          finish();
          h.onError(data);
          return;
        }
        if (finished) return;
        h.onConnection?.(false);
        if (es && es.readyState === EventSource.CLOSED) {
          void recover();
        }
        // readyState CONNECTING: the browser retries on its own, and the
        // server answers a reconnect with a full snapshot.
      });
    };

    connect();
    return finish;
  },
};

// ── Implementation switch ───────────────────────────────────────────────────

let implPromise: Promise<ApiImpl> | null = null;

function impl(): Promise<ApiImpl> {
  if (!implPromise) {
    implPromise = __MOCK__ ? import("./mock/mockApi").then((m) => m.mockApi) : Promise.resolve(realApi);
  }
  return implPromise;
}

export const api = {
  getMeta: () => impl().then((i) => i.getMeta()),
  createRun: (body: CreateRunBody) => impl().then((i) => i.createRun(body)),
  resetSession: (sessionId: string) => impl().then((i) => i.resetSession(sessionId)),
  getRun: (runId: string) => impl().then((i) => i.getRun(runId)),
  getTrace: (runId: string) => impl().then((i) => i.getTrace(runId)),
  subscribeRun(runId: string, handlers: RunHandlers): () => void {
    let cancelled = false;
    let unsubscribe: (() => void) | null = null;
    void impl().then((i) => {
      if (!cancelled) unsubscribe = i.subscribeRun(runId, handlers);
    });
    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  },
  recordToDone,
  recordToError,
};

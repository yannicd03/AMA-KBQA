// Span tree for a finished run, built client-side from /api/runs/{id}/trace
// (parent_span_id chain, siblings ordered by start time, as in
// trace_render.build_tree). Clicking a span shows its attributes and payload.
import { useEffect, useMemo, useState } from "react";
import { api, ApiError, type TraceEvent, type TraceResponse } from "../api";
import type { RunStatus } from "../state";
import { formatDuration, formatInt } from "../lib/format";
import { IconChevronRight } from "./icons";

const cache = new Map<string, TraceResponse>();

interface Tree {
  roots: TraceEvent[];
  children: Map<string, TraceEvent[]>;
  byId: Map<string, TraceEvent>;
}

function buildTree(events: TraceEvent[]): Tree {
  const byId = new Map(events.map((e) => [e.span_id, e]));
  const children = new Map<string, TraceEvent[]>();
  const roots: TraceEvent[] = [];
  for (const e of events) {
    const parent = e.parent_span_id && byId.has(e.parent_span_id) ? e.parent_span_id : null;
    if (parent === null) roots.push(e);
    else {
      const list = children.get(parent) ?? [];
      list.push(e);
      children.set(parent, list);
    }
  }
  const byStart = (a: TraceEvent, b: TraceEvent) => (a.start_time_unix_nano ?? 0) - (b.start_time_unix_nano ?? 0);
  roots.sort(byStart);
  children.forEach((l) => l.sort(byStart));
  return { roots, children, byId };
}

function tokenBadge(attrs: Record<string, unknown>): string {
  const pt = attrs.prompt_tokens;
  const ct = attrs.completion_tokens;
  if (pt == null && ct == null) return "";
  return [pt != null ? `↑${pt}` : "", ct != null ? `↓${ct}` : ""].filter(Boolean).join(" ");
}

function Json({ value }: { value: unknown }) {
  let text: string;
  try {
    text = JSON.stringify(value, null, 2) ?? "null";
  } catch {
    text = String(value);
  }
  return <pre className="code">{text}</pre>;
}

export function TracePanel({ runId, status }: { runId: string; status: RunStatus }) {
  // A cancelled run is finished too: its partial trace is already frozen.
  const finished = status === "done" || status === "error" || status === "cancelled";
  const [data, setData] = useState<TraceResponse | null>(() => cache.get(runId) ?? null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(null);
    const cached = cache.get(runId);
    setData(cached ?? null);
    if (!finished || cached) return;
    let alive = true;
    api
      .getTrace(runId)
      .then((d) => {
        cache.set(runId, d);
        if (alive) setData(d);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(
          e instanceof ApiError && e.status === 409
            ? "Trace details will appear here once the run completes."
            : e instanceof Error
              ? e.message
              : "The trace could not be loaded.",
        );
      });
    return () => {
      alive = false;
    };
  }, [runId, finished, reloadKey]);

  if (status === "running" || status === "loading") {
    return <p className="empty">Trace details will appear here once the run completes.</p>;
  }
  if (status === "lost") return <p className="empty">This run is no longer on the server.</p>;
  if (error) {
    return (
      <div className="empty">
        <p>{error}</p>
        <button type="button" className="btn btn--soft" onClick={() => setReloadKey((k) => k + 1)}>
          Try again
        </button>
      </div>
    );
  }
  if (!data) return <p className="empty">Loading trace…</p>;
  return <TraceView key={runId} data={data} />;
}

function TraceView({ data }: { data: TraceResponse }) {
  const tree = useMemo(() => buildTree(data.events), [data]);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [selected, setSelected] = useState<string | null>(() => tree.roots[0]?.span_id ?? data.events[0]?.span_id ?? null);
  const s = data.summary;

  const rows: { e: TraceEvent; depth: number }[] = [];
  const walk = (e: TraceEvent, depth: number) => {
    rows.push({ e, depth });
    if (collapsed.has(e.span_id)) return;
    for (const c of tree.children.get(e.span_id) ?? []) walk(c, depth + 1);
  };
  tree.roots.forEach((r) => walk(r, 0));

  const toggle = (id: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const parents = [...tree.children.keys()];
  const evt = selected ? tree.byId.get(selected) : undefined;

  return (
    <div className="trace">
      <dl className="trace-summary">
        <div>
          <dt>Total</dt>
          <dd>{formatDuration(s.total_duration_ms)}</dd>
        </div>
        <div>
          <dt>LLM calls</dt>
          <dd>{s.n_llm_calls}</dd>
        </div>
        <div>
          <dt>Tool calls</dt>
          <dd>{s.n_tool_calls}</dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>{formatInt(s.total_tokens)}</dd>
        </div>
        <div>
          <dt>Errors</dt>
          <dd className={s.n_errors ? "is-error" : undefined}>{s.n_errors}</dd>
        </div>
      </dl>

      {data.events.length === 0 ? (
        <p className="empty">This trace has no recorded events.</p>
      ) : (
        <>
          <div className="trace__toolbar">
            <h3 className="panel-subtitle">Spans</h3>
            <div className="trace__toolbar-actions">
              <button type="button" className="link-btn" onClick={() => setCollapsed(new Set())}>
                Expand all
              </button>
              <button type="button" className="link-btn" onClick={() => setCollapsed(new Set(parents))}>
                Collapse all
              </button>
            </div>
          </div>
          <ul className="trace-tree" aria-label="Spans">
            {rows.map(({ e, depth }) => {
              const hasKids = (tree.children.get(e.span_id)?.length ?? 0) > 0;
              const isOpen = !collapsed.has(e.span_id);
              const badge = tokenBadge(e.attributes ?? {});
              return (
                <li
                  key={e.span_id}
                  className={`span-row${selected === e.span_id ? " is-selected" : ""}${e.is_event ? " is-event" : ""}${e.status === "error" ? " is-error" : ""}`}
                  style={{ paddingLeft: 4 + depth * 14 }}
                >
                  {hasKids ? (
                    <button
                      type="button"
                      className="span-row__toggle"
                      aria-expanded={isOpen}
                      aria-label={`${isOpen ? "Collapse" : "Expand"} ${e.kind} ${e.name}`}
                      onClick={() => toggle(e.span_id)}
                    >
                      <IconChevronRight size={14} />
                    </button>
                  ) : (
                    <span className="span-row__toggle span-row__toggle--leaf" aria-hidden="true" />
                  )}
                  <button
                    type="button"
                    className="span-row__main"
                    aria-pressed={selected === e.span_id}
                    onClick={() => setSelected(e.span_id)}
                  >
                    <span className={`span-pill kind-${e.kind.replace(/[^a-z_]/gi, "")}`}>{e.kind}</span>
                    <span className="span-row__name">{String(e.name ?? "").slice(0, 90)}</span>
                    {e.status === "error" && <span className="span-row__err" aria-label="error" />}
                    {badge && <span className="span-row__tokens">{badge}</span>}
                    {!e.is_event && <span className="span-row__dur">{formatDuration(e.duration_ms ?? 0)}</span>}
                  </button>
                </li>
              );
            })}
          </ul>
          {evt ? <SpanDetail evt={evt} /> : <p className="empty">Select a span to inspect.</p>}
        </>
      )}
    </div>
  );
}

function SpanDetail({ evt }: { evt: TraceEvent }) {
  const attrs = evt.attributes ?? {};
  const payload = evt.payload ?? {};
  const pt = attrs.prompt_tokens as number | undefined;
  const ct = attrs.completion_tokens as number | undefined;
  const isLlm = evt.kind === "llm_call" || evt.kind === "classify" || evt.kind === "synthesis";
  const messages = Array.isArray(payload.messages) ? (payload.messages as Record<string, unknown>[]) : null;

  return (
    <section className="span-detail" aria-label="Span details">
      <h3 className="span-detail__title">
        <span className={`span-pill kind-${evt.kind.replace(/[^a-z_]/gi, "")}`}>{evt.kind}</span>
        <span>{evt.name}</span>
      </h3>
      <dl className="span-detail__meta">
        <div>
          <dt>Duration</dt>
          <dd>{evt.is_event ? "event" : formatDuration(evt.duration_ms ?? 0)}</dd>
        </div>
        <div>
          <dt>Status</dt>
          <dd className={evt.status === "error" ? "is-error" : "is-ok"}>{evt.status === "ok" ? "OK" : evt.status}</dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>{pt != null || ct != null ? `↑${pt ?? 0} / ↓${ct ?? 0}` : "none"}</dd>
        </div>
        <div>
          <dt>Span ID</dt>
          <dd className="mono">{evt.span_id.slice(0, 12)}</dd>
        </div>
      </dl>

      {evt.error && <p className="error-card error-card--inline">{evt.error}</p>}

      {isLlm && messages && (
        <div className="span-section">
          <h4>Messages</h4>
          {messages.map((m, i) => {
            const role = String(m.role ?? "?");
            return (
              <details key={i} className="disclosure disclosure--msg" open={role === "user" || role === "assistant"}>
                <summary>
                  <IconChevronRight size={14} className="disclosure__chevron" />
                  <span className="disclosure__label">{role}</span>
                </summary>
                <div className="disclosure__body">
                  {typeof m.content === "string" ? <pre className="code">{m.content}</pre> : <Json value={m.content} />}
                  {"tool_calls" in m && (
                    <>
                      <p className="small muted">tool_calls</p>
                      <Json value={m.tool_calls} />
                    </>
                  )}
                </div>
              </details>
            );
          })}
        </div>
      )}
      {isLlm && !messages && typeof payload.assistant_content === "string" && (
        <div className="span-section">
          <h4>Assistant content</h4>
          <pre className="code">{payload.assistant_content}</pre>
        </div>
      )}

      {evt.kind === "tool_call" && (
        <div className="span-section">
          <h4>Arguments</h4>
          <Json value={payload.arguments ?? {}} />
          <h4>Result</h4>
          {typeof payload.result === "string" && payload.result.length > 4000 && (
            <p className="small muted">Showing the first 4,000 characters.</p>
          )}
          <pre className="code">{String(payload.result ?? "").slice(0, 4000)}</pre>
        </div>
      )}

      <details className="disclosure">
        <summary>
          <IconChevronRight size={14} className="disclosure__chevron" />
          <span className="disclosure__label">Attributes</span>
        </summary>
        <div className="disclosure__body">
          <Json value={attrs} />
        </div>
      </details>
      <details className="disclosure">
        <summary>
          <IconChevronRight size={14} className="disclosure__chevron" />
          <span className="disclosure__label">Payload</span>
        </summary>
        <div className="disclosure__body">
          <Json value={payload} />
        </div>
      </details>
      <details className="disclosure">
        <summary>
          <IconChevronRight size={14} className="disclosure__chevron" />
          <span className="disclosure__label">Raw event</span>
        </summary>
        <div className="disclosure__body">
          <Json value={evt} />
        </div>
      </details>
    </section>
  );
}

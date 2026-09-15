// Server-rendered lifecycle / orchestrator figures. The SVG strings are built
// and escaped by lifecycle_svg.py / orchestrator_svg.py on the server, so
// injecting them is safe; the colours come from styles.css (.lifecycle-svg).
import { useEffect, useRef } from "react";
import type { Figure, SubagentPane } from "../api";

interface StatusLine {
  stage: string | null;
  elapsed: number;
  spans: number;
}

export function LifecycleFigure({ figure, status }: { figure: Figure | null; status?: StatusLine }) {
  return (
    <div className="figure">
      {figure ? (
        <div className="figure__svg" dangerouslySetInnerHTML={{ __html: figure.svg }} />
      ) : (
        <div className="figure__empty">Waiting for the first lifecycle update…</div>
      )}
      {status && (
        <dl className="figure__status">
          <div>
            <dt>Stage</dt>
            <dd>{status.stage || "starting…"}</dd>
          </div>
          <div>
            <dt>Elapsed</dt>
            <dd>{status.elapsed.toFixed(1)}s</dd>
          </div>
          <div>
            <dt>Spans</dt>
            <dd>{status.spans}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}

/** One collapsible pane per dispatched specialist (expanded while it runs). */
export function SubagentPanes({ subagents, live }: { subagents: SubagentPane[]; live: boolean }) {
  if (!subagents.length) return null;
  return (
    <div className="subagents">
      {subagents.map((s) => (
        <details key={s.id} className="disclosure disclosure--pane" open={live && s.status === "running"}>
          <summary>
            <span className="disclosure__label">{s.display}</span>
            <span className={`status-pill status-pill--${s.status === "running" ? "running" : s.status === "error" ? "error" : "done"}`}>
              {s.status}
            </span>
          </summary>
          <div className="disclosure__body">
            <div className="figure figure--flat">
              <div className="figure__svg" dangerouslySetInnerHTML={{ __html: s.svg }} />
            </div>
          </div>
        </details>
      ))}
    </div>
  );
}

/** Captured stdout, converted to coloured spans by ansi_to_html on the server. */
export function LogConsole({ html, follow = true }: { html: string; follow?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  useEffect(() => {
    const el = ref.current;
    if (el && follow && pinned.current) el.scrollTop = el.scrollHeight;
  }, [html, follow]);

  if (!html) return <p className="muted small">No output yet.</p>;
  return (
    <div
      ref={ref}
      className="console"
      tabIndex={0}
      role="log"
      aria-label="Agent log"
      onScroll={(e) => {
        const el = e.currentTarget;
        pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

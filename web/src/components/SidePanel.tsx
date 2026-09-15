// Right-hand inspector: explored subgraph, lifecycle figure, span tree.
import { lazy, Suspense, useEffect, useId, useRef } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { RunView } from "../state";
import { graphCounts } from "../state";
import { clockTime, truncate } from "../lib/format";
import { DARK_PALETTE, LEGEND, LIGHT_PALETTE } from "../lib/graphPalette";
import { useNow } from "../lib/hooks";
import { IconChevronRight, IconClose, IconFit } from "./icons";
import { LifecycleFigure, LogConsole, SubagentPanes } from "./Lifecycle";
import { TracePanel } from "./TracePanel";

const GraphView = lazy(() => import("./GraphView"));

export type PanelTab = "graph" | "lifecycle" | "trace";

interface SidePanelProps {
  runs: RunView[];
  run: RunView;
  onSelectRun: (runId: string) => void;
  liveGraph: boolean;
  tab: PanelTab;
  onTab: (t: PanelTab) => void;
  onClose: () => void;
  sheet: boolean;
  dark: boolean;
  reducedMotion: boolean;
}

const TAB_LABEL: Record<PanelTab, string> = { graph: "Subgraph", lifecycle: "Lifecycle", trace: "Trace" };

function runLabel(r: RunView): string {
  const state = r.status === "running" ? " (running)" : r.status === "error" ? " (error)" : "";
  return `${clockTime(r.startedAt)}  ${r.agent}: ${truncate(r.question, 60)}${state}`;
}

export function SidePanel(props: SidePanelProps) {
  const { runs, run, liveGraph, tab, onTab, onClose, sheet } = props;
  const tabs: PanelTab[] = liveGraph ? ["graph", "lifecycle", "trace"] : ["lifecycle", "trace"];
  const active = tabs.includes(tab) ? tab : tabs[0];
  const baseId = useId();
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const rootRef = useRef<HTMLElement>(null);
  const selectId = useId();

  // As a full-screen sheet, move focus into the panel on open.
  useEffect(() => {
    if (sheet) rootRef.current?.focus();
  }, [sheet]);

  const onTabKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const i = tabs.indexOf(active);
    let next: PanelTab | null = null;
    if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
    if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
    if (e.key === "Home") next = tabs[0];
    if (e.key === "End") next = tabs[tabs.length - 1];
    if (next) {
      e.preventDefault();
      onTab(next);
      tabRefs.current[next]?.focus();
    }
  };

  return (
    <aside
      ref={rootRef}
      className={`panel${sheet ? " panel--sheet" : ""}`}
      aria-label="Run inspector"
      tabIndex={-1}
      role={sheet ? "dialog" : undefined}
      aria-modal={sheet ? true : undefined}
    >
      <header className="panel__header">
        {runs.length > 1 ? (
          <div className="panel__select">
            <label htmlFor={selectId} className="sr-only">
              Inspect run
            </label>
            <div className="select select--compact">
              <select id={selectId} value={run.runId} onChange={(e) => props.onSelectRun(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.runId} value={r.runId}>
                    {runLabel(r)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <p className="panel__title" title={run.question}>
            {truncate(run.question, 80)}
          </p>
        )}
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close inspector" title="Close inspector">
          <IconClose size={18} />
        </button>
      </header>

      <div className="tabs" role="tablist" aria-label="Inspector views" onKeyDown={onTabKey}>
        {tabs.map((t) => (
          <button
            key={t}
            ref={(el) => {
              tabRefs.current[t] = el;
            }}
            id={`${baseId}-tab-${t}`}
            type="button"
            role="tab"
            aria-selected={active === t}
            aria-controls={`${baseId}-panel-${t}`}
            tabIndex={active === t ? 0 : -1}
            className="tab"
            onClick={() => onTab(t)}
          >
            {TAB_LABEL[t]}
          </button>
        ))}
      </div>

      {liveGraph && (
        <div
          id={`${baseId}-panel-graph`}
          role="tabpanel"
          aria-labelledby={`${baseId}-tab-graph`}
          className="tabpanel tabpanel--graph"
          hidden={active !== "graph"}
        >
          <GraphPanel run={run} dark={props.dark} reducedMotion={props.reducedMotion} />
        </div>
      )}
      {active === "lifecycle" && (
        <div id={`${baseId}-panel-lifecycle`} role="tabpanel" aria-labelledby={`${baseId}-tab-lifecycle`} className="tabpanel">
          <LifecycleTab run={run} />
        </div>
      )}
      {active === "trace" && (
        <div id={`${baseId}-panel-trace`} role="tabpanel" aria-labelledby={`${baseId}-tab-trace`} className="tabpanel">
          <TracePanel runId={run.runId} status={run.status} />
        </div>
      )}
    </aside>
  );
}

function LifecycleTab({ run }: { run: RunView }) {
  const now = useNow(run.status === "running", 250);
  if (run.status === "loading") return <p className="empty">Loading…</p>;
  if (run.status === "running") {
    const elapsed = run.elapsedS + Math.max(0, (now - run.elapsedAt) / 1000);
    return (
      <div className="stack">
        <LifecycleFigure figure={run.figure} status={{ stage: run.stage, elapsed, spans: run.spanCount }} />
        <SubagentPanes subagents={run.subagents} live />
        <details className="disclosure">
          <summary>
            <IconChevronRight size={15} className="disclosure__chevron" />
            <span className="disclosure__label">Live log</span>
          </summary>
          <div className="disclosure__body">
            <LogConsole html={run.logHtml} />
          </div>
        </details>
      </div>
    );
  }
  if (!run.figure) {
    return <p className="empty">{run.status === "error" ? "The run stopped before the lifecycle could be drawn." : "No trace selected."}</p>;
  }
  return (
    <div className="stack">
      <LifecycleFigure figure={run.figure} />
      <SubagentPanes subagents={run.subagents} live={false} />
    </div>
  );
}

function GraphPanel({ run, dark, reducedMotion }: { run: RunView; dark: boolean; reducedMotion: boolean }) {
  const controls = useRef<{ fit: () => void } | null>(null);
  const live = run.status === "running";
  const stats = run.graphStats;
  const local = graphCounts(run.graphNodes);
  const counts = {
    entities: stats?.entities ?? local.entities,
    literals: stats?.literals ?? local.literals,
    candidates: stats?.candidates ?? local.candidates,
  };
  // Two nodes can share a label; list each label once.
  const answerLabels = [...new Set(run.answerNodes)];
  const pal = dark ? DARK_PALETTE : LIGHT_PALETTE;
  const empty = run.graphNodes.length === 0;

  let caption = live ? "Building…" : run.status === "done" ? "Final" : "";
  if (stats && stats.truncated > 0) caption += ` Showing ${stats.nodes} of ${stats.nodes + stats.truncated} nodes.`;

  return (
    <div className="graph">
      <div className="graph__head">
        <div>
          <h3 className="panel-subtitle">Explored subgraph</h3>
          {caption && (
            <p className="graph__caption">
              {live && <span className="live-dot" aria-hidden="true" />}
              {caption}
            </p>
          )}
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={() => controls.current?.fit()}
          disabled={empty}
          aria-label="Fit graph to view"
          title="Fit to view"
        >
          <IconFit size={18} />
        </button>
      </div>

      <dl className="graph__stats">
        <div title={`${counts.candidates} further search candidate(s) the agent has seen but not visited.`}>
          <dt>Entities</dt>
          <dd>{counts.entities}</dd>
        </div>
        <div>
          <dt>Literals</dt>
          <dd>{counts.literals}</dd>
        </div>
        <div>
          <dt>Edges</dt>
          <dd>{run.graphEdges.length}</dd>
        </div>
      </dl>

      <div className="graph-canvas">
        <Suspense fallback={<p className="graph-canvas__empty">Loading graph view…</p>}>
          <GraphView
            runKey={run.runId}
            nodes={run.graphNodes}
            edges={run.graphEdges}
            highlight={run.highlight}
            dark={dark}
            reducedMotion={reducedMotion}
            onReady={(c) => {
              controls.current = c;
            }}
          />
        </Suspense>
        {empty && (
          <p className="graph-canvas__empty">
            {live
              ? "The graph fills in as the agent looks things up."
              : run.status === "done"
                ? "No graph data for this run."
                : run.status === "error"
                  ? "The run stopped before any graph data arrived."
                  : "No graph data for this run."}
          </p>
        )}
      </div>

      <ul className="legend" aria-label="Legend">
        {LEGEND.map((l) => {
          const sw = l.key === "answer" ? pal.answer : pal.groups[l.key];
          return (
            <li key={l.key}>
              <span
                className={`legend__swatch legend__swatch--${l.shape}`}
                style={{ background: l.shape === "dashed" ? "transparent" : sw.background, borderColor: sw.border }}
              />
              {l.label}
            </li>
          );
        })}
      </ul>

      {run.status === "done" && !empty && (
        <details className="disclosure">
          <summary>
            <IconChevronRight size={15} className="disclosure__chevron" />
            <span className="disclosure__label">Nodes in the answer</span>
            <span className="disclosure__count">{run.highlight.length}</span>
          </summary>
          <div className="disclosure__body">
            {answerLabels.length ? (
              <>
                <ul className="answer-nodes">
                  {answerLabels.map((label) => (
                    <li key={label}>{label}</li>
                  ))}
                </ul>
                {run.highlight.length > run.answerNodes.length && (
                  <p className="small muted">and {run.highlight.length - run.answerNodes.length} more</p>
                )}
              </>
            ) : (
              <p className="small muted">No node of the graph matched the answer text.</p>
            )}
          </div>
        </details>
      )}
    </div>
  );
}

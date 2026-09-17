import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { RunView, Turn } from "../state";
import { footerParts } from "../lib/format";
import { useNow } from "../lib/hooks";
import { IconAlert, IconCheck, IconChevronRight, IconCopy, IconGraph } from "./icons";
import { LifecycleFigure, LogConsole, SubagentPanes } from "./Lifecycle";
import { Mark } from "./Mark";

interface ConversationProps {
  turns: Turn[];
  runs: Record<string, RunView>;
  simplified: boolean;
  inspectAvailable: boolean;
  onInspect: (runId: string) => void;
}

export function Conversation({ turns, runs, simplified, inspectAvailable, onInspect }: ConversationProps) {
  return (
    <div className="thread">
      {turns.map((t) => {
        if (t.kind === "user") {
          return (
            <div key={t.id} className="turn turn--user">
              <p className="bubble">{t.text}</p>
            </div>
          );
        }
        if (t.kind === "notice") {
          return (
            <p key={t.id} className="notice-turn" role="status">
              {t.text}
            </p>
          );
        }
        return (
          <AssistantTurn
            key={t.id}
            run={runs[t.runId]}
            runId={t.runId}
            simplified={simplified}
            inspectAvailable={inspectAvailable}
            onInspect={onInspect}
          />
        );
      })}
    </div>
  );
}

interface AssistantTurnProps {
  run: RunView | undefined;
  runId: string;
  simplified: boolean;
  inspectAvailable: boolean;
  onInspect: (runId: string) => void;
}

function AssistantTurn({ run, runId, simplified, inspectAvailable, onInspect }: AssistantTurnProps) {
  const running = run?.status === "running";
  return (
    <article className="turn turn--assistant" data-run-id={runId} aria-busy={running || run?.status === "loading"}>
      <div className="turn__avatar">
        <Mark size={30} thinking={running} />
      </div>
      <div className="turn__body">
        {!run || run.status === "lost" ? (
          <p className="muted">{run?.errorMessage ?? "This answer is no longer available on the server."}</p>
        ) : run.status === "loading" ? (
          <p className="muted thinking-line">Loading answer…</p>
        ) : run.status === "running" ? (
          simplified ? (
            <p className="muted thinking-line">Agent is thinking…</p>
          ) : (
            <LiveReasoning run={run} />
          )
        ) : run.status === "cancelled" ? (
          <CancelledAnswer run={run} simplified={simplified} />
        ) : run.status === "error" ? (
          <ErrorAnswer run={run} simplified={simplified} />
        ) : (
          <DoneAnswer run={run} simplified={simplified} inspectAvailable={inspectAvailable} onInspect={onInspect} />
        )}
      </div>
    </article>
  );
}

function LiveReasoning({ run }: { run: RunView }) {
  const now = useNow(true, 200);
  const elapsed = run.elapsedS + Math.max(0, (now - run.elapsedAt) / 1000);
  return (
    <details className="reasoning" open>
      <summary>
        <IconChevronRight size={16} className="reasoning__chevron" />
        <span className="reasoning__title">Thinking</span>
        <span className="reasoning__stage">{run.connected ? run.stage || "starting…" : "Reconnecting…"}</span>
        <span className="reasoning__time">{elapsed.toFixed(1)}s</span>
      </summary>
      <div className="reasoning__body">
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
    </details>
  );
}

function DoneAnswer({
  run,
  simplified,
  inspectAvailable,
  onInspect,
}: {
  run: RunView;
  simplified: boolean;
  inspectAvailable: boolean;
  onInspect: (runId: string) => void;
}) {
  const parts = footerParts(run.durationS, run.tokens, run.cost);
  return (
    <>
      {!simplified && run.logHtml && (
        <details className="reasoning reasoning--done">
          <summary>
            <IconChevronRight size={16} className="reasoning__chevron" />
            <span className="reasoning__title">View reasoning trace</span>
          </summary>
          <div className="reasoning__body">
            <LogConsole html={run.logHtml} follow={false} />
          </div>
        </details>
      )}
      <div className="answer prose">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
          }}
        >
          {run.answer || "_The agent returned an empty answer._"}
        </Markdown>
      </div>
      <div className="turn__meta">
        {!simplified && parts.length > 0 && (
          <p className="footer-meta">
            {parts.map((p, i) => (
              <span key={i}>
                {i > 0 && (
                  <span className="footer-meta__sep" aria-hidden="true">
                    |
                  </span>
                )}
                {p}
              </span>
            ))}
          </p>
        )}
        <div className="turn__actions">
          <CopyButton text={run.answer ?? ""} />
          {!simplified && inspectAvailable && (
            <button type="button" className="icon-btn icon-btn--sm" onClick={() => onInspect(run.runId)} title="Inspect this run" aria-label="Inspect this run">
              <IconGraph size={16} />
            </button>
          )}
        </div>
      </div>
    </>
  );
}

// Stopped on request, so this is a neutral status rather than an error card.
function CancelledAnswer({ run, simplified }: { run: RunView; simplified: boolean }) {
  return (
    <>
      <p className="muted" role="status">
        Stopped. The agent may take a moment to finish the step it was on.
      </p>
      {!simplified && run.logHtml && (
        <details className="disclosure">
          <summary>
            <IconChevronRight size={15} className="disclosure__chevron" />
            <span className="disclosure__label">Log</span>
          </summary>
          <div className="disclosure__body">
            <LogConsole html={run.logHtml} follow={false} />
          </div>
        </details>
      )}
    </>
  );
}

function ErrorAnswer({ run, simplified }: { run: RunView; simplified: boolean }) {
  return (
    <div className="error-card" role="alert">
      <p className="error-card__title">
        <IconAlert size={18} />
        Execution error
      </p>
      <p className="error-card__message">{run.errorMessage}</p>
      {!simplified && run.logHtml && (
        <details className="disclosure">
          <summary>
            <IconChevronRight size={15} className="disclosure__chevron" />
            <span className="disclosure__label">Log</span>
          </summary>
          <div className="disclosure__body">
            <LogConsole html={run.logHtml} follow={false} />
          </div>
        </details>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  if (!text || !navigator.clipboard) return null;
  return (
    <button
      type="button"
      className="icon-btn icon-btn--sm"
      aria-label={copied ? "Copied" : "Copy answer"}
      title={copied ? "Copied" : "Copy answer"}
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
    </button>
  );
}

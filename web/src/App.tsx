import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type Meta, type Suggestion } from "./api";
import { AboutDialog } from "./components/AboutDialog";
import { Composer } from "./components/Composer";
import { Conversation } from "./components/Conversation";
import { IconAlert, IconGraph, IconHelp, IconMenu, IconNewChat, SuggestionIcon } from "./components/icons";
import { LifecycleDefs } from "./components/LifecycleDefs";
import { Mark } from "./components/Mark";
import { SidePanel, type PanelTab } from "./components/SidePanel";
import { Sidebar, type Settings } from "./components/Sidebar";
import { modelLabel } from "./lib/format";
import { useMediaQuery, usePrefersDark, useReducedMotion } from "./lib/hooks";
import { aboutSeen, getSessionId, loadState, markAboutSeen, saveState, uid } from "./lib/session";
import { applyDone, applyError, applySnapshot, newRunView, type RunInfo, type RunView, type Turn } from "./state";

interface Persisted {
  settings?: Partial<Settings>;
  turns?: Turn[];
  runs?: RunInfo[];
}

interface InlineNotice {
  tone: "error" | "warn";
  text: string;
}

// Same cap as MAX_TRACES in chat.py: the inspector keeps the last 30 runs.
const MAX_RUNS = 30;
const LOST_MESSAGE = "The server no longer has this run. It may have restarted.";

function resolveSettings(meta: Meta, saved?: Partial<Settings>): Settings {
  const names = meta.agents.map((a) => a.name);
  const fallbackAgent = names.includes(meta.default_agent) ? meta.default_agent : names[0];
  const t = saved?.temperature;
  return {
    agent: saved?.agent && names.includes(saved.agent) ? saved.agent : fallbackAgent,
    model: saved?.model && meta.models.some((m) => m.id === saved.model) ? saved.model : meta.default_model,
    temperature: typeof t === "number" && t >= 0 && t <= 2 ? t : meta.default_temperature,
    simplified: saved?.simplified ?? false,
    liveGraph: saved?.liveGraph ?? true,
  };
}

export default function App() {
  const sessionId = getSessionId();
  const persisted = useMemo(() => loadState<Persisted>(), []);

  const [meta, setMeta] = useState<Meta | null>(null);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [turns, setTurns] = useState<Turn[]>(() => persisted?.turns ?? []);
  const [runInfos, setRunInfos] = useState<RunInfo[]>(() => persisted?.runs ?? []);
  const [runs, setRuns] = useState<Record<string, RunView>>(() =>
    Object.fromEntries((persisted?.runs ?? []).map((i) => [i.runId, newRunView(i, "loading")])),
  );
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<InlineNotice | null>(null);

  const wide = useMediaQuery("(min-width: 1000px)");
  const mobile = useMediaQuery("(max-width: 759px)");
  const dark = usePrefersDark();
  const reducedMotion = useReducedMotion();

  const [sidebarExpanded, setSidebarExpanded] = useState(() => window.matchMedia("(min-width: 1200px)").matches);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [panelTab, setPanelTab] = useState<PanelTab>("graph");
  const [panelRunId, setPanelRunId] = useState<string | null>(null);
  const [aboutOpen, setAboutOpen] = useState(() => !aboutSeen());

  const subs = useRef(new Map<string, () => void>());
  const scrollRef = useRef<HTMLDivElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // ── Meta ──────────────────────────────────────────────────────────────────
  const loadMeta = useCallback(() => {
    setMetaError(null);
    api
      .getMeta()
      .then((m) => {
        setMeta(m);
        setSettings((prev) => prev ?? resolveSettings(m, persisted?.settings));
        document.title = m.title || "AMA-KBQA Assistant";
      })
      .catch((e: unknown) => setMetaError(e instanceof Error ? e.message : "The demo server can't be reached."));
  }, [persisted]);

  useEffect(() => {
    loadMeta();
  }, [loadMeta]);

  // ── Runs ──────────────────────────────────────────────────────────────────
  const patchRun = useCallback((id: string, fn: (r: RunView) => RunView) => {
    setRuns((prev) => {
      const r = prev[id];
      if (!r) return prev;
      const next = fn(r);
      return next === r ? prev : { ...prev, [id]: next };
    });
  }, []);

  const subscribe = useCallback(
    (id: string) => {
      subs.current.get(id)?.();
      const unsub = api.subscribeRun(id, {
        onSnapshot: (s) => patchRun(id, (r) => applySnapshot(r, s)),
        onDone: (d) => {
          subs.current.delete(id);
          patchRun(id, (r) => applyDone(r, d));
        },
        onError: (e) => {
          subs.current.delete(id);
          patchRun(id, (r) => applyError(r, e));
        },
        onLost: (reason) => {
          subs.current.delete(id);
          patchRun(id, (r) => ({ ...r, status: "lost", errorMessage: reason }));
        },
        onConnection: (c) => patchRun(id, (r) => (r.connected === c ? r : { ...r, connected: c })),
      });
      subs.current.set(id, unsub);
    },
    [patchRun],
  );

  useEffect(() => {
    const map = subs.current;
    return () => {
      map.forEach((u) => u());
      map.clear();
    };
  }, []);

  // After a reload, re-attach to runs from this tab's earlier life.
  const hydrated = useRef(false);
  useEffect(() => {
    if (!meta || hydrated.current) return;
    hydrated.current = true;
    for (const info of runInfos) {
      const id = info.runId;
      api
        .getRun(id)
        .then((rec) => {
          if (rec.status === "done") patchRun(id, (r) => applyDone(r, api.recordToDone(rec)));
          else if (rec.status === "error") patchRun(id, (r) => applyError(r, api.recordToError(rec)));
          else {
            patchRun(id, (r) => ({ ...r, status: "running" }));
            subscribe(id);
          }
        })
        .catch((e: unknown) =>
          patchRun(id, (r) => ({
            ...r,
            status: "lost",
            errorMessage: e instanceof ApiError && e.status !== 404 ? e.message : LOST_MESSAGE,
          })),
        );
    }
  }, [meta, runInfos, patchRun, subscribe]);

  // Persist the transcript (not the heavy run payloads; those are refetched).
  useEffect(() => {
    if (settings) saveState({ settings, turns, runs: runInfos } satisfies Persisted);
  }, [settings, turns, runInfos]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const agentMeta = meta?.agents.find((a) => a.name === settings?.agent);
  const busyRun = runInfos.map((i) => runs[i.runId]).find((r) => r && (r.status === "running" || r.status === "loading"));
  const running = Boolean(busyRun) || submitting;
  const inConversation = turns.length > 0;
  // A build may offer the live graph and still have the toggle suppressed by
  // its settings description (older backends send no `settings` at all).
  const liveGraphAvailable = Boolean(meta?.live_graph && (meta?.settings?.controls?.live_graph ?? true));
  const liveGraphOn = liveGraphAvailable && Boolean(settings?.liveGraph);
  const simplified = settings?.simplified ?? false;
  const panelAvailable = !simplified && inConversation && runInfos.length > 0;
  const orderedRuns = runInfos
    .slice()
    .reverse()
    .map((i) => runs[i.runId])
    .filter(Boolean);
  const panelRun = (panelRunId && runs[panelRunId]) || orderedRuns[0] || null;
  const showPanel = panelAvailable && panelOpen && panelRun !== null;
  const lastRun = orderedRuns[0];

  // ── Actions ───────────────────────────────────────────────────────────────
  const pushNotice = useCallback((text: string) => {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const base = prev[prev.length - 1]?.kind === "notice" ? prev.slice(0, -1) : prev;
      return [...base, { kind: "notice", id: uid(), text }];
    });
  }, []);

  const resetServer = useCallback(() => {
    api.resetSession(sessionId).catch(() => {
      /* 409 (a run in flight) or offline: the next run decides anyway */
    });
  }, [sessionId]);

  const submit = useCallback(
    async (question: string): Promise<boolean> => {
      if (!meta || !settings || running) return false;
      setSubmitting(true);
      setNotice(null);
      try {
        const res = await api.createRun({
          session_id: sessionId,
          question,
          agent: settings.agent,
          model: settings.model,
          temperature: settings.temperature,
        });
        const info: RunInfo = { runId: res.run_id, question, agent: settings.agent, startedAt: Date.now() };
        setRuns((prev) => ({ ...prev, [info.runId]: newRunView(info, "running", res.continuation) }));
        setRunInfos((prev) => [...prev, info].slice(-MAX_RUNS));
        setTurns((prev) => [
          ...prev,
          { kind: "user", id: uid(), text: question },
          { kind: "assistant", id: uid(), runId: info.runId },
        ]);
        setPanelRunId(info.runId);
        if (liveGraphOn && !settings.simplified && wide) {
          setPanelOpen(true);
          setPanelTab("graph");
          // Like a canvas opening in chat products: give the conversation the
          // room back by folding the sidebar to its icon rail.
          if (window.innerWidth < 1500) setSidebarExpanded(false);
        }
        pinned.current = true;
        subscribe(info.runId);
        return true;
      } catch (e: unknown) {
        const status = e instanceof ApiError ? e.status : 0;
        setNotice({
          tone: status === 429 ? "warn" : "error",
          text: e instanceof Error ? e.message : "The question could not be sent.",
        });
        return false;
      } finally {
        setSubmitting(false);
      }
    },
    [meta, settings, running, sessionId, liveGraphOn, wide, subscribe],
  );

  const newChat = useCallback(() => {
    if (running) return;
    setTurns([]);
    setNotice(null);
    setPanelOpen(false);
    setDrawerOpen(false);
    resetServer();
  }, [running, resetServer]);

  const changeAgent = (name: string) => {
    if (!settings || name === settings.agent) return;
    setSettings({ ...settings, agent: name });
    setNotice(null);
    if (inConversation) {
      resetServer();
      pushNotice(`Switched to ${name}. The next question starts a new conversation.`);
    }
  };

  const changeModel = (id: string) => {
    if (!settings || !meta || id === settings.model) return;
    setSettings({ ...settings, model: id });
    // A persisted multiturn agent is bound to the model it was built with.
    resetServer();
    if (inConversation) {
      const names = new Map(meta.models.map((m) => [m.id, m.name]));
      pushNotice(`Model changed to ${modelLabel(id, names)}. The next question starts a new conversation.`);
    }
  };

  const inspect = (runId: string) => {
    setPanelRunId(runId);
    setPanelOpen(true);
    if (wide && window.innerWidth < 1500) setSidebarExpanded(false);
  };

  const closeAbout = () => {
    setAboutOpen(false);
    markAboutSeen();
  };

  // ── Effects: keyboard, scrolling ──────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || aboutOpen || e.defaultPrevented) return;
      if (drawerOpen) setDrawerOpen(false);
      else if (!wide && panelOpen) setPanelOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [aboutOpen, drawerOpen, wide, panelOpen]);

  useEffect(() => {
    if (!mobile) setDrawerOpen(false);
  }, [mobile]);

  // Keep the thread pinned to the bottom while it grows, unless the reader
  // scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    const thread = threadRef.current;
    if (!el || !thread) return;
    const onScroll = () => {
      pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
    };
    const ro = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight;
    });
    el.addEventListener("scroll", onScroll, { passive: true });
    ro.observe(thread);
    return () => {
      el.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, [inConversation, meta]);

  // When an answer lands, bring the start of the assistant turn into view.
  const lastStatus = lastRun?.status;
  const lastId = lastRun?.runId;
  useEffect(() => {
    if (lastStatus !== "done" || !pinned.current || !lastId) return;
    const el = scrollRef.current?.querySelector<HTMLElement>(`[data-run-id="${lastId}"]`);
    if (el) {
      pinned.current = false;
      el.scrollIntoView({ block: "start", behavior: reducedMotion ? "auto" : "smooth" });
    }
  }, [lastStatus, lastId, reducedMotion]);

  // ── Render ────────────────────────────────────────────────────────────────
  if (metaError && !meta) {
    return (
      <main className="boot">
        <Mark size={44} />
        <h1 className="boot__title">AMA-KBQA Assistant</h1>
        <p className="boot__text">{metaError}</p>
        <button type="button" className="btn btn--primary" onClick={loadMeta}>
          Try again
        </button>
      </main>
    );
  }
  if (!meta || !settings || !agentMeta) {
    return (
      <main className="boot" aria-busy="true">
        <Mark size={44} thinking />
        <p className="boot__text">Connecting to the demo server…</p>
      </main>
    );
  }

  const suggestions: Suggestion[] = meta.suggestions[settings.agent] ?? [];
  const sidebarMode = mobile ? "drawer" : sidebarExpanded ? "full" : "rail";
  const showSidebar = !mobile || drawerOpen;
  const followups = agentMeta.followups;

  const inlineNotice = notice && (
    <div className={`inline-notice inline-notice--${notice.tone}`} role="alert">
      <IconAlert size={18} />
      <span>{notice.text}</span>
    </div>
  );

  return (
    <div className={`app${showPanel ? " app--panel" : ""}`}>
      <LifecycleDefs />
      <h1 className="sr-only">{meta.title}</h1>

      {showSidebar && (
        <>
          {mobile && <div className="scrim" onClick={() => setDrawerOpen(false)} aria-hidden="true" />}
          <Sidebar
            mode={sidebarMode}
            onToggle={() => (mobile ? setDrawerOpen(false) : setSidebarExpanded((v) => !v))}
            models={meta.models}
            modelNotices={meta.model_notices ?? []}
            settings={settings}
            settingsMeta={meta.settings}
            liveGraphAvailable={liveGraphAvailable}
            running={running}
            onModelChange={changeModel}
            onTemperatureChange={(t) => setSettings({ ...settings, temperature: t })}
            onSimplifiedChange={(v) => setSettings({ ...settings, simplified: v })}
            onLiveGraphChange={(v) => setSettings({ ...settings, liveGraph: v })}
            onNewChat={newChat}
            newChatDisabled={running || !inConversation}
            onAbout={() => {
              setDrawerOpen(false);
              setAboutOpen(true);
            }}
          />
        </>
      )}

      <main className="main">
        <header className="topbar">
          {mobile && (
            <button type="button" className="icon-btn" onClick={() => setDrawerOpen(true)} aria-label="Open sidebar">
              <IconMenu />
            </button>
          )}
          <span className="topbar__title">{meta.title}</span>
          <span className="topbar__spacer" />
          {mobile && inConversation && (
            <button type="button" className="icon-btn" onClick={newChat} disabled={running} aria-label="New chat" title="New chat">
              <IconNewChat />
            </button>
          )}
          {panelAvailable && (
            <button
              type="button"
              className={`btn btn--ghost${showPanel ? " is-active" : ""}`}
              aria-pressed={showPanel}
              onClick={() => setPanelOpen((o) => !o)}
              title={showPanel ? "Hide inspector" : "Show the explored subgraph, lifecycle and trace"}
            >
              <IconGraph size={18} />
              <span className="btn__label">Inspector</span>
              {busyRun && !showPanel && <span className="live-dot" aria-hidden="true" />}
            </button>
          )}
          {!wide && (
            <button type="button" className="icon-btn" onClick={() => setAboutOpen(true)} aria-label="About this demo" title="About this demo">
              <IconHelp />
            </button>
          )}
        </header>

        {!inConversation ? (
          <section className="landing" aria-labelledby="landing-title">
            <div className="landing__inner">
              <h2 id="landing-title" className="landing__title">
                <Mark size={40} className="landing__mark" />
                Explore the Knowledge Graph.
              </h2>
              <Composer
                variant="landing"
                agents={meta.agents}
                agent={settings.agent}
                onAgentChange={changeAgent}
                pickerDisabled={running}
                showInput
                placeholder="Ask a question…"
                inputDisabled={running}
                onSubmit={submit}
                autoFocus={!mobile && !aboutOpen}
              />
              {inlineNotice}
              {!simplified && suggestions.length > 0 && (
                <ul className="suggestions" aria-label="Example questions">
                  {suggestions.map((s) => (
                    <li key={s.label}>
                      <button
                        type="button"
                        className="suggestion"
                        data-color={s.color ?? undefined}
                        disabled={running}
                        onClick={() => void submit(s.question)}
                        title={s.question}
                      >
                        <SuggestionIcon icon={s.icon} size={18} className="suggestion__icon" />
                        {s.label}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        ) : (
          <>
            <div className="scroll" ref={scrollRef}>
              <div className="column" ref={threadRef}>
                <Conversation
                  turns={turns}
                  runs={runs}
                  simplified={simplified}
                  inspectAvailable={panelAvailable}
                  onInspect={inspect}
                />
              </div>
            </div>
            <div className="dock">
              <div className="dock__inner">
                {inlineNotice}
                <Composer
                  variant="docked"
                  agents={meta.agents}
                  agent={settings.agent}
                  onAgentChange={changeAgent}
                  pickerDisabled={running}
                  showInput={followups}
                  placeholder="Follow up…"
                  inputDisabled={running}
                  onSubmit={submit}
                  autoFocus={!mobile}
                  caption={
                    simplified ? undefined : (
                      <>
                        The Orchestrator answers one question at a time. Use <strong>New chat</strong> for a new
                        question, or pick KQAPro or SciQA in the agent menu for a follow-up conversation.
                      </>
                    )
                  }
                  onRestart={newChat}
                  restartDisabled={running}
                />
              </div>
            </div>
          </>
        )}
      </main>

      {showPanel && panelRun && (
        <SidePanel
          runs={orderedRuns}
          run={panelRun}
          onSelectRun={setPanelRunId}
          liveGraph={liveGraphOn}
          tab={panelTab}
          onTab={setPanelTab}
          onClose={() => setPanelOpen(false)}
          sheet={!wide}
          dark={dark}
          reducedMotion={reducedMotion}
        />
      )}

      {wide && (
        <button type="button" className="help-fab" onClick={() => setAboutOpen(true)} aria-label="About this demo" title="About this demo">
          ?
        </button>
      )}

      <AboutDialog open={aboutOpen} onClose={closeAbout} />
    </div>
  );
}

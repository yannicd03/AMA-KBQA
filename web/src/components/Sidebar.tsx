import { useId, type ReactNode } from "react";
import type { EndpointRow, ModelMeta, SettingsMeta } from "../api";
import { modelLabel } from "../lib/format";
import { IconAlert, IconClose, IconHelp, IconNewChat, IconSidebar, IconSliders } from "./icons";
import { Mark } from "./Mark";

export interface Settings {
  agent: string;
  model: string;
  temperature: number;
  simplified: boolean;
  liveGraph: boolean;
}

interface SidebarProps {
  mode: "full" | "rail" | "drawer";
  onToggle: () => void;
  models: ModelMeta[];
  modelNotices: string[];
  settings: Settings;
  /** What this build offers; absent on an older backend (everything shown). */
  settingsMeta?: SettingsMeta | null;
  liveGraphAvailable: boolean;
  running: boolean;
  onModelChange: (id: string) => void;
  onTemperatureChange: (t: number) => void;
  onSimplifiedChange: (v: boolean) => void;
  onLiveGraphChange: (v: boolean) => void;
  onNewChat: () => void;
  newChatDisabled: boolean;
  onAbout: () => void;
}

function Switch({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  const id = useId();
  return (
    <div className="field field--switch">
      <div className="field__row">
        <label htmlFor={id} className="field__label">
          {label}
        </label>
        <button
          id={id}
          type="button"
          role="switch"
          aria-checked={checked}
          aria-describedby={`${id}-hint`}
          className="switch"
          onClick={() => onChange(!checked)}
        >
          <span className="switch__thumb" />
        </button>
      </div>
      <p id={`${id}-hint`} className="field__hint">
        {hint}
      </p>
    </div>
  );
}

// ── Endpoints / build facts ────────────────────────────────────────────────
// Rendered from whatever /api/meta sends. Nothing here knows a provider name:
// a branch adds rows server-side and they show up. Statuses outside the four
// known ones fall back to neutral styling and their own humanised text.

const STATUS_TONES: Record<string, "ok" | "warn" | "bad" | "unknown"> = {
  ok: "ok",
  no_key: "warn",
  unreachable: "bad",
  unknown: "unknown",
};

const STATUS_TEXT: Record<string, string> = {
  ok: "Reachable",
  no_key: "No API key",
  unreachable: "Unreachable",
  unknown: "Unknown",
};

/** "no_key" → "No key"; "rate_limited" → "Rate limited". */
function humanize(value: string): string {
  const text = value.replace(/[_-]+/g, " ").trim();
  return text ? text[0].toUpperCase() + text.slice(1) : "";
}

function StatusPill({ status }: { status: string }) {
  const tone = STATUS_TONES[status] ?? "unknown";
  return (
    <span className={`status status--${tone}`}>
      <span className="status__dot" aria-hidden="true" />
      {STATUS_TEXT[status] ?? humanize(status)}
    </span>
  );
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </>
  );
}

function Endpoint({ row }: { row: EndpointRow }) {
  const key = row.api_key;
  const caption = [row.role, row.provider].filter(Boolean).join(" · ");
  return (
    <li className="endpoint">
      <div className="endpoint__head">
        <span className="endpoint__label">{row.label}</span>
        {row.status ? <StatusPill status={row.status} /> : null}
      </div>
      {caption ? <p className="endpoint__caption">{caption}</p> : null}
      <dl className="facts">
        {row.base_url ? <Fact term="URL">{row.base_url}</Fact> : null}
        {row.model ? <Fact term="Model">{row.model}</Fact> : null}
        {row.model_source ? <Fact term="Source">{row.model_source}</Fact> : null}
        {key ? (
          <Fact term="API key">
            {key.configured ? "configured" : "not configured"}
            {key.hint ? ` (${key.hint})` : ""}
          </Fact>
        ) : null}
      </dl>
      {row.detail ? <p className="field__hint">{row.detail}</p> : null}
    </li>
  );
}

export function Sidebar(props: SidebarProps) {
  const { mode, onToggle, models, settings, liveGraphAvailable, running } = props;
  const controls = props.settingsMeta?.controls ?? {};
  const showModel = controls.model ?? true;
  const showTemperature = controls.temperature ?? true;
  const showSimplified = controls.simplified_view ?? true;
  const endpoints = props.settingsMeta?.endpoints;
  const endpointRows = endpoints?.rows ?? [];
  const endpointNotices = endpoints?.notices ?? [];
  const diagnostics = props.settingsMeta?.diagnostics;
  const diagnosticRows = diagnostics?.rows ?? [];
  const modelId = useId();
  const tempId = useId();
  const names = new Map(models.map((m) => [m.id, m.name]));
  const current = models.find((m) => m.id === settings.model);
  // Provider-aware builds label each model; group them once there is a choice.
  const byProvider = new Map<string, ModelMeta[]>();
  for (const m of models) {
    const key = m.provider || "Other";
    byProvider.set(key, [...(byProvider.get(key) ?? []), m]);
  }
  const groups = [...byProvider.entries()];

  if (mode === "rail") {
    return (
      <nav className="sidebar sidebar--rail" aria-label="Sidebar">
        <button type="button" className="icon-btn" onClick={onToggle} aria-label="Open sidebar" title="Open sidebar">
          <IconSidebar />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={props.onNewChat}
          disabled={props.newChatDisabled}
          aria-label="New chat"
          title="New chat"
        >
          <IconNewChat />
        </button>
        <button type="button" className="icon-btn" onClick={onToggle} aria-label="Settings" title="Settings">
          <IconSliders />
        </button>
        <span className="sidebar__spacer" />
        <button type="button" className="icon-btn" onClick={props.onAbout} aria-label="About this demo" title="About this demo">
          <IconHelp />
        </button>
      </nav>
    );
  }

  return (
    <nav className={`sidebar sidebar--${mode}`} aria-label="Sidebar">
      <div className="sidebar__top">
        <span className="sidebar__brand">
          <Mark size={30} />
          <span className="sidebar__brandname">AMA-KBQA</span>
        </span>
        <button
          type="button"
          className="icon-btn"
          onClick={onToggle}
          aria-label={mode === "drawer" ? "Close sidebar" : "Collapse sidebar"}
          title={mode === "drawer" ? "Close sidebar" : "Collapse sidebar"}
        >
          {mode === "drawer" ? <IconClose /> : <IconSidebar />}
        </button>
      </div>

      <button type="button" className="sidebar__newchat" onClick={props.onNewChat} disabled={props.newChatDisabled}>
        <IconNewChat size={18} />
        New chat
      </button>

      <section className="sidebar__section" aria-labelledby="settings-heading">
        <h2 id="settings-heading" className="sidebar__heading">
          Settings
        </h2>

        {showModel && (
        <div className="field">
          <label htmlFor={modelId} className="field__label">
            Model
          </label>
          <div className="select">
            <select
              id={modelId}
              value={settings.model}
              disabled={running}
              onChange={(e) => props.onModelChange(e.target.value)}
              aria-describedby={`${modelId}-hint${props.modelNotices.length ? ` ${modelId}-notices` : ""}`}
            >
              {groups.length > 1
                ? groups.map(([provider, list]) => (
                    <optgroup key={provider} label={provider}>
                      {list.map((m) => (
                        <option key={m.id} value={m.id}>
                          {modelLabel(m.id, names)}
                        </option>
                      ))}
                    </optgroup>
                  ))
                : models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {modelLabel(m.id, names)}
                    </option>
                  ))}
            </select>
          </div>
          <p id={`${modelId}-hint`} className="field__hint">
            {groups.length > 1 && current?.provider
              ? current.price
                ? `${current.provider}: ${current.price}`
                : current.provider
              : current?.price || "KIT-hosted chat model."}
            {running && " Available again once the answer is ready."}
          </p>
          {props.modelNotices.length > 0 && (
            <ul id={`${modelId}-notices`} className="field__notices">
              {props.modelNotices.map((n, i) => (
                <li key={i}>
                  <IconAlert size={15} />
                  <span>{n}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        )}

        {showTemperature && (
        <div className="field">
          <div className="field__row">
            <label htmlFor={tempId} className="field__label">
              Temperature
            </label>
            <output htmlFor={tempId} className="field__value">
              {settings.temperature.toFixed(2)}
            </output>
          </div>
          <input
            id={tempId}
            className="range"
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={settings.temperature}
            onChange={(e) => props.onTemperatureChange(Number(e.target.value))}
            style={{ ["--fill" as string]: `${(settings.temperature / 2) * 100}%` }}
          />
          <p className="field__hint">Sampling temperature for the agent's LLM calls.</p>
        </div>
        )}

        {showSimplified && (
          <Switch
            label="Simplified view"
            hint="Hide the lifecycle figure, reasoning traces, token counters, and the inspector. Just chat."
            checked={settings.simplified}
            onChange={props.onSimplifiedChange}
          />
        )}
        {liveGraphAvailable && (
          <Switch
            label="Live graph"
            hint="Show the subgraph the agent gathers while it answers."
            checked={settings.liveGraph}
            onChange={props.onLiveGraphChange}
          />
        )}
      </section>

      {endpointRows.length > 0 && (
        <section className="sidebar__section sidebar__section--tight" aria-labelledby="endpoints-heading">
          <h2 id="endpoints-heading" className="sidebar__heading">
            {endpoints?.label || "Endpoints"}
          </h2>
          <ul className="endpoints">
            {endpointRows.map((row) => (
              <Endpoint key={row.id} row={row} />
            ))}
          </ul>
          {endpointNotices.length > 0 && (
            <ul className="field__notices">
              {endpointNotices.map((n, i) => (
                <li key={i}>
                  <IconAlert size={15} />
                  <span>{n}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {diagnosticRows.length > 0 && (
        <section className="sidebar__section sidebar__section--tight" aria-labelledby="build-heading">
          <h2 id="build-heading" className="sidebar__heading">
            {diagnostics?.label || "This build"}
          </h2>
          <dl className="facts facts--standalone">
            {diagnosticRows.map((row) => (
              <Fact key={row.label} term={row.label}>
                <span title={row.detail || undefined}>{row.value}</span>
              </Fact>
            ))}
          </dl>
        </section>
      )}

      <span className="sidebar__spacer" />

      <button type="button" className="sidebar__link" onClick={props.onAbout}>
        <IconHelp size={18} />
        About this demo
      </button>
    </nav>
  );
}

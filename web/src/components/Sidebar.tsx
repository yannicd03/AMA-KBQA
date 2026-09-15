import { useId } from "react";
import type { ModelMeta } from "../api";
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

export function Sidebar(props: SidebarProps) {
  const { mode, onToggle, models, settings, liveGraphAvailable, running } = props;
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

        <Switch
          label="Simplified view"
          hint="Hide the lifecycle figure, reasoning traces, token counters, and the inspector. Just chat."
          checked={settings.simplified}
          onChange={props.onSimplifiedChange}
        />
        {liveGraphAvailable && (
          <Switch
            label="Live graph"
            hint="Show the subgraph the agent gathers while it answers."
            checked={settings.liveGraph}
            onChange={props.onLiveGraphChange}
          />
        )}
      </section>

      <span className="sidebar__spacer" />

      <button type="button" className="sidebar__link" onClick={props.onAbout}>
        <IconHelp size={18} />
        About this demo
      </button>
    </nav>
  );
}

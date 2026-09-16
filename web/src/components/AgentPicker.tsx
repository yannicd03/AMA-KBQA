import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import type { AgentMeta } from "../api";
import { IconCheck, IconChevronDown } from "./icons";

interface AgentPickerProps {
  agents: AgentMeta[];
  value: string;
  onChange: (name: string) => void;
  disabled?: boolean;
  placement?: "up" | "down";
}

/** The agent chip inside the composer, with a keyboard-operable menu. */
export function AgentPicker({ agents, value, onChange, disabled = false, placement = "down" }: AgentPickerProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const menuId = useId();

  const close = useCallback((refocus: boolean) => {
    setOpen(false);
    if (refocus) buttonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!open) return;
    const selected = Math.max(
      0,
      agents.findIndex((a) => a.name === value),
    );
    itemRefs.current[selected]?.focus();
    const onPointer = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", onPointer);
    return () => document.removeEventListener("pointerdown", onPointer);
  }, [open, agents, value, close]);

  useEffect(() => {
    if (disabled && open) setOpen(false);
  }, [disabled, open]);

  const onMenuKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const items = itemRefs.current.filter(Boolean) as HTMLButtonElement[];
    const idx = items.indexOf(document.activeElement as HTMLButtonElement);
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      close(true);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      items[(idx + 1) % items.length]?.focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      items[(idx - 1 + items.length) % items.length]?.focus();
    } else if (e.key === "Home") {
      e.preventDefault();
      items[0]?.focus();
    } else if (e.key === "End") {
      e.preventDefault();
      items[items.length - 1]?.focus();
    } else if (e.key === "Tab") {
      close(false);
    }
  };

  const onButtonKey = (e: ReactKeyboardEvent<HTMLButtonElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setOpen(true);
    }
  };

  return (
    <div className="picker" ref={rootRef}>
      <button
        ref={buttonRef}
        type="button"
        className="chip chip--agent"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onButtonKey}
        title={disabled ? "The agent can be changed once the current answer is ready" : "Choose an agent"}
      >
        <span className="chip__dot" aria-hidden="true" />
        <span className="chip__label">
          <span className="sr-only">Agent: </span>
          {value}
        </span>
        <IconChevronDown size={16} className="chip__chevron" />
      </button>
      {open && (
        <div
          id={menuId}
          className={`popover popover--${placement}`}
          role="menu"
          aria-label="Choose an agent"
          onKeyDown={onMenuKey}
        >
          <p className="popover__title" aria-hidden="true">
            Choose an agent
          </p>
          {agents.map((a, i) => {
            const selected = a.name === value;
            return (
              <button
                key={a.name}
                ref={(el) => {
                  itemRefs.current[i] = el;
                }}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                className={`agent-option${selected ? " is-selected" : ""}`}
                onClick={() => {
                  close(true);
                  if (!selected) onChange(a.name);
                }}
              >
                <span className="agent-option__text">
                  <span className="agent-option__name">{a.name}</span>
                  <span className="agent-option__tagline">{a.tagline || a.description}</span>
                </span>
                <span className="agent-option__check" aria-hidden="true">
                  {selected && <IconCheck size={18} />}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

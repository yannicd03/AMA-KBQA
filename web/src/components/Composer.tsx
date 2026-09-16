import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import type { AgentMeta } from "../api";
import { AgentPicker } from "./AgentPicker";
import { IconArrowUp, IconNewChat, IconStop } from "./icons";

const MAX_CHARS = 4000;

interface ComposerProps {
  variant: "landing" | "docked";
  agents: AgentMeta[];
  agent: string;
  onAgentChange: (name: string) => void;
  pickerDisabled: boolean;
  /** false for orchestrator entries in a conversation: no follow-up input. */
  showInput: boolean;
  placeholder: string;
  inputDisabled: boolean;
  onSubmit: (text: string) => Promise<boolean>;
  /** shown instead of the input when `showInput` is false */
  caption?: ReactNode;
  onRestart?: () => void;
  restartDisabled?: boolean;
  /** Set only while a run is in flight; takes over the action slot. */
  onStop?: () => void;
  /** true once Stop was pressed and the server has been told. */
  stopping?: boolean;
  autoFocus?: boolean;
}

export function Composer({
  variant,
  agents,
  agent,
  onAgentChange,
  pickerDisabled,
  showInput,
  placeholder,
  inputDisabled,
  onSubmit,
  caption,
  onRestart,
  restartDisabled,
  onStop,
  stopping,
  autoFocus,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to ~8 lines, then scroll.
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text, showInput]);

  useEffect(() => {
    if (autoFocus && showInput && !inputDisabled) areaRef.current?.focus({ preventScroll: true });
  }, [autoFocus, showInput, inputDisabled]);

  const trimmed = text.trim();
  const canSend = showInput && !inputDisabled && !sending && trimmed.length > 0;

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!canSend) return;
    setSending(true);
    const ok = await onSubmit(trimmed);
    setSending(false);
    if (ok) setText("");
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    }
  };

  return (
    <form className={`composer composer--${variant}`} onSubmit={submit} aria-label="Ask the agent">
      {showInput ? (
        <>
          <label className="sr-only" htmlFor={`composer-${variant}`}>
            {placeholder}
          </label>
          <textarea
            id={`composer-${variant}`}
            ref={areaRef}
            className="composer__input"
            rows={variant === "landing" ? 2 : 1}
            maxLength={MAX_CHARS}
            placeholder={placeholder}
            value={text}
            disabled={inputDisabled}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
          />
        </>
      ) : (
        caption && <p className="composer__caption">{caption}</p>
      )}
      <div className="composer__bar">
        <AgentPicker
          agents={agents}
          value={agent}
          onChange={onAgentChange}
          disabled={pickerDisabled}
          placement={variant === "docked" ? "up" : "down"}
        />
        <div className="composer__actions">
          {showInput && text.length > MAX_CHARS - 400 && (
            <span className="composer__count" aria-live="polite">
              {text.length}/{MAX_CHARS}
            </span>
          )}
          {/* While a run is in flight Stop takes the slot, the way the send
              button becomes a stop button in chat products. Send is disabled
              then anyway, and New chat cannot be used mid-run either. */}
          {onStop ? (
            <button
              type="button"
              className="send send--stop"
              onClick={onStop}
              disabled={stopping}
              aria-label={stopping ? "Stopping the run" : "Stop the run"}
              // Honest: the token is cooperative, so the agent finishes the
              // step it is on before the run actually ends.
              title={stopping ? "Stopping — the agent is finishing its current step" : "Stop the run"}
            >
              <IconStop size={16} />
            </button>
          ) : showInput ? (
            <button
              type="submit"
              className="send"
              disabled={!canSend}
              aria-label={sending ? "Sending" : "Send question"}
              title="Send (Enter)"
            >
              <IconArrowUp size={18} />
            </button>
          ) : (
            onRestart && (
              <button type="button" className="btn btn--soft" onClick={onRestart} disabled={restartDisabled}>
                <IconNewChat size={17} />
                New chat
              </button>
            )
          )}
        </div>
      </div>
    </form>
  );
}

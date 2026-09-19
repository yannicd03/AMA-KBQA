import type { Tokens } from "../api";

const int = new Intl.NumberFormat("en-US");

/** Wall-clock run length for the answer footer.
 *
 * Seconds below a minute, as chat.py prints them. Longer runs get minutes and
 * hours instead: a question answered across a laptop suspend really does take
 * hours, and "75128.35s" tells the reader nothing.
 */
export function formatRunDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${s}s`;
}

/** Footer parts, same wording as chat.py `_render_message_footer`. */
export function footerParts(durationS: number | null | undefined, tokens: Tokens | null | undefined, cost: string | null | undefined): string[] {
  const parts: string[] = [];
  if (typeof durationS === "number") parts.push(formatRunDuration(durationS));
  if (tokens && tokens.total > 0) {
    parts.push(
      `${int.format(tokens.prompt)} prompt + ${int.format(tokens.completion)} completion = ${int.format(tokens.total)} tokens`,
    );
    if (cost) parts.push(`~${cost} est.`);
  }
  return parts;
}

/** Span duration, same thresholds as trace_render.format_duration. */
export function formatDuration(ms: number): string {
  if (ms < 1) return `${Math.round(ms * 1000)}µs`;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  const m = Math.floor(ms / 60_000);
  const s = (ms - m * 60_000) / 1000;
  return `${m}m${Math.round(s)}s`;
}

export function formatInt(n: number): string {
  return int.format(n);
}

export function truncate(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
}

export function clockTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Model display name: the endpoint's name, else the id without `kit.`. */
export function modelLabel(id: string, names: Map<string, string>): string {
  return names.get(id) ?? (id.startsWith("kit.") ? id.slice(4) : id);
}

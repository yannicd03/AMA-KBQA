// Per-tab persistence. Everything lives in sessionStorage: a new tab is a new
// demo session (fresh server-side agent and fresh demo limits), a reload of
// the same tab resumes it.

const SESSION_KEY = "amakbqa:session";
const STATE_KEY = "amakbqa:state";
const ABOUT_KEY = "amakbqa:about-seen";

function newId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // crypto.randomUUID needs a secure context; plain-http booth hosts other
  // than localhost do not have one.
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function read(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string): void {
  try {
    window.sessionStorage.setItem(key, value);
  } catch {
    /* storage full or blocked: persistence is a convenience only */
  }
}

let cachedSessionId: string | null = null;

export function getSessionId(): string {
  if (cachedSessionId) return cachedSessionId;
  let id = read(SESSION_KEY);
  if (!id) {
    id = newId();
    write(SESSION_KEY, id);
  }
  cachedSessionId = id;
  return id;
}

export function aboutSeen(): boolean {
  return read(ABOUT_KEY) === "1";
}

export function markAboutSeen(): void {
  write(ABOUT_KEY, "1");
}

export function loadState<T>(): T | null {
  const raw = read(STATE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function saveState(value: unknown): void {
  write(STATE_KEY, JSON.stringify(value));
}

export function uid(): string {
  return newId();
}

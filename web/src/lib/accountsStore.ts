/*
 * Browser-side account tokens (HF / Civitai). The mold server has NO config
 * keys for these — server-side credentials come from env (HF_TOKEN /
 * CIVITAI_TOKEN) or the HF CLI cache. The web app instead forwards a token
 * per catalog request via the `x-mold-hf-token` / `x-mold-civitai-token`
 * headers the server already honours (crates/mold-server/src/catalog_api.rs,
 * ForwardedCatalogCredentials) — exactly how the desktop app does it. So the
 * tokens live in this browser only, never in a URL, and are attached solely to
 * catalog fetches. Same localStorage pattern as the host registry.
 */
const STORAGE_KEY = "mold.web.accounts.v1";

export interface AccountTokens {
  hfToken?: string;
  civitaiToken?: string;
}

function read(): AccountTokens {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as AccountTokens;
    return {
      hfToken: typeof parsed.hfToken === "string" ? parsed.hfToken : undefined,
      civitaiToken:
        typeof parsed.civitaiToken === "string"
          ? parsed.civitaiToken
          : undefined,
    };
  } catch {
    return {};
  }
}

/**
 * Persist the token set. Returns whether the browser actually kept it, so a
 * caller never claims a save that didn't happen: quota-exceeded, Safari
 * private mode and policy-blocked storage all fail here, some by throwing and
 * some by accepting the call and dropping the value. Success is therefore
 * decided by reading the row back, not by the absence of an exception.
 */
function write(next: AccountTokens): boolean {
  // Drop empty entries so a cleared token doesn't linger as "".
  const clean: AccountTokens = {};
  if (next.hfToken) clean.hfToken = next.hfToken;
  if (next.civitaiToken) clean.civitaiToken = next.civitaiToken;
  const payload =
    clean.hfToken || clean.civitaiToken ? JSON.stringify(clean) : null;
  try {
    if (payload === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, payload);
    return localStorage.getItem(STORAGE_KEY) === payload;
  } catch {
    return false;
  }
}

export function getAccountTokens(): AccountTokens {
  return read();
}

/** @returns true only when the token is genuinely stored in this browser. */
export function setHfToken(token: string): boolean {
  const cur = read();
  return write({ ...cur, hfToken: token.trim() || undefined });
}

/** @returns true only when the token is genuinely stored in this browser. */
export function setCivitaiToken(token: string): boolean {
  const cur = read();
  return write({ ...cur, civitaiToken: token.trim() || undefined });
}

/** @returns true only when the token is genuinely gone from this browser. */
export function clearHfToken(): boolean {
  const cur = read();
  return write({ ...cur, hfToken: undefined });
}

/** @returns true only when the token is genuinely gone from this browser. */
export function clearCivitaiToken(): boolean {
  const cur = read();
  return write({ ...cur, civitaiToken: undefined });
}

/** A privacy-safe indicator: never the full token, just enough to recognize
 * it (a short prefix when present + the last four). */
export function maskToken(token: string | undefined): string {
  if (!token) return "";
  const t = token.trim();
  if (t.length <= 4) return "••••";
  const last4 = t.slice(-4);
  const prefixMatch = t.match(/^([a-z]{2,6}_)/i);
  const prefix = prefixMatch ? prefixMatch[1] : "";
  return `${prefix}••••${last4}`;
}

/** Headers to forward the stored tokens on catalog requests. Omits any token
 * that isn't set, so an anonymous browser sends nothing. */
export function catalogAuthHeaders(): Record<string, string> {
  const { hfToken, civitaiToken } = read();
  const headers: Record<string, string> = {};
  if (hfToken) headers["x-mold-hf-token"] = hfToken;
  if (civitaiToken) headers["x-mold-civitai-token"] = civitaiToken;
  return headers;
}

import { apiFetchTo, apiJsonTo, type ApiTarget } from "./client";
import type { ConfigRow } from "./types";

/**
 * Per-host `/api/config/:key` helpers. The Settings store (`settingsConfig`)
 * only ever targets the primary; Machines ▸ host detail edits a REMOTE host's
 * own engine config (trash retention today) and must name that host
 * explicitly — never route through whichever host happens to be primary.
 */

function keyPath(key: string): string {
  return `/api/config/${encodeURIComponent(key)}`;
}

/** `GET /api/config/:key` on `target` — one `ConfigRow` (value + provenance). */
export function fetchHostConfigKey(
  target: ApiTarget,
  key: string,
  signal: AbortSignal | null = null,
): Promise<ConfigRow> {
  return apiJsonTo<ConfigRow>(target, keyPath(key), { signal });
}

/** `PUT /api/config/:key {value}` on `target`. `null` clears optional keys. */
export function setHostConfigKey(
  target: ApiTarget,
  key: string,
  value: ConfigRow["value"],
): Promise<Response> {
  return apiFetchTo(target, keyPath(key), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

/** `DELETE /api/config/:key` on `target` — back to the engine default. */
export function resetHostConfigKey(target: ApiTarget, key: string): Promise<Response> {
  return apiFetchTo(target, keyPath(key), { method: "DELETE" });
}

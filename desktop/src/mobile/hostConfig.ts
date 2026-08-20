/**
 * iPhone's first `/api/config` client — one key at a time, against the exact
 * Keychain-authenticated target of a host. Today it carries only the Library
 * trash retention (`gallery.trash_retention_days`), which is a SERVER setting:
 * mobile settings (`mold.mobile.settings.v1`) stay three local fields.
 */

import { apiJsonTo, type ApiTarget } from "../lib/api/client";

/** Mirrors `mold_core::ConfigEntry` (`GET`/`PUT /api/config/:key`). */
export interface HostConfigEntry {
  key: string;
  value: string | number | boolean | null;
  /** `"db"` / `"file"` / `"env"` / `"default"`. */
  source: string;
  env_var?: string | null;
  restart_required?: boolean;
}

export const TRASH_RETENTION_CONFIG_KEY = "gallery.trash_retention_days";

export function fetchHostConfigKey(
  target: ApiTarget,
  key: string,
  signal?: AbortSignal,
): Promise<HostConfigEntry> {
  return apiJsonTo<HostConfigEntry>(target, `/api/config/${encodeURIComponent(key)}`, {
    signal: signal ?? null,
  });
}

export function setHostConfigKey(
  target: ApiTarget,
  key: string,
  value: string | number | boolean | null,
): Promise<HostConfigEntry> {
  return apiJsonTo<HostConfigEntry>(target, `/api/config/${encodeURIComponent(key)}`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

/** Retention days from a config value (`"30"`, `30`, `null` → 0 = forever). */
export function retentionDaysFromConfigValue(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return 0;
  const parsed = typeof value === "number" ? value : Number(String(value).trim());
  if (!Number.isFinite(parsed) || parsed < 0) return null;
  return Math.floor(parsed);
}

/** A key an environment variable pins cannot be edited from any client. */
export function hostConfigLocked(entry: HostConfigEntry | null): boolean {
  return entry?.source === "env";
}

/** Whether a config control may be edited at all. Unknown authority (no
 * entry — the probe failed or has not answered yet) is read-only: enabling
 * the control on a failed probe would let an env-pinned key be edited. */
export function hostConfigEditable(entry: HostConfigEntry | null): boolean {
  return entry !== null && !hostConfigLocked(entry);
}

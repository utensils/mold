import { apiFetch, apiJson } from "./client";
import type { ConfigProfiles, ConfigRow } from "./types";

/**
 * Config surface. The endpoints are new and their exact shapes may still
 * drift, so every reader normalizes defensively: `GET /api/config` is accepted
 * as either a bare array or `{ rows: [...] }`. A 404 from any of these means
 * the engine predates the config API — callers treat that as "unavailable".
 */
export async function fetchConfig(): Promise<ConfigRow[]> {
  const body = await apiJson<ConfigRow[] | { rows?: ConfigRow[] }>("/api/config");
  if (Array.isArray(body)) return body;
  return body.rows ?? [];
}

export function setConfig(key: string, value: ConfigRow["value"]): Promise<Response> {
  return apiFetch(`/api/config/${encodeURIComponent(key)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
}

export function resetConfig(key: string): Promise<Response> {
  return apiFetch(`/api/config/${encodeURIComponent(key)}`, { method: "DELETE" });
}

export async function fetchProfiles(): Promise<ConfigProfiles> {
  const body = await apiJson<Partial<ConfigProfiles>>("/api/config/profiles");
  return { profiles: body.profiles ?? [], active: body.active ?? "default" };
}

export function setProfile(name: string): Promise<Response> {
  return apiFetch("/api/config/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

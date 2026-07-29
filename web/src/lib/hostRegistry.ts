/*
 * Machines host registry (spec §08 G1) — the browser's list of mold servers
 * it can reach. The primary host is ALWAYS the serving origin
 * (`window.location.origin`): it is id "origin", named "this server", never
 * stored and never removable. Every other host is a user-added remote entry
 * persisted to localStorage under `mold.web.hosts.v1`.
 *
 * API keys stay out of URLs: they live on the entry and are attached as the
 * server's `x-api-key` header by hostClient (see crates/mold-server/src/auth.rs).
 * Entries dedupe by the server's `/api/status.instance_id` so one box reached
 * by hostname / IP / MagicDNS collapses to a single row (keeping the earliest
 * id, per the desktop's instance-UUID rule).
 */

export const ORIGIN_HOST_ID = "origin";
export const HOSTS_STORAGE_KEY = "mold.web.hosts.v1";
export const HOSTS_CHANGED_EVENT = "mold:hosts-changed";

export interface HostEntry {
  /** Slug of the host URL, or "origin" for the serving host. */
  id: string;
  name: string;
  /** Origin URL — scheme + host + optional port, no trailing slash. */
  url: string;
  /** Per-host API key. Never placed in a URL; sent as `x-api-key`. */
  apiKey?: string;
  /** Last-seen `/api/status.instance_id`, used to dedupe re-adds. */
  instanceId?: string;
}

/** The serving origin, always the immutable primary host. */
export function originUrl(): string {
  return typeof window !== "undefined" ? window.location.origin : "";
}

export function originHost(): HostEntry {
  return { id: ORIGIN_HOST_ID, name: "this server", url: originUrl() };
}

/**
 * Normalize a user-typed address (host, host:port, http(s)://…) to an origin
 * URL, or null when it is empty / unparseable. Schemeless input defaults to
 * http:// — the common LAN / IP / MagicDNS case — and, with no port given,
 * to mold's :7680 (the same rule as desktop's normalizeHostUrl and iOS).
 * An explicit scheme is taken as typed: its standard port applies.
 */
export function normalizeHostAddress(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const hasScheme = /^https?:\/\//i.test(trimmed);
  const candidate = hasScheme ? trimmed : `http://${trimmed}`;
  try {
    const url = new URL(candidate);
    if (!url.hostname) return null;
    if (!hasScheme && !url.port) url.port = "7680";
    // `origin` drops any path/query/hash and keeps a non-default port.
    return url.origin;
  } catch {
    return null;
  }
}

/** Derive a stable slug id from an origin URL. */
export function hostIdFromUrl(url: string): string {
  let raw = url;
  try {
    const u = new URL(url);
    raw = `${u.hostname}${u.port ? `-${u.port}` : ""}`;
  } catch {
    // Fall through with the raw string.
  }
  const slug = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!slug) return "host";
  // Never collide with the reserved primary id.
  return slug === ORIGIN_HOST_ID ? `${slug}-1` : slug;
}

function isHostEntry(value: unknown): value is HostEntry {
  if (!value || typeof value !== "object") return false;
  const c = value as Partial<HostEntry>;
  return (
    typeof c.id === "string" &&
    typeof c.name === "string" &&
    typeof c.url === "string" &&
    (c.apiKey === undefined || typeof c.apiKey === "string") &&
    (c.instanceId === undefined || typeof c.instanceId === "string") &&
    c.id !== ORIGIN_HOST_ID
  );
}

/** User-added remote hosts (excludes the primary origin). */
export function listStoredHosts(): HostEntry[] {
  const raw = localStorage.getItem(HOSTS_STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHostEntry);
  } catch {
    return [];
  }
}

function writeStoredHosts(hosts: HostEntry[]): void {
  localStorage.setItem(HOSTS_STORAGE_KEY, JSON.stringify(hosts));
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(HOSTS_CHANGED_EVENT));
  }
}

/** Every host the browser knows: the primary origin first, then remotes. */
export function listHosts(): HostEntry[] {
  return [originHost(), ...listStoredHosts()];
}

export function getHost(id: string): HostEntry | null {
  return listHosts().find((h) => h.id === id) ?? null;
}

/**
 * The stored remote whose recorded instance id matches `instanceId`, or null.
 * A blank instance id never matches — an unknown server must not merge.
 */
export function dedupeByInstanceId(instanceId: string): HostEntry | null {
  if (!instanceId) return null;
  return listStoredHosts().find((h) => h.instanceId === instanceId) ?? null;
}

export interface AddHostInput {
  url: string;
  name: string;
  apiKey?: string;
  instanceId?: string;
}

/**
 * Add or merge a remote host. Adding a host whose instance id (or, failing
 * that, URL slug) already exists updates that entry in place — keeping its
 * earliest id — rather than creating a duplicate row. Adding the serving
 * origin is a no-op that returns the immutable primary.
 */
export function addHost(input: AddHostInput): HostEntry {
  const url = normalizeHostAddress(input.url) ?? input.url;
  const name = input.name.trim() || url;

  if (url === originUrl()) return originHost();

  const stored = listStoredHosts();
  const byInstance = input.instanceId
    ? stored.find((h) => h.instanceId === input.instanceId)
    : undefined;
  const id = hostIdFromUrl(url);
  const existing = byInstance ?? stored.find((h) => h.id === id);

  const entry: HostEntry = {
    id: existing?.id ?? id,
    name,
    url,
  };
  if (input.apiKey) entry.apiKey = input.apiKey;
  else if (existing?.apiKey) entry.apiKey = existing.apiKey;
  if (input.instanceId) entry.instanceId = input.instanceId;
  else if (existing?.instanceId) entry.instanceId = existing.instanceId;

  const next = existing
    ? stored.map((h) => (h.id === existing.id ? entry : h))
    : [...stored, entry];
  writeStoredHosts(next);
  return entry;
}

/** Patch a stored host. The immutable primary cannot be updated (returns null). */
export function updateHost(
  id: string,
  patch: Partial<Omit<HostEntry, "id">>,
): HostEntry | null {
  if (id === ORIGIN_HOST_ID) return null;
  const stored = listStoredHosts();
  const current = stored.find((h) => h.id === id);
  if (!current) return null;
  const updated: HostEntry = { ...current, ...patch, id };
  writeStoredHosts(stored.map((h) => (h.id === id ? updated : h)));
  return updated;
}

/** Remove a stored host. The immutable primary is never removed. */
export function removeHost(id: string): void {
  if (id === ORIGIN_HOST_ID) return;
  writeStoredHosts(listStoredHosts().filter((h) => h.id !== id));
}

export const GENERATE_TARGET_STORAGE_KEY = "mold.web.generateTarget.v1";

/** Host id chosen as the generation target (defaults to the primary origin). */
export function getGenerateTargetId(): string {
  return localStorage.getItem(GENERATE_TARGET_STORAGE_KEY) ?? ORIGIN_HOST_ID;
}

export function setGenerateTargetId(id: string): void {
  localStorage.setItem(GENERATE_TARGET_STORAGE_KEY, id);
}

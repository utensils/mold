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
export const GENERATE_TARGET_CHANGED_EVENT = "mold:generate-target-changed";
const TRACKED_SEQUENCES_KEY = "mold.create.tracked-sequences.v1";
const LEGACY_SEQUENCE_HOST_KEY = "mold.create.chain-job-host";
const GENERATE_JOBS_KEY = "mold.generate.jobs";
const GENERATE_RECOVERY_PREFIX = `${GENERATE_JOBS_KEY}.recovery.`;

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
  /** Successful connection that selected `url`; failed attempts never update it. */
  lastConnectedAtMs?: number;
  /** False only after an explicit disconnect. Missing means connected for
   *  compatibility with registries written by older Mold versions. */
  connected?: boolean;
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
    (c.lastConnectedAtMs === undefined ||
      typeof c.lastConnectedAtMs === "number") &&
    (c.connected === undefined || typeof c.connected === "boolean") &&
    c.id !== ORIGIN_HOST_ID
  );
}

function normalizedInstanceId(value: string | null | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

export function mergeStoredHostsByInstanceId(hosts: readonly HostEntry[]): {
  hosts: HostEntry[];
  dropped: Array<{ loser: string; survivor: string }>;
} {
  const winnerByUuid = new Map<string, HostEntry>();
  for (const host of hosts) {
    const uuid = normalizedInstanceId(host.instanceId);
    if (!uuid) continue;
    const current = winnerByUuid.get(uuid);
    const hostConnected = host.connected !== false;
    const currentConnected = current?.connected !== false;
    if (
      !current ||
      (hostConnected && !currentConnected) ||
      (hostConnected === currentConnected &&
        (host.lastConnectedAtMs ?? 0) > (current.lastConnectedAtMs ?? 0))
    ) {
      winnerByUuid.set(uuid, host);
    }
  }
  const merged: HostEntry[] = [];
  const dropped: Array<{ loser: string; survivor: string }> = [];
  for (const host of hosts) {
    const uuid = normalizedInstanceId(host.instanceId);
    const winner = uuid ? winnerByUuid.get(uuid) : undefined;
    if (!winner || winner.id === host.id) {
      const fallbackKey = uuid
        ? hosts.find(
            (candidate) =>
              normalizedInstanceId(candidate.instanceId) === uuid &&
              Boolean(candidate.apiKey),
          )?.apiKey
        : undefined;
      merged.push(
        fallbackKey && !host.apiKey ? { ...host, apiKey: fallbackKey } : host,
      );
    } else dropped.push({ loser: host.id, survivor: winner.id });
  }
  return { hosts: merged, dropped };
}

function remapPersistedHostIds(
  dropped: ReadonlyArray<{ loser: string; survivor: string }>,
): void {
  if (dropped.length === 0) return;
  const remap = new Map(
    dropped.map(({ loser, survivor }) => [loser, survivor]),
  );
  const remapJson = (key: string, arrayRoot: boolean): void => {
    const raw = localStorage.getItem(key);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw) as unknown;
      const rows = arrayRoot
        ? parsed
        : parsed && typeof parsed === "object"
          ? (parsed as { jobs?: unknown }).jobs
          : null;
      if (!Array.isArray(rows)) return;
      let changed = false;
      for (const row of rows) {
        if (!row || typeof row !== "object") continue;
        const record = row as { hostId?: unknown };
        if (typeof record.hostId !== "string") continue;
        const survivor = remap.get(record.hostId);
        if (!survivor) continue;
        record.hostId = survivor;
        changed = true;
      }
      if (changed) localStorage.setItem(key, JSON.stringify(parsed));
    } catch {
      // Recovery state is best effort; malformed records remain untouched.
    }
  };
  remapJson(TRACKED_SEQUENCES_KEY, true);
  remapJson(GENERATE_JOBS_KEY, false);
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (key?.startsWith(GENERATE_RECOVERY_PREFIX)) remapJson(key, false);
  }
  const legacyHost = localStorage.getItem(LEGACY_SEQUENCE_HOST_KEY);
  const legacySurvivor = legacyHost ? remap.get(legacyHost) : null;
  if (legacySurvivor)
    localStorage.setItem(LEGACY_SEQUENCE_HOST_KEY, legacySurvivor);
}

function applyAliasRemap(
  dropped: ReadonlyArray<{ loser: string; survivor: string }>,
): void {
  if (dropped.length === 0) return;
  const target = getGenerateTargetId();
  const targetRemap = dropped.find(({ loser }) => loser === target);
  if (targetRemap)
    localStorage.setItem(GENERATE_TARGET_STORAGE_KEY, targetRemap.survivor);
  remapPersistedHostIds(dropped);
}

/** User-added remote hosts (excludes the primary origin). */
export function listStoredHosts(): HostEntry[] {
  const raw = localStorage.getItem(HOSTS_STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    const valid = parsed.filter(isHostEntry);
    const merged = mergeStoredHostsByInstanceId(valid);
    if (merged.dropped.length > 0) {
      localStorage.setItem(HOSTS_STORAGE_KEY, JSON.stringify(merged.hosts));
      applyAliasRemap(merged.dropped);
    }
    return merged.hosts;
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

/** Every host currently in the active routing/polling mix. */
export function listHosts(): HostEntry[] {
  return [
    originHost(),
    ...listStoredHosts().filter((host) => host.connected !== false),
  ];
}

/** Every remembered host, including explicitly disconnected remotes. */
export function listKnownHosts(): HostEntry[] {
  return [originHost(), ...listStoredHosts()];
}

export function getHost(id: string): HostEntry | null {
  return listHosts().find((h) => h.id === id) ?? null;
}

/** A remembered host regardless of whether it is currently connected. */
export function getKnownHost(id: string): HostEntry | null {
  return listKnownHosts().find((h) => h.id === id) ?? null;
}

/**
 * The stored remote whose recorded instance id matches `instanceId`, or null.
 * A blank instance id never matches — an unknown server must not merge.
 */
export function dedupeByInstanceId(instanceId: string): HostEntry | null {
  const normalized = normalizedInstanceId(instanceId);
  if (!normalized) return null;
  return (
    listStoredHosts().find(
      (host) => normalizedInstanceId(host.instanceId) === normalized,
    ) ?? null
  );
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
  const instanceId = normalizedInstanceId(input.instanceId);
  const byInstance = instanceId
    ? stored.find(
        (host) => normalizedInstanceId(host.instanceId) === instanceId,
      )
    : undefined;
  const slug = hostIdFromUrl(url);
  const bySlug = stored.find((host) => host.id === slug);
  const slugInstanceId = normalizedInstanceId(bySlug?.instanceId);
  const slugConflict = Boolean(
    bySlug && instanceId && slugInstanceId && instanceId !== slugInstanceId,
  );
  const existing = byInstance ?? (slugConflict ? undefined : bySlug);
  const id =
    existing?.id ??
    (slugConflict && instanceId
      ? `${slug}-${instanceId
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")
          .slice(0, 12)}`
      : slug);

  const entry: HostEntry = {
    id: existing?.id ?? id,
    name,
    url,
    connected: true,
    lastConnectedAtMs: Date.now(),
  };
  if (input.apiKey) entry.apiKey = input.apiKey;
  else if (existing?.apiKey) entry.apiKey = existing.apiKey;
  if (instanceId) entry.instanceId = instanceId;
  else if (existing?.instanceId) entry.instanceId = existing.instanceId;

  const retired = slugConflict
    ? stored.map((host) =>
        host.id === bySlug?.id ? { ...host, connected: false } : host,
      )
    : stored;
  const next = existing
    ? retired.map((h) => (h.id === existing.id ? entry : h))
    : [...retired, entry];
  writeStoredHosts(next);
  return entry;
}

/** Remove a remembered alias when it proves to be the browser's origin. */
export function reconcileOriginInstanceId(instanceId: string): void {
  const normalized = normalizedInstanceId(instanceId);
  if (!normalized) return;
  const stored = listStoredHosts();
  const aliases = stored.filter(
    (host) => normalizedInstanceId(host.instanceId) === normalized,
  );
  if (aliases.length === 0) return;
  const aliasIds = new Set(aliases.map((host) => host.id));
  writeStoredHosts(stored.filter((host) => !aliasIds.has(host.id)));
  applyAliasRemap(
    aliases.map((host) => ({ loser: host.id, survivor: ORIGIN_HOST_ID })),
  );
}

/** Persist a UUID learned from a successful exact-authority status poll.
 * The answering address wins, then every alias-owned recovery record follows
 * the surviving row id. */
export function recordSuccessfulHostInstance(
  id: string,
  instanceId: string | null | undefined,
): HostEntry | null {
  const normalized = normalizedInstanceId(instanceId);
  if (!normalized) return getKnownHost(id);
  if (id === ORIGIN_HOST_ID) {
    reconcileOriginInstanceId(normalized);
    return originHost();
  }
  const stored = listStoredHosts();
  const current = stored.find((host) => host.id === id);
  if (!current) return null;
  if (normalizedInstanceId(current.instanceId) === normalized) return current;
  const successfulAt = Math.max(
    Date.now(),
    ...stored.map((host) => (host.lastConnectedAtMs ?? 0) + 1),
  );
  const stamped = stored.map((host) =>
    host.id === id
      ? {
          ...host,
          instanceId: normalized,
          connected: true,
          lastConnectedAtMs: successfulAt,
        }
      : host,
  );
  const merged = mergeStoredHostsByInstanceId(stamped);
  const survivorId =
    merged.dropped.find(({ loser }) => loser === id)?.survivor ?? id;
  writeStoredHosts(merged.hosts);
  applyAliasRemap(merged.dropped);
  return merged.hosts.find((host) => host.id === survivorId) ?? null;
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

/** Keep a host remembered while including/excluding it from every live mix. */
export function setHostConnected(
  id: string,
  connected: boolean,
): HostEntry | null {
  return updateHost(id, { connected });
}

export const GENERATE_TARGET_STORAGE_KEY = "mold.web.generateTarget.v1";

/** Host id chosen as the generation target (defaults to the primary origin). */
export function getGenerateTargetId(): string {
  // Default to model-aware Auto routing (desktop parity: null = auto). A
  // fresh install pinned to the origin hid every connected machine's models
  // from the Create picker until the user found the target control.
  return localStorage.getItem(GENERATE_TARGET_STORAGE_KEY) ?? "auto";
}

export function setGenerateTargetId(id: string): void {
  localStorage.setItem(GENERATE_TARGET_STORAGE_KEY, id);
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent(GENERATE_TARGET_CHANGED_EVENT));
  }
}

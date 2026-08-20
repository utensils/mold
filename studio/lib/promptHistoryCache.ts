/**
 * Durable, host-aware prompt-history cache shared by desktop and web.
 *
 * Only prompt metadata is persisted. Host URLs and API keys deliberately stay
 * outside this cache. A successful read replaces that host's cached slice
 * (including with an empty slice after Clear); an unreachable host keeps its
 * last good slice until the host is removed from the client's registry.
 */

export const PROMPT_HISTORY_CACHE_KEY = "mold.prompt-history-cache.v1";
export const PROMPT_HISTORY_PER_HOST_LIMIT = 100;
/** Keep comfortably below common 5 MiB localStorage quotas (UTF-16 storage). */
export const PROMPT_HISTORY_CACHE_CODE_UNIT_BUDGET = 2_000_000;
export const PROMPT_HISTORY_MAX_UTF8_BYTES = 77_000;

export interface PromptHistoryCacheHost {
  hostId: string;
  hostLabel: string;
}

export interface CachedPromptHistoryEntry extends PromptHistoryCacheHost {
  prompt: string;
  model: string;
  used_at: number;
}

interface PromptHistoryCacheWire {
  version: 1;
  entries: CachedPromptHistoryEntry[];
}

export interface PromptHistoryStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** Access itself can throw SecurityError when browser persistence is denied. */
export function availablePromptHistoryStorage(
  scope: { readonly localStorage?: PromptHistoryStorage } = globalThis,
): PromptHistoryStorage | null {
  try {
    return scope.localStorage ?? null;
  } catch {
    return null;
  }
}

function validString(value: unknown, max: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= max;
}

function validPrompt(value: unknown): value is string {
  return (
    validString(value, PROMPT_HISTORY_MAX_UTF8_BYTES) &&
    new TextEncoder().encode(value).byteLength <= PROMPT_HISTORY_MAX_UTF8_BYTES
  );
}

function parseEntry(value: unknown): CachedPromptHistoryEntry | null {
  if (typeof value !== "object" || value === null) return null;
  const row = value as Record<string, unknown>;
  if (
    !validString(row.hostId, 512) ||
    !validString(row.hostLabel, 512) ||
    !validPrompt(row.prompt) ||
    typeof row.model !== "string" ||
    row.model.length > 2_048 ||
    typeof row.used_at !== "number" ||
    !Number.isSafeInteger(row.used_at) ||
    row.used_at < 0
  ) {
    return null;
  }
  return {
    hostId: row.hostId,
    hostLabel: row.hostLabel,
    prompt: row.prompt,
    model: row.model,
    used_at: row.used_at,
  };
}

export function readPromptHistoryCache(
  storage: PromptHistoryStorage | null | undefined,
): CachedPromptHistoryEntry[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(PROMPT_HISTORY_CACHE_KEY);
    if (!raw) return [];
    const decoded = JSON.parse(raw) as Partial<PromptHistoryCacheWire>;
    if (decoded.version !== 1 || !Array.isArray(decoded.entries)) return [];
    return decoded.entries.flatMap((entry) => {
      const parsed = parseEntry(entry);
      return parsed ? [parsed] : [];
    });
  } catch {
    return [];
  }
}

function chronological(
  entries: CachedPromptHistoryEntry[],
): CachedPromptHistoryEntry[] {
  return entries.sort(
    (a, b) =>
      b.used_at - a.used_at ||
      a.hostId.localeCompare(b.hostId) ||
      a.prompt.localeCompare(b.prompt),
  );
}

function fitCacheBudget(
  entries: CachedPromptHistoryEntry[],
): CachedPromptHistoryEntry[] {
  const kept: CachedPromptHistoryEntry[] = [];
  let codeUnits = JSON.stringify({ version: 1, entries: [] }).length;
  // Measure the actual wire shape. Newest-first eviction is deterministic and
  // preserves the prompts a user is most likely to recall while hosts are off.
  for (const entry of entries) {
    const entryCodeUnits =
      JSON.stringify(entry).length + (kept.length > 0 ? 1 : 0);
    if (codeUnits + entryCodeUnits > PROMPT_HISTORY_CACHE_CODE_UNIT_BUDGET)
      break;
    kept.push(entry);
    codeUnits += entryCodeUnits;
  }
  return kept;
}

/** Any configured-host or reachability transition must refresh/re-scope. */
export function promptHistoryHostSignature(
  hosts: readonly {
    id: string;
    label: string;
    status: string;
    baseUrl?: string | null;
    url?: string | null;
  }[],
): string {
  return hosts
    .map(
      (host) =>
        `${host.id}\u0000${host.label}\u0000${host.status}\u0000${host.baseUrl ?? host.url ?? ""}`,
    )
    .sort()
    .join("\u0001");
}

/**
 * Reconcile last-good cached slices with a multi-host fetch and return one
 * newest-first timeline for the currently configured hosts.
 */
export function reconcilePromptHistoryCache(
  storage: PromptHistoryStorage | null | undefined,
  hosts: readonly PromptHistoryCacheHost[],
  liveEntries: readonly CachedPromptHistoryEntry[],
  refreshedHostIds: readonly string[],
): CachedPromptHistoryEntry[] {
  // Registry hydration can briefly expose no hosts at startup. That is not an
  // authoritative "forget everything" event, so surface the last-good cache
  // without erasing it; the next registry update will scope it normally.
  if (hosts.length === 0) return chronological(readPromptHistoryCache(storage));
  const hostsById = new Map(hosts.map((host) => [host.hostId, host]));
  const refreshed = new Set(refreshedHostIds.filter((id) => hostsById.has(id)));
  const merged = readPromptHistoryCache(storage).filter(
    (entry) => hostsById.has(entry.hostId) && !refreshed.has(entry.hostId),
  );

  for (const entry of liveEntries) {
    const host = hostsById.get(entry.hostId);
    const parsed = parseEntry(entry);
    if (!host || !refreshed.has(entry.hostId) || !parsed) continue;
    merged.push({ ...parsed, hostLabel: host.hostLabel });
  }

  const bounded = chronological(
    hosts.flatMap((host) =>
      merged
        .filter((entry) => entry.hostId === host.hostId)
        .sort((a, b) => b.used_at - a.used_at)
        .slice(0, PROMPT_HISTORY_PER_HOST_LIMIT),
    ),
  ).map((entry) => ({
    ...entry,
    hostLabel: hostsById.get(entry.hostId)?.hostLabel ?? entry.hostLabel,
  }));

  if (storage) {
    try {
      const cached = fitCacheBudget(bounded);
      storage.setItem(
        PROMPT_HISTORY_CACHE_KEY,
        JSON.stringify({
          version: 1,
          entries: cached,
        } satisfies PromptHistoryCacheWire),
      );
    } catch {
      // Private browsing and full quotas must not disable live history recall.
    }
  }
  return bounded;
}

/** Persist a just-accepted prompt before the host can go offline or reload. */
export function recordPromptHistoryCache(
  storage: PromptHistoryStorage | null | undefined,
  hosts: readonly PromptHistoryCacheHost[],
  hostId: string,
  entry: PromptHistoryLiveEntry,
): CachedPromptHistoryEntry[] {
  const host = hosts.find((candidate) => candidate.hostId === hostId);
  if (!host) return chronological(readPromptHistoryCache(storage));
  const hostEntries = readPromptHistoryCache(storage).filter(
    (cached) => cached.hostId === hostId,
  );
  return reconcilePromptHistoryCache(
    storage,
    hosts,
    [{ ...entry, hostId, hostLabel: host.hostLabel }, ...hostEntries],
    [hostId],
  );
}

export interface PromptHistoryFleetHost<T> extends PromptHistoryCacheHost {
  fetchable: boolean;
  source: T;
}

export interface PromptHistoryLiveEntry {
  prompt: string;
  model: string;
  used_at: number;
}

/**
 * One shared fan-out/stale-response authority for desktop and web Create.
 * Secrets may exist in `source`, but only normalized entry metadata reaches
 * `reconcilePromptHistoryCache` and persistence.
 */
export class PromptHistoryCoordinator {
  private epoch = 0;

  async load<T>(
    storage: PromptHistoryStorage | null | undefined,
    hosts: readonly PromptHistoryFleetHost<T>[],
    fetchHost: (source: T) => Promise<readonly PromptHistoryLiveEntry[]>,
  ): Promise<CachedPromptHistoryEntry[] | null> {
    const epoch = ++this.epoch;
    const targets = hosts.filter((host) => host.fetchable);
    const settled = await Promise.allSettled(
      targets.map(async (host) => ({
        host,
        entries: await fetchHost(host.source),
      })),
    );
    if (epoch !== this.epoch) return null;

    const refreshedHostIds: string[] = [];
    const liveEntries: CachedPromptHistoryEntry[] = [];
    for (const result of settled) {
      if (result.status !== "fulfilled") continue;
      refreshedHostIds.push(result.value.host.hostId);
      for (const entry of result.value.entries) {
        liveEntries.push({
          ...entry,
          hostId: result.value.host.hostId,
          hostLabel: result.value.host.hostLabel,
        });
      }
    }
    return reconcilePromptHistoryCache(
      storage,
      hosts,
      liveEntries,
      refreshedHostIds,
    );
  }

  invalidate(): void {
    this.epoch += 1;
  }
}

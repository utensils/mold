import { ApiError, apiFetchTo, apiJsonTo, currentTarget, type ApiTarget } from "./client";

export interface HistoryEntry {
  prompt: string;
  model: string;
  /** Unix epoch milliseconds (mold_core::HistoryEntry::used_at). */
  used_at: number;
}

/** A history entry tagged with the host it came from. */
export interface HostHistoryEntry extends HistoryEntry {
  hostId: string;
  hostLabel: string;
}

/** A ready host the prompt log can be read from / cleared on. */
export interface HistoryHostTarget {
  hostId: string;
  label: string;
  target: ApiTarget;
}

export interface MultiHostHistory {
  /** Every host's entries merged, newest first, host-tagged. */
  entries: HostHistoryEntry[];
  /** Hosts whose GET /api/history succeeded — i.e. that support history. */
  supportedHostIds: string[];
  /** Hosts that failed with something OTHER than 404/503 — a network blip,
   *  not a server that lacks the history API. Lets the empty state say
   *  "couldn't reach" instead of wrongly blaming old servers. */
  unreachableHostIds: string[];
}

/** Fetch one host's prompt log. */
export async function fetchHistoryFrom(
  target: ApiTarget,
  query = "",
  limit = 100,
): Promise<HistoryEntry[]> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  params.set("limit", String(limit));
  const listing = await apiJsonTo<{ entries: HistoryEntry[] }>(target, `/api/history?${params}`);
  return listing.entries;
}

/** Fetch the primary connection's prompt log (single-host callers). */
export async function fetchHistory(query = "", limit = 100): Promise<HistoryEntry[]> {
  return fetchHistoryFrom(currentTarget(), query, limit);
}

/**
 * Fan the prompt log out over every connected host and merge newest-first.
 * A host that fails — 404/503 `HISTORY_UNAVAILABLE` on older servers or with
 * the DB off, or plain unreachable — is silently skipped; callers show the
 * "history isn't available" state only when NO host supports it.
 */
export async function fetchHistoryAll(
  hostTargets: HistoryHostTarget[],
  query = "",
): Promise<MultiHostHistory> {
  const results = await Promise.allSettled(
    hostTargets.map(async (host) => ({
      host,
      entries: await fetchHistoryFrom(host.target, query),
    })),
  );
  const entries: HostHistoryEntry[] = [];
  const supportedHostIds: string[] = [];
  const unreachableHostIds: string[] = [];
  results.forEach((result, i) => {
    if (result.status !== "fulfilled") {
      // 404/503 = the server genuinely lacks the history API (or runs with
      // the DB off); anything else is a transient reachability failure.
      const status = result.reason instanceof ApiError ? result.reason.status : 0;
      if (status !== 404 && status !== 503) unreachableHostIds.push(hostTargets[i]!.hostId);
      return;
    }
    const { host } = result.value;
    supportedHostIds.push(host.hostId);
    for (const entry of result.value.entries) {
      entries.push({ ...entry, hostId: host.hostId, hostLabel: host.label });
    }
  });
  entries.sort((a, b) => b.used_at - a.used_at);
  return { entries, supportedHostIds, unreachableHostIds };
}

/** Clear one host's prompt log. */
export async function clearHistoryOn(target: ApiTarget, keep?: number): Promise<void> {
  const suffix = keep !== undefined ? `?keep=${keep}` : "";
  await apiFetchTo(target, `/api/history${suffix}`, { method: "DELETE" });
}

/** Clear the primary connection's prompt log (single-host callers). */
export async function clearHistory(keep?: number): Promise<void> {
  await clearHistoryOn(currentTarget(), keep);
}

/**
 * Which hosts a Clear should hit under the active chip filter: the one
 * filtered host, or every given (history-capable) host when "all".
 */
export function clearScope<H extends { hostId: string }>(filter: string, hosts: H[]): H[] {
  if (filter === "all") return hosts;
  return hosts.filter((h) => h.hostId === filter);
}

/** Group entries into Today / Yesterday / date buckets, newest first. */
export function groupByDay<T extends HistoryEntry>(
  entries: T[],
  now = new Date(),
): Array<{ label: string; entries: T[] }> {
  const dayKey = (d: Date) => d.toDateString();
  const today = dayKey(now);
  const yesterday = dayKey(new Date(now.getTime() - 86_400_000));
  const groups: Array<{ label: string; entries: T[] }> = [];
  for (const entry of entries) {
    const date = new Date(entry.used_at);
    const key = dayKey(date);
    const label =
      key === today
        ? "Today"
        : key === yesterday
          ? "Yesterday"
          : date.toLocaleDateString(undefined, { month: "long", day: "numeric" });
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.entries.push(entry);
    else groups.push({ label, entries: [entry] });
  }
  return groups;
}

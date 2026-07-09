import { apiFetch, apiJson } from "./client";

export interface HistoryEntry {
  prompt: string;
  model: string;
  /** Unix seconds. */
  used_at: number;
}

export async function fetchHistory(query = "", limit = 100): Promise<HistoryEntry[]> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  params.set("limit", String(limit));
  const listing = await apiJson<{ entries: HistoryEntry[] }>(`/api/history?${params}`);
  return listing.entries;
}

export async function clearHistory(keep?: number): Promise<void> {
  const suffix = keep !== undefined ? `?keep=${keep}` : "";
  await apiFetch(`/api/history${suffix}`, { method: "DELETE" });
}

/** Group entries into Today / Yesterday / date buckets, newest first. */
export function groupByDay(
  entries: HistoryEntry[],
  now = new Date(),
): Array<{ label: string; entries: HistoryEntry[] }> {
  const dayKey = (d: Date) => d.toDateString();
  const today = dayKey(now);
  const yesterday = dayKey(new Date(now.getTime() - 86_400_000));
  const groups: Array<{ label: string; entries: HistoryEntry[] }> = [];
  for (const entry of entries) {
    const date = new Date(entry.used_at * 1000);
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

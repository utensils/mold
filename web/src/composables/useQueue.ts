import { getCurrentInstance, onUnmounted, ref, type Ref } from "vue";
import { fetchQueue } from "../api";
import type { QueueEntry } from "../types";

export interface UseQueue {
  entries: Ref<QueueEntry[]>;
  error: Ref<Error | null>;
  refresh: () => Promise<void>;
  stop: () => void;
}

export function useQueue(options: { pollMs?: number } = {}): UseQueue {
  const pollMs = options.pollMs ?? 5_000;
  const entries = ref<QueueEntry[]>([]);
  const error = ref<Error | null>(null);
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  async function refresh() {
    try {
      const listing = await fetchQueue();
      entries.value = listing.entries;
      error.value = null;
    } catch (e) {
      error.value = e instanceof Error ? e : new Error(String(e));
    }
  }

  async function tick() {
    if (stopped) return;
    await refresh();
    if (!stopped) timer = setTimeout(tick, pollMs);
  }

  function stop() {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = null;
  }

  void tick();
  if (getCurrentInstance()) onUnmounted(stop);

  return { entries, error, refresh, stop };
}

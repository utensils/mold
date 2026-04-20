import {
  computed,
  onBeforeUnmount,
  ref,
  type ComputedRef,
  type Ref,
} from "vue";
import type { GpuSnapshot, ResourceSnapshot } from "../types";

export interface UseResources {
  snapshot: Ref<ResourceSnapshot | null>;
  gpuList: ComputedRef<GpuSnapshot[]>;
  error: Ref<string | null>;
  /** Close the underlying EventSource and stop reconnect attempts. */
  stop: () => void;
}

/**
 * Singleton-style SSE consumer for `/api/resources/stream`.
 *
 * The design mirrors `useStatusPoll` but uses SSE instead of polling, with
 * exponential-backoff reconnect (capped at 30 s) because the aggregator
 * ticks at 1 Hz and dropping frames is fine — we only ever want the latest.
 *
 * Agent C's `PlacementPanel.vue` consumes `gpuList` to populate the device
 * selector. Keep that return shape stable.
 */
export function useResources(): UseResources {
  const snapshot = ref<ResourceSnapshot | null>(null);
  const error = ref<string | null>(null);

  let es: EventSource | null = null;
  let retryDelay = 1000;
  const MAX_RETRY = 30_000;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    try {
      es = new EventSource("/api/resources/stream");
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
      scheduleRetry();
      return;
    }

    es.addEventListener("snapshot", (evt: MessageEvent) => {
      try {
        snapshot.value = JSON.parse(evt.data) as ResourceSnapshot;
        error.value = null;
        retryDelay = 1000; // reset backoff on success
      } catch (e) {
        error.value = `parse failed: ${e instanceof Error ? e.message : String(e)}`;
      }
    });

    es.onerror = () => {
      error.value = "resource telemetry stream lost";
      if (es) {
        es.close();
        es = null;
      }
      scheduleRetry();
    };
  }

  function scheduleRetry() {
    if (stopped) return;
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      connect();
      retryDelay = Math.min(retryDelay * 2, MAX_RETRY);
    }, retryDelay);
  }

  function stop() {
    stopped = true;
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (es) {
      es.close();
      es = null;
    }
  }

  // Kick off immediately (used at module scope via singleton wrapper).
  connect();

  onBeforeUnmount(stop);

  const gpuList = computed<GpuSnapshot[]>(() => snapshot.value?.gpus ?? []);

  return { snapshot, gpuList, error, stop };
}

// ── Singleton wrapper ────────────────────────────────────────────────────────
// App.vue mounts `useResources()` once via `provide()`; pages consume via
// `inject()` so every subscriber shares a single EventSource.
export const RESOURCES_INJECTION_KEY = Symbol("mold.resources");

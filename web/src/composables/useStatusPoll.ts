import { onBeforeUnmount, onMounted, ref, type Ref } from "vue";
import { fetchStatus } from "../api";
import type { ServerStatus } from "../types";

export interface UseStatusPoll {
  status: Ref<ServerStatus | null>;
  error: Ref<string | null>;
}

export function useStatusPoll(intervalMs = 5000): UseStatusPoll {
  const status = ref<ServerStatus | null>(null);
  const error = ref<string | null>(null);
  let timer: ReturnType<typeof setInterval> | null = null;
  let controller: AbortController | null = null;

  async function tick() {
    controller?.abort();
    controller = new AbortController();
    try {
      status.value = await fetchStatus(controller.signal);
      error.value = null;
    } catch (e) {
      if (controller.signal.aborted) return;
      error.value = e instanceof Error ? e.message : String(e);
    }
  }

  function start() {
    if (timer) return;
    tick();
    timer = setInterval(tick, intervalMs);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    controller?.abort();
    controller = null;
  }

  function onVisibilityChange() {
    if (document.hidden) stop();
    else start();
  }

  onMounted(() => {
    start();
    document.addEventListener("visibilitychange", onVisibilityChange);
  });
  onBeforeUnmount(() => {
    stop();
    document.removeEventListener("visibilitychange", onVisibilityChange);
  });

  return { status, error };
}

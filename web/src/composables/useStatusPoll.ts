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
  let timer: ReturnType<typeof setTimeout> | null = null;
  let controller: AbortController | null = null;

  async function tick() {
    if (controller !== null) return;

    const requestController = new AbortController();
    controller = requestController;
    try {
      const nextStatus = await fetchStatus(requestController.signal);
      if (requestController.signal.aborted || controller !== requestController)
        return;
      status.value = nextStatus;
      error.value = null;
    } catch (e) {
      if (requestController.signal.aborted || controller !== requestController)
        return;
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      if (controller !== requestController) return;
      controller = null;
      timer = setTimeout(() => {
        timer = null;
        void tick();
      }, intervalMs);
    }
  }

  function start() {
    if (timer !== null || controller !== null) return;
    void tick();
  }

  function stop() {
    if (timer !== null) clearTimeout(timer);
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

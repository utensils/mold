import { fetchEventSource } from "@microsoft/fetch-event-source";
import { onScopeDispose, ref, watch, type Ref } from "vue";
import { chainJobEventsUrl, getChainJob } from "../api";
import type { StreamTarget } from "../api";
import type { ChainJobDetail, ChainJobEvent } from "../types";

export function useChainJobStream(
  jobId: Ref<string | null>,
  target?: Ref<StreamTarget | undefined>,
): {
  detail: Ref<ChainJobDetail | null>;
  connected: Ref<boolean>;
  refresh: () => Promise<void>;
} {
  const detail = ref<ChainJobDetail | null>(null);
  const connected = ref(false);
  let abort: AbortController | null = null;
  let reconnectTimer: number | null = null;
  let pollTimer: number | null = null;
  let stopped = false;

  function clearTimers() {
    if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
    if (pollTimer !== null) window.clearTimeout(pollTimer);
    reconnectTimer = null;
    pollTimer = null;
  }

  function closeSource() {
    abort?.abort();
    abort = null;
    connected.value = false;
  }

  function applyEvent(event: ChainJobEvent) {
    const current = detail.value;
    switch (event.type) {
      case "snapshot":
        detail.value = event.job;
        break;
      case "stage_start":
        if (current) {
          current.current_stage = event.stage_idx;
          current.state = "running";
          const stage = current.stages.find((s) => s.idx === event.stage_idx);
          if (stage) stage.state = "running";
        }
        break;
      case "stage_done":
        if (current) {
          const stage = current.stages.find((s) => s.idx === event.stage_idx);
          if (stage) {
            stage.state = "completed";
            stage.frames_emitted = event.frames_emitted;
            stage.has_preview = event.has_preview;
          }
        }
        break;
      case "state_changed":
        if (current) {
          current.state = event.state;
          current.error = event.error;
        }
        break;
      case "finalized":
      case "finalizing":
      case "yielded":
      case "denoise_step":
        break;
    }
  }

  async function pollOnce(id: string) {
    try {
      detail.value = await getChainJob(id, target?.value);
    } catch {
      // The reconnect loop remains authoritative; a failed poll should not
      // tear down a stream that may come back.
    }
  }

  function scheduleReconnect(id: string) {
    if (stopped) return;
    clearTimers();
    pollTimer = window.setTimeout(() => void pollOnce(id), 250);
    reconnectTimer = window.setTimeout(() => connect(id), 1_000);
  }

  function connect(id: string) {
    closeSource();
    if (stopped) return;
    const controller = new AbortController();
    abort = controller;
    const route = target?.value;
    void fetchEventSource(chainJobEventsUrl(id, route), {
      signal: controller.signal,
      openWhenHidden: true,
      headers: route?.apiKey ? { "x-api-key": route.apiKey } : {},
      onopen: async (response) => {
        if (!response.ok)
          throw new Error(`Sequence events failed: ${response.status}`);
        connected.value = true;
      },
      onmessage: (message) => {
        if (message.event !== "chain_job") return;
        try {
          applyEvent(JSON.parse(message.data) as ChainJobEvent);
        } catch {
          // Bad frames are ignored; the poll fallback re-synchronizes state.
        }
      },
      onclose: () => {
        if (!controller.signal.aborted) scheduleReconnect(id);
      },
      onerror: () => {
        if (!controller.signal.aborted) scheduleReconnect(id);
      },
    }).catch(() => {
      if (!controller.signal.aborted) scheduleReconnect(id);
    });
  }

  async function refresh() {
    const id = jobId.value;
    if (id) await pollOnce(id);
  }

  watch(
    [jobId, () => target?.value?.baseUrl, () => target?.value?.apiKey],
    ([id]) => {
      clearTimers();
      closeSource();
      detail.value = null;
      if (id) {
        void pollOnce(id);
        connect(id);
      }
    },
    { immediate: true },
  );

  onScopeDispose(() => {
    stopped = true;
    clearTimers();
    closeSource();
  });

  return { detail, connected, refresh };
}

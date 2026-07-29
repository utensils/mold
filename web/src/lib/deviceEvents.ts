import { fetchEventSource } from "@microsoft/fetch-event-source";
import { apiHeaders, apiJsonTo, type ApiTarget } from "@studio/api/client";

type SnapshotEvent = {
  type?: unknown;
};

/**
 * Treat device and queue-plan events as invalidation hints. A full refresh on
 * every successful connection repairs frames dropped while disconnected.
 */
export function subscribeToDeviceSnapshots(
  target: ApiTarget,
  signal: AbortSignal,
  refresh: () => void,
): void {
  void (async () => {
    try {
      const capabilities = await apiJsonTo<{
        events?: { available?: boolean };
      }>(target, "/api/capabilities");
      if (signal.aborted || capabilities.events?.available !== true) return;

      await fetchEventSource(`${target.baseUrl}/api/events`, {
        method: "GET",
        headers: Object.fromEntries(apiHeaders(target).entries()),
        signal,
        openWhenHidden: true,
        onopen: async (response) => {
          if (!response.ok) {
            const error = new Error(
              `events stream failed: ${response.status}`,
            ) as Error & { status?: number };
            error.status = response.status;
            throw error;
          }
          refresh();
        },
        onmessage: (message) => {
          try {
            const event = JSON.parse(message.data) as SnapshotEvent;
            if (
              event.type === "device_state_changed" ||
              event.type === "queue_plan_changed"
            ) {
              refresh();
            }
          } catch {
            // A reconnect or the next valid event repairs a malformed frame.
          }
        },
        onclose: () => {
          if (!signal.aborted) throw new Error("events stream closed");
        },
        onerror: (error) => {
          const status = (error as { status?: number }).status;
          if (status === 401 || status === 403 || status === 404) throw error;
        },
      });
    } catch {
      // Capability discovery and terminal HTTP errors leave the current
      // bootstrap/poll snapshot intact without a permanent retry loop.
    }
  })();
}

import type { ApiTarget } from "./client";
import { sseStream } from "./sse";

type SnapshotEvent = {
  type?: unknown;
};

/**
 * Subscribe to device/plan invalidations for one explicit host.
 *
 * Event payloads are only invalidation hints. Consumers refetch `/api/devices`
 * and `/api/queue`, including after every successful SSE (re)connect, so a
 * dropped frame can never leave local state authoritative.
 */
export function subscribeToDeviceSnapshots(
  target: ApiTarget,
  signal: AbortSignal,
  refresh: () => void,
): void {
  void sseStream("/api/events", {
    target,
    signal,
    retry: true,
    onOpen: refresh,
    onEvent: (_event, data) => {
      try {
        const event = JSON.parse(data) as SnapshotEvent;
        if (event.type === "device_state_changed" || event.type === "queue_plan_changed") {
          refresh();
        }
      } catch {
        // The next valid event or reconnect snapshot repairs malformed frames.
      }
    },
  });
}

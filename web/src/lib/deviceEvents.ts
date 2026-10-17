import { originAuthenticatedFetch } from "./originAuth";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { apiHeaders, apiJsonTo, type ApiTarget } from "@studio/api/client";

type SnapshotEvent = {
  type?: unknown;
};

interface EventHintOptions {
  /** Does this frame (`type` of the JSON body, or the SSE event name)
   * invalidate the caller's snapshot? */
  matches: (type: string | null, eventName: string | undefined) => boolean;
  refresh: () => void;
  /** The stream opened (`true`) or ended for good (`false`). */
  onLive?: (live: boolean) => void;
}

/**
 * Treat matching `/api/events` frames as invalidation hints. A full refresh
 * on every successful connection repairs frames dropped while disconnected.
 */
function subscribeToEventHints(
  target: ApiTarget,
  signal: AbortSignal,
  options: EventHintOptions,
): void {
  const { matches, refresh, onLive } = options;
  void (async () => {
    try {
      const capabilities = await apiJsonTo<{
        events?: { available?: boolean };
      }>(target, "/api/capabilities");
      if (signal.aborted || capabilities.events?.available !== true) return;

      await fetchEventSource(`${target.baseUrl}/api/events`, {
        fetch: originAuthenticatedFetch,
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
          onLive?.(true);
          refresh();
        },
        onmessage: (message) => {
          if (matches(null, message.event)) {
            refresh();
            return;
          }
          try {
            const event = JSON.parse(message.data) as SnapshotEvent;
            if (
              typeof event.type === "string" &&
              matches(event.type, message.event)
            )
              refresh();
          } catch {
            // A reconnect or the next valid event repairs a malformed frame.
          }
        },
        onclose: () => {
          if (!signal.aborted) throw new Error("events stream closed");
        },
        onerror: (error) => {
          onLive?.(false);
          const status = (error as { status?: number }).status;
          if (status === 401 || status === 403 || status === 404) throw error;
        },
      });
    } catch {
      // Capability discovery and terminal HTTP errors leave the current
      // bootstrap/poll snapshot intact without a permanent retry loop.
    } finally {
      onLive?.(false);
    }
  })();
}

export function subscribeToDeviceSnapshots(
  target: ApiTarget,
  signal: AbortSignal,
  refresh: () => void,
): void {
  subscribeToEventHints(target, signal, {
    matches: (type) =>
      type === "device_state_changed" || type === "queue_plan_changed",
    refresh,
  });
}

/**
 * Gallery frames (`gallery_*`) and `resync_required` are TICKLES: they carry
 * no state the Library applies, they only say "pull the replica". `onLive`
 * reports whether the stream is up so the caller can relax its poll.
 */
export function subscribeToGalleryTickles(
  target: ApiTarget,
  signal: AbortSignal,
  tickle: () => void,
  onLive?: (live: boolean) => void,
): void {
  subscribeToEventHints(target, signal, {
    matches: (type, eventName) =>
      eventName === "resync_required" ||
      (type !== null && type.startsWith("gallery_")),
    refresh: tickle,
    ...(onLive ? { onLive } : {}),
  });
}

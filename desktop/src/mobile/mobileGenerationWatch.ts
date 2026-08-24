import type { ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";

export type MobileGenerationReconcileReason =
  "open" | "event" | "event_gap" | "instance_mismatch" | "malformed" | "close" | "wake";

export interface MobileGenerationEventAuthority {
  instanceId: string;
  missedEvents?: number;
}

export interface MobileGenerationHostWatchOptions {
  target: ApiTarget;
  expectedInstanceId: string;
  onReconcile: (reason: MobileGenerationReconcileReason) => void;
  onGap: (authority: MobileGenerationEventAuthority) => void;
  stream?: typeof sseStream;
}

export interface MobileGenerationHostWatch {
  wake(): void;
  stop(): void;
}

const INVALIDATION_EVENT_TYPES = new Set([
  "job_queued",
  "job_started",
  "job_ended",
  "job_state_committed",
  "gallery_added",
]);

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** One authenticated server-wide stream per host. Events are only hints: a
 * microtask-coalesced bulk status read remains the lifecycle authority. */
export function watchMobileGenerationHost(
  options: MobileGenerationHostWatchOptions,
): MobileGenerationHostWatch {
  const stream = options.stream ?? sseStream;
  const abort = new AbortController();
  let stopped = false;
  let reconcileQueued = false;

  function reconcile(reason: MobileGenerationReconcileReason): void {
    if (stopped || reconcileQueued) return;
    reconcileQueued = true;
    queueMicrotask(() => {
      reconcileQueued = false;
      if (!stopped) options.onReconcile(reason);
    });
  }

  void stream("/api/events", {
    target: options.target,
    signal: abort.signal,
    retry: true,
    terminalHttpStatuses: [401, 403, 404],
    onOpen: () => reconcile("open"),
    onEvent: (event, data) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(data);
      } catch {
        reconcile("malformed");
        return;
      }
      const payload = record(parsed);
      if (!payload) {
        reconcile("malformed");
        return;
      }
      if (event === "authority") {
        if (typeof payload.instance_id !== "string") {
          reconcile("malformed");
        } else if (payload.instance_id !== options.expectedInstanceId) {
          options.onGap({ instanceId: payload.instance_id });
          reconcile("instance_mismatch");
        }
        return;
      }
      if (event === "resync_required") {
        if (
          typeof payload.instance_id !== "string" ||
          !Number.isSafeInteger(payload.missed_events) ||
          Number(payload.missed_events) < 1
        ) {
          reconcile("malformed");
          return;
        }
        options.onGap({
          instanceId: payload.instance_id,
          missedEvents: Number(payload.missed_events),
        });
        reconcile(
          payload.instance_id === options.expectedInstanceId ? "event_gap" : "instance_mismatch",
        );
        return;
      }
      if (event !== "event" || typeof payload.type !== "string") {
        reconcile("malformed");
        return;
      }
      if (INVALIDATION_EVENT_TYPES.has(payload.type)) reconcile("event");
    },
    onClose: () => reconcile("close"),
  });

  return {
    wake() {
      reconcile("wake");
    },
    stop() {
      if (stopped) return;
      stopped = true;
      abort.abort();
    },
  };
}

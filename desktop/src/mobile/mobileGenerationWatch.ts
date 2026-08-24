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
  /** Undefined job ids request one host-wide authority read. */
  onReconcile: (reason: MobileGenerationReconcileReason, jobIds?: ReadonlySet<string>) => void;
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
  let queuedReason: MobileGenerationReconcileReason = "event";
  let queuedJobIds: Set<string> | null = new Set();
  let postCommitEventsAvailable = false;

  function reconcile(reason: MobileGenerationReconcileReason, jobId?: string): void {
    if (stopped) return;
    if (reconcileQueued) {
      if (jobId === undefined) queuedJobIds = null;
      else queuedJobIds?.add(jobId);
      return;
    }
    reconcileQueued = true;
    queuedReason = reason;
    queuedJobIds = jobId === undefined ? null : new Set([jobId]);
    queueMicrotask(() => {
      reconcileQueued = false;
      const ids = queuedJobIds;
      queuedJobIds = new Set();
      if (!stopped) options.onReconcile(queuedReason, ids ?? undefined);
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
      if (payload.type === "job_state_committed" && typeof payload.id === "string") {
        postCommitEventsAvailable = true;
        reconcile("event", payload.id);
      } else if (payload.type === "generation_states_committed") {
        postCommitEventsAvailable = true;
        reconcile("event");
      } else if (payload.type === "gallery_added" && postCommitEventsAvailable) {
        // This exact stream proved that a correctly ordered commit hint follows.
      } else if (INVALIDATION_EVENT_TYPES.has(payload.type)) {
        reconcile("event", typeof payload.id === "string" ? payload.id : undefined);
      }
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

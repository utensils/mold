/**
 * Per-GPU queue lane computation for a machine's queue panel — the desktop
 * port of the web SPA's RunningStrip lane logic
 * (web/src/components/RunningStrip.vue). Pure functions so lane grouping and
 * the drag/drop policy are unit-testable.
 */

import { compareNewestQueueEntry } from "@studio/lib/activityOrder";

export type LaneKey = number | "queue";

/** The subset of a queue row the lane computation needs. */
export interface LaneEntryLike {
  id: string;
  /** `held` is additive: a journalled job the host parked after exhausting its
   * replay or dispatch budget. It occupies no lane and waits for no turn. */
  state: "queued" | "running" | "paused" | "held";
  position: number;
  started_at_unix_ms: number;
  /** GPU ordinal currently running this row (running rows only). */
  gpu?: number;
  /** Requested GPU ordinal for queued rows; absent = Auto. */
  target_gpu?: number;
}

export interface QueueLane<T extends LaneEntryLike> {
  key: LaneKey;
  label: string;
  entries: T[];
}

/** GPU lane a queue row belongs to for display; null = Auto/unassigned. */
export function laneForEntry(entry: LaneEntryLike): number | null {
  if (entry.state === "running") return entry.gpu ?? null;
  return entry.target_gpu ?? null;
}

function laneOrder<T extends LaneEntryLike>(a: T, b: T): number {
  return compareNewestQueueEntry(a, b);
}

/**
 * Group a host's queue rows into per-GPU lanes. Ordinals are the union of the
 * host-reported worker GPUs and any lane a row already references. With fewer
 * than two known ordinals the whole queue stays one flat `"queue"` lane —
 * zero visual change for single-GPU hosts. Auto rows (no target_gpu) fall
 * into the default (lowest-ordinal) lane.
 */
export function computeQueueLanes<T extends LaneEntryLike>(
  entries: T[],
  gpuOrdinals: number[],
): QueueLane<T>[] {
  const ordinals = new Set<number>(gpuOrdinals);
  for (const entry of entries) {
    const lane = laneForEntry(entry);
    if (lane !== null) ordinals.add(lane);
  }
  const keys = Array.from(ordinals).sort((a, b) => a - b);
  if (keys.length < 2) {
    return [{ key: "queue", label: "Queue", entries: [...entries].sort(laneOrder) }];
  }
  const defaultLane = keys[0]!;
  return keys.map((key) => ({
    key,
    label: `GPU ${key}`,
    entries: entries.filter((e) => (laneForEntry(e) ?? defaultLane) === key).sort(laneOrder),
  }));
}

/** A drag in flight: which host owns the row and where it currently sits. */
export interface DraggedQueueRow {
  hostId: string;
  entryId: string;
  /** Lane the row is displayed in right now; null = Auto. */
  lane: number | null;
}

export type DropAction =
  | { kind: "reassign"; hostId: string; entryId: string; targetGpu: number }
  | { kind: "reject-cross-host" }
  | { kind: "none" };

/**
 * Drop policy for lane drags. Jobs can never migrate between servers — a
 * cross-host drop is explicitly rejected (distinct from a plain no-op) so the
 * UI can explain why nothing happened.
 */
export function resolveDropAction(
  dragged: DraggedQueueRow | null,
  dropHostId: string,
  laneKey: LaneKey,
): DropAction {
  if (!dragged || laneKey === "queue") return { kind: "none" };
  if (dragged.hostId !== dropHostId) return { kind: "reject-cross-host" };
  if (dragged.lane === laneKey) return { kind: "none" };
  return {
    kind: "reassign",
    hostId: dropHostId,
    entryId: dragged.entryId,
    targetGpu: laneKey,
  };
}

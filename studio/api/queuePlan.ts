import { IncompatibleHostError, apiJsonTo, type ApiTarget } from "./client";

export type EstimateConfidence = "low" | "medium" | "high";

export interface QueueEntry {
  id: string;
  model: string;
  state: string;
  started_at_unix_ms: number;
  position: number;
  gpu?: number;
  target_gpu?: number | null;
}

export interface QueueWorkItem {
  work_id: string;
  parent_id: string;
  work_kind: string;
  priority_class: string;
  queue_rank: number;
  bypass_count: number;
  gpu?: number | null;
  hard_pinned_device_id?: string | null;
  target_gpu?: number | null;
  planned_device_id?: string | null;
  lane_order?: number | null;
  estimated_start_unix_ms?: number | null;
  estimated_finish_unix_ms?: number | null;
  estimate_confidence: EstimateConfidence;
  reason?: string | null;
}

export interface QueuePlan {
  plan_version: number;
  state_version: number;
  optimizer_state: string;
  dirty_since_unix_ms: number | null;
  next_replan_at_unix_ms: number | null;
  work_items: QueueWorkItem[];
}

export interface QueueListing {
  entries: QueueEntry[];
  plan: QueuePlan | null;
}

export type QueuePlanChangedEvent = {
  type: "queue_plan_changed";
  plan: QueuePlan;
};

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function parseQueueListing(value: unknown): QueueListing {
  const root = record(value);
  if (!root || !Array.isArray(root.entries)) {
    throw new IncompatibleHostError(["entries"]);
  }
  const entries = root.entries.map((raw, index) => {
    const row = record(raw);
    if (
      !row ||
      typeof row.id !== "string" ||
      typeof row.model !== "string" ||
      typeof row.state !== "string" ||
      typeof row.started_at_unix_ms !== "number" ||
      typeof row.position !== "number"
    ) {
      throw new IncompatibleHostError([`entries[${index}]`]);
    }
    return row as unknown as QueueEntry;
  });
  return {
    entries,
    plan: root.plan == null ? null : parseQueuePlan(root.plan),
  };
}

export function parseQueuePlan(value: unknown): QueuePlan {
  const plan = record(value);
  if (
    !plan ||
    typeof plan.plan_version !== "number" ||
    typeof plan.state_version !== "number" ||
    typeof plan.optimizer_state !== "string" ||
    !Array.isArray(plan.work_items)
  ) {
    throw new IncompatibleHostError(["plan"]);
  }
  return {
    plan_version: plan.plan_version,
    state_version: plan.state_version,
    optimizer_state: plan.optimizer_state,
    dirty_since_unix_ms: nullableNumber(plan.dirty_since_unix_ms),
    next_replan_at_unix_ms: nullableNumber(plan.next_replan_at_unix_ms),
    work_items: plan.work_items as QueueWorkItem[],
  };
}

export async function listQueue(target: ApiTarget): Promise<QueueListing> {
  return parseQueueListing(await apiJsonTo<unknown>(target, "/api/queue"));
}

/** Set or clear a durable stable-device pin for queued work. */
export async function setQueueDevicePin(
  target: ApiTarget,
  workId: string,
  deviceId: string | null,
): Promise<QueueEntry> {
  return apiJsonTo<QueueEntry>(
    target,
    `/api/queue/${encodeURIComponent(workId)}`,
    {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ hard_pinned_device_id: deviceId }),
    },
  );
}

export function reduceQueuePlanEvent(
  current: QueueListing,
  event: QueuePlanChangedEvent,
): QueueListing {
  if (current.plan && current.plan.plan_version >= event.plan.plan_version) {
    return current;
  }
  return { ...current, plan: event.plan };
}

export function predictedCompletionUnixMs(
  plan: QueuePlan | null,
  nowUnixMs = Date.now(),
): number {
  return Math.max(
    nowUnixMs,
    ...(plan?.work_items
      .map((item) => item.estimated_finish_unix_ms)
      .filter((value): value is number => typeof value === "number") ?? []),
  );
}

import {
  IncompatibleHostError,
  apiFetchTo,
  apiJsonTo,
  parseCurrentServerStatus,
  type ApiTarget,
} from "./client";
import { parseHostMemory, type HostMemorySnapshot } from "../lib/hostMemory";
import {
  getGenerationBatch,
  isDefiniteGenerationAdmissionRejection,
  type GenerationBatchStatus,
} from "./generationAdmission";

export type EstimateConfidence = "low" | "medium" | "high" | (string & {});
export type QueueLaneKind = "device" | "host_utility" | (string & {});

export interface QueueEntry {
  id: string;
  model: string;
  state: string;
  started_at_unix_ms: number;
  position: number;
  gpu?: number;
  target_gpu?: number | null;
  /** Additive request settings exposed by current servers. Kept unknown in
   * studio because desktop and web own structurally compatible metadata types. */
  metadata?: unknown;
  seed_pinned?: boolean | null;
  durable?: boolean | null;
  /** Whether this row was resumed from the journal rather than submitted by a
   * live client (additive; absent on older servers). */
  replayed?: boolean | null;
  /** How many times a worker has claimed this row for execution. Diagnoses a
   * held row (additive; absent on older servers). */
  dispatch_attempts?: number | null;
  held_reason?: string | null;
  /** Durable preparation error. Present on held protocol-v2 rows. */
  error?: string | null;
  /** Exact opt-in fence for POST /api/queue/{id}/retry. */
  retryable?: boolean | null;
}

export interface QueueWorkItem {
  work_id: string;
  parent_id: string;
  work_kind: string;
  chain_stage?: number | null;
  batch_partition?: {
    index: number;
    count: number;
    size: number;
  } | null;
  priority_class: string;
  queue_rank: number;
  bypass_count: number;
  gpu?: number | null;
  hard_pinned_device_id?: string | null;
  target_gpu?: number | null;
  planned_device_id?: string | null;
  planned_lane_kind?: QueueLaneKind | null;
  lane_order?: number | null;
  estimated_start_unix_ms?: number | null;
  estimated_finish_unix_ms?: number | null;
  estimate_confidence: EstimateConfidence;
  reason?: string | null;
  blocked_reason?: string | null;
  assignment_reason?: string | null;
  warm_wait_deadline_unix_ms?: number | null;
  activity_phase?:
    | "queued"
    | "blocked"
    | "warm_wait"
    | "dispatching"
    | "active"
    | "cpu"
    | (string & {});
  execution_fingerprint?: string | null;
}

export interface QueuePlan {
  plan_version: number;
  state_version: number;
  optimizer_state: string;
  dirty_since_unix_ms: number | null;
  next_replan_at_unix_ms: number | null;
  work_items: QueueWorkItem[];
  /** Host-RAM ledger snapshot (additive; absent on older servers). */
  host_memory?: HostMemorySnapshot | null;
}

export interface QueueListing {
  entries: QueueEntry[];
  plan: QueuePlan | null;
  /** Active rows without durable backing. Repeated on every explicit page. */
  live_only_entries?: QueueEntry[];
  /** Present only when the caller explicitly requested a durable page. */
  page?: QueuePage;
}

export interface QueuePageRequest {
  limit: number;
  cursor?: string;
}

export interface QueuePage {
  limit: number;
  offset: number;
  returned: number;
  next_cursor?: string;
}

/** Current servers expose the runtime queue capacity in `/api/status`. That
 * is the authoritative amount of live work a hot client needs in one frame,
 * and therefore the only honest default for a payload-free durable page.
 * `undefined` keeps old hosts on their legacy endpoint; callers must never
 * substitute a guessed cap. */
export function queuePageRequestForCapacity(
  capacity: unknown,
): QueuePageRequest | undefined {
  return typeof capacity === "number" &&
    Number.isFinite(capacity) &&
    Number.isInteger(capacity) &&
    capacity > 0
    ? { limit: capacity }
    : undefined;
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

function queueEntries(value: unknown, field: string): QueueEntry[] {
  if (!Array.isArray(value)) throw new IncompatibleHostError([field]);
  return value.map((raw, index) => {
    const row = record(raw);
    if (
      !row ||
      typeof row.id !== "string" ||
      typeof row.model !== "string" ||
      typeof row.state !== "string" ||
      typeof row.started_at_unix_ms !== "number" ||
      typeof row.position !== "number"
    ) {
      throw new IncompatibleHostError([`${field}[${index}]`]);
    }
    return row as unknown as QueueEntry;
  });
}

function nonnegativeInteger(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    Number.isInteger(value) &&
    value >= 0
  );
}

function parseQueuePage(value: unknown): QueuePage {
  const page = record(value);
  const missing: string[] = [];
  if (!page || !nonnegativeInteger(page.limit) || page.limit === 0) {
    missing.push("page.limit");
  }
  if (!page || !nonnegativeInteger(page.offset)) {
    missing.push("page.offset");
  }
  if (
    !page ||
    !nonnegativeInteger(page.returned) ||
    (nonnegativeInteger(page.limit) && page.returned > page.limit)
  ) {
    missing.push("page.returned");
  }
  if (
    page?.next_cursor !== undefined &&
    (typeof page.next_cursor !== "string" || page.next_cursor.length === 0)
  ) {
    missing.push("page.next_cursor");
  }
  if (!page || missing.length > 0) throw new IncompatibleHostError(missing);
  return {
    limit: page.limit as number,
    offset: page.offset as number,
    returned: page.returned as number,
    ...(typeof page.next_cursor === "string"
      ? { next_cursor: page.next_cursor }
      : {}),
  };
}

/** Build an explicit queue-page path without interpreting the opaque cursor. */
export function queueListingPath(request?: QueuePageRequest): string {
  if (!request) return "/api/queue";
  if (
    !Number.isFinite(request.limit) ||
    !Number.isInteger(request.limit) ||
    request.limit <= 0
  ) {
    throw new RangeError("queue page limit must be a positive integer");
  }
  if (
    request.cursor !== undefined &&
    (typeof request.cursor !== "string" || request.cursor.length === 0)
  ) {
    throw new TypeError("queue page cursor must be a nonempty string");
  }
  const query = new URLSearchParams({ limit: String(request.limit) });
  if (request.cursor !== undefined) query.set("cursor", request.cursor);
  return `/api/queue?${query.toString()}`;
}

export function parseQueueListing(value: unknown): QueueListing {
  const root = record(value);
  if (!root || !Array.isArray(root.entries)) {
    throw new IncompatibleHostError(["entries"]);
  }
  const entries = queueEntries(root.entries, "entries");
  return {
    entries,
    plan: root.plan == null ? null : parseQueuePlan(root.plan),
    ...(root.live_only_entries === undefined
      ? {}
      : {
          live_only_entries: queueEntries(
            root.live_only_entries,
            "live_only_entries",
          ),
        }),
    ...(root.page === undefined ? {} : { page: parseQueuePage(root.page) }),
  };
}

/** Keep durable page order, then append the first occurrence of each live-only
 * id. Callers may flatten live-only rows from several pages before merging. */
export function mergeQueueEntries(
  entries: readonly QueueEntry[],
  liveOnlyEntries: readonly QueueEntry[],
): QueueEntry[] {
  const seen = new Set(entries.map(({ id }) => id));
  const merged = [...entries];
  for (const entry of liveOnlyEntries) {
    if (seen.has(entry.id)) continue;
    seen.add(entry.id);
    merged.push(entry);
  }
  return merged;
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
    host_memory: parseHostMemory(plan.host_memory),
  };
}

export async function listQueue(
  target: ApiTarget,
  /** undefined discovers the host capacity first; null records that the
   * caller already observed a legacy status response with no capacity. */
  page?: QueuePageRequest | null,
  signal?: AbortSignal,
): Promise<QueueListing> {
  let request = page ?? undefined;
  if (page === undefined) {
    const status =
      signal === undefined
        ? await apiJsonTo<unknown>(target, "/api/status")
        : await apiJsonTo<unknown>(target, "/api/status", { signal });
    request = queuePageRequestForCapacity(record(status)?.queue_capacity);
  }
  const path = queueListingPath(request);
  const value =
    signal === undefined
      ? await apiJsonTo<unknown>(target, path)
      : await apiJsonTo<unknown>(target, path, { signal });
  return parseQueueListing(value);
}

/** Resolve one queue row for an explicit user action without ever asking the
 * server for its legacy all-rows response. This may traverse every bounded
 * page, but only on demand (for example, opening one activity row), never from
 * a health or polling loop. */
export async function findQueueEntryById(
  target: ApiTarget,
  id: string,
  signal?: AbortSignal,
): Promise<QueueEntry | null> {
  let listing = await listQueue(target, undefined, signal);
  const seenCursors = new Set<string>();
  for (;;) {
    const match = mergeQueueEntries(
      listing.entries,
      listing.live_only_entries ?? [],
    ).find((entry) => entry.id === id);
    if (match) return match;
    const cursor = listing.page?.next_cursor;
    const limit = listing.page?.limit;
    if (!cursor || !limit) return null;
    if (seenCursors.has(cursor)) {
      throw new Error("host repeated a queue continuation cursor");
    }
    seenCursors.add(cursor);
    listing = await listQueue(target, { limit, cursor }, signal);
  }
}

/** Cancel work that is still waiting in the explicit host's generation queue. */
export async function cancelQueueJob(
  target: ApiTarget,
  workId: string,
): Promise<void> {
  await apiFetchTo(target, `/api/queue/${encodeURIComponent(workId)}`, {
    method: "DELETE",
  });
}

export interface QueueJobAuthority {
  instanceId: string;
  batchId: string;
  clientBatchId: string;
  jobId: string;
}

/** Resume one explicitly retryable durable hold on the exact host. The body
 * repeats the complete captured authority so the server can fence the
 * mutation transaction rather than trusting a preceding status read. */
export async function retryQueueJob(
  target: ApiTarget,
  authority: QueueJobAuthority,
): Promise<void> {
  await apiFetchTo(
    target,
    `/api/queue/${encodeURIComponent(authority.jobId)}/retry`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        instance_id: authority.instanceId,
        batch_id: authority.batchId,
        client_batch_id: authority.clientBatchId,
        job_id: authority.jobId,
      }),
    },
  );
}

export type RetryQueueJobOutcome =
  | { kind: "accepted" }
  | { kind: "reconciled"; batch: GenerationBatchStatus }
  | { kind: "uncertain"; error: string };

const AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS = 5;
const AMBIGUOUS_RETRY_CONFIRM_DELAY_MS = 1_000;

function validatedRetryChild(
  batch: GenerationBatchStatus,
  authority: QueueJobAuthority,
) {
  const child = batch.children.find(
    (candidate) => candidate.job_id === authority.jobId,
  );
  if (
    batch.instance_id !== authority.instanceId ||
    batch.id !== authority.batchId ||
    batch.client_batch_id !== authority.clientBatchId ||
    !child
  ) {
    throw new Error(
      "The retry reconciliation response did not match its captured authority.",
    );
  }
  return child;
}

function retryConfirmationDelay(): Promise<void> {
  return new Promise((resolve) =>
    setTimeout(resolve, AMBIGUOUS_RETRY_CONFIRM_DELAY_MS),
  );
}

/** Retry once, then recover a lost/invalid response only through the captured
 * batch authority. An uncertain mutation is never permission to send a second
 * retry POST. */
export async function retryQueueJobRecoveringAmbiguity(
  target: ApiTarget,
  authority: QueueJobAuthority,
): Promise<RetryQueueJobOutcome> {
  try {
    await retryQueueJob(target, authority);
    return { kind: "accepted" };
  } catch (error) {
    if (isDefiniteGenerationAdmissionRejection(error)) throw error;
    const detail = error instanceof Error ? error.message : String(error);
    let pendingDetail = detail;
    for (
      let attempt = 0;
      attempt < AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS;
      attempt += 1
    ) {
      let lookup;
      try {
        lookup = await getGenerationBatch(target, authority.batchId);
      } catch (lookupError) {
        const lookupDetail =
          lookupError instanceof Error
            ? lookupError.message
            : String(lookupError);
        pendingDetail = `${detail}; exact retry reconciliation failed: ${lookupDetail}`;
        if (
          isDefiniteGenerationAdmissionRejection(lookupError) ||
          attempt + 1 === AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
        ) {
          return { kind: "uncertain", error: pendingDetail };
        }
        await retryConfirmationDelay();
        continue;
      }
      if (lookup.kind === "missing") {
        pendingDetail = `${detail}; the durable batch lookup is still missing`;
        if (attempt + 1 === AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS) {
          return { kind: "uncertain", error: pendingDetail };
        }
        await retryConfirmationDelay();
        continue;
      }
      const batch = lookup.batch;
      const child = validatedRetryChild(batch, authority);
      if (
        child.state !== "held" ||
        attempt + 1 === AMBIGUOUS_RETRY_CONFIRM_ATTEMPTS
      ) {
        return { kind: "reconciled", batch };
      }
      await retryConfirmationDelay();
    }
    return { kind: "uncertain", error: pendingDetail };
  }
}

export type QueueJobMutation = "cancel" | "retry";

/** Mutate durable work only after the target proves it is still the server
 * instance that admitted the job. URLs and credentials can outlive a server
 * replacement, so neither is sufficient authority for a recovered job. */
export async function mutateQueueJobOnExpectedInstance(
  target: ApiTarget,
  authority: QueueJobAuthority,
  mutation: QueueJobMutation,
): Promise<void> {
  if (!authority.instanceId.trim()) {
    throw new TypeError("queue mutation requires an expected server instance");
  }
  const status = parseCurrentServerStatus(
    await apiJsonTo<unknown>(target, "/api/status"),
  );
  if (status.instance_id !== authority.instanceId) {
    throw new Error(
      "The original machine identity changed; this queue action is unavailable.",
    );
  }
  if (mutation === "cancel") {
    await cancelQueueJob(target, authority.jobId);
  } else {
    await retryQueueJob(target, authority);
  }
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
): number | null {
  const finiteFinishes =
    plan?.work_items
      .map((item) => item.estimated_finish_unix_ms)
      .filter(
        (value): value is number =>
          typeof value === "number" && Number.isFinite(value),
      ) ?? [];
  if (finiteFinishes.length === 0) return null;
  return Math.max(nowUnixMs, ...finiteFinishes);
}

import type {
  GenerationBatchChild,
  GenerationBatchStatus,
  GenerationBatchStatusRequest,
  GenerationBatchStatusResponse,
  GenerationLifecyclePhase,
} from "../api/generationAdmission";

export type GenerationAdmissionPhase =
  "pending" | "uncertain" | "confirmed" | "rejected";

export type GenerationLookupResolution = "unchecked" | "found" | "missing";

export interface GenerationAdmissionState {
  phase: GenerationAdmissionPhase;
  lookup: GenerationLookupResolution;
  submittedAtMs: number;
  error: string | null;
}

export type GenerationReconciliationReason =
  | "event_gap"
  | "instance_mismatch"
  | "batch_mismatch"
  | "missing"
  | "incomplete_response";

export interface GenerationReconciliationState {
  required: boolean;
  reason: GenerationReconciliationReason | null;
}

export interface GenerationAuthority {
  hostId: string;
  instanceId: string;
  batchId: string;
  jobId: string;
}

export interface GenerationLifecycleVersion {
  updatedAtMs: number;
  revision: number | null;
}

export interface GenerationLifecycleJob {
  key: string;
  authority: GenerationAuthority;
  clientBatchId: string;
  childIndex: number;
  phase: GenerationLifecyclePhase;
  retryable: boolean | null;
  createdAtMs: number;
  completedAtMs: number | null;
  version: GenerationLifecycleVersion;
  error: string | null;
  /** Typed cause of a held child; `null` when the machine gave none. */
  errorCode: string | null;
  terminalError: unknown;
  result: { filename?: string; originalFilename?: string } | null;
}

/**
 * Persistable recovery state. It intentionally has no route URL, API key,
 * request body, upload handle, or media bytes; surfaces resolve secrets from
 * their native stores only after the instance fence passes.
 */
export interface GenerationBatchTracker {
  hostId: string;
  expectedInstanceId: string;
  clientBatchId: string;
  serverBatchId: string | null;
  admission: GenerationAdmissionState;
  reconciliation: GenerationReconciliationState;
  jobs: Record<string, GenerationLifecycleJob>;
}

export interface CreateGenerationBatchTrackerInput {
  hostId: string;
  expectedInstanceId: string;
  clientBatchId: string;
  submittedAtMs: number;
}

export type GenerationLifecycleAction =
  | { type: "admission_uncertain"; error: string }
  | { type: "admission_rejected"; error: string }
  | {
      type: "batch_snapshot";
      batch: GenerationBatchStatus;
      revision?: number | null;
    }
  | {
      type: "job_event";
      instanceId: string;
      batchId: string;
      clientBatchId: string;
      child: GenerationBatchChild;
      revision?: number | null;
    }
  | { type: "event_gap"; instanceId: string }
  | { type: "lookup_missing"; batchId?: string | null };

const TERMINAL_PHASES = new Set<GenerationLifecyclePhase>([
  "complete",
  "failed",
  "cancelled",
]);

const FORWARD_PHASE_RANK: Record<GenerationLifecyclePhase, number> = {
  accepted: 0,
  queued: 1,
  held: 2,
  running: 3,
  cancelling: 4,
  complete: 5,
  failed: 5,
  cancelled: 5,
};

export function isTerminalGenerationPhase(
  phase: GenerationLifecyclePhase,
): boolean {
  return TERMINAL_PHASES.has(phase);
}

function keyPart(value: string): string {
  return `${value.length}:${value}`;
}

/** Collision-free authority key across hosts, server replacements and jobs. */
export function generationAuthorityKey(authority: GenerationAuthority): string {
  return [
    authority.hostId,
    authority.instanceId,
    authority.batchId,
    authority.jobId,
  ]
    .map(keyPart)
    .join("|");
}

export function createGenerationBatchTracker(
  input: CreateGenerationBatchTrackerInput,
): GenerationBatchTracker {
  return {
    hostId: input.hostId,
    expectedInstanceId: input.expectedInstanceId,
    clientBatchId: input.clientBatchId,
    serverBatchId: null,
    admission: {
      phase: "pending",
      lookup: "unchecked",
      submittedAtMs: input.submittedAtMs,
      error: null,
    },
    reconciliation: { required: false, reason: null },
    jobs: {},
  };
}

function requireReconciliation(
  state: GenerationBatchTracker,
  reason: GenerationReconciliationReason,
): GenerationBatchTracker {
  if (state.reconciliation.required && state.reconciliation.reason === reason) {
    return state;
  }
  return { ...state, reconciliation: { required: true, reason } };
}

function isStrictlyNewer(
  incoming: GenerationLifecycleVersion,
  current: GenerationLifecycleVersion,
): boolean {
  if (incoming.revision !== null && current.revision !== null) {
    return incoming.revision > current.revision;
  }
  return incoming.updatedAtMs > current.updatedAtMs;
}

function equivalentJob(
  left: GenerationLifecycleJob,
  right: GenerationLifecycleJob,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

/**
 * A child's own revision, or `null` when it has no revision authority.
 *
 * `0` is deliberately not a revision: rows admitted before the server grew
 * the column sit at `0` until their next transition, so reading it as one
 * would let a pre-migration snapshot outrank nothing and be outranked by
 * nothing. Both absence and `0` fall back to the timestamp comparison.
 */
function childRevision(child: GenerationBatchChild): number | null {
  const revision = child.revision;
  if (typeof revision !== "number" || !Number.isFinite(revision)) return null;
  return revision > 0 ? revision : null;
}

function lifecycleJob(
  state: GenerationBatchTracker,
  instanceId: string,
  batchId: string,
  child: GenerationBatchChild,
  revision: number | null,
): GenerationLifecycleJob {
  const authority = {
    hostId: state.hostId,
    instanceId,
    batchId,
    jobId: child.job_id,
  };
  return {
    key: generationAuthorityKey(authority),
    authority,
    clientBatchId: state.clientBatchId,
    childIndex: child.index,
    phase: child.state,
    retryable: child.retryable ?? null,
    createdAtMs: child.created_at_ms,
    completedAtMs: child.completed_at_ms ?? null,
    version: {
      updatedAtMs: child.updated_at_ms,
      revision: revision ?? childRevision(child),
    },
    error: child.error ?? null,
    errorCode: child.error_code ?? null,
    terminalError: child.terminal_error ?? null,
    result: child.result
      ? {
          ...(child.result.filename ? { filename: child.result.filename } : {}),
          ...(child.result.original_filename
            ? { originalFilename: child.result.original_filename }
            : {}),
        }
      : null,
  };
}

function mergeJob(
  state: GenerationBatchTracker,
  incoming: GenerationLifecycleJob,
): GenerationBatchTracker {
  const current = state.jobs[incoming.key];
  if (!current) {
    return { ...state, jobs: { ...state.jobs, [incoming.key]: incoming } };
  }
  // A terminal transition is an effect boundary. Once observed, no replayed
  // event or later snapshot can mutate it or emit it a second time.
  if (isTerminalGenerationPhase(current.phase)) return state;

  const newer = isStrictlyNewer(incoming.version, current.version);
  const sameVersion =
    incoming.version.updatedAtMs === current.version.updatedAtMs &&
    incoming.version.revision === current.version.revision;
  const forwardAtSameVersion =
    sameVersion &&
    FORWARD_PHASE_RANK[incoming.phase] > FORWARD_PHASE_RANK[current.phase];
  if (!newer && !forwardAtSameVersion) return state;

  if (equivalentJob(current, incoming)) return state;
  return {
    ...state,
    jobs: { ...state.jobs, [incoming.key]: incoming },
  };
}

function attachBatchSnapshot(
  state: GenerationBatchTracker,
  batch: GenerationBatchStatus,
  revision: number | null,
): GenerationBatchTracker {
  if (batch.instance_id !== state.expectedInstanceId) {
    return requireReconciliation(state, "instance_mismatch");
  }
  if (batch.client_batch_id !== state.clientBatchId) {
    return requireReconciliation(state, "batch_mismatch");
  }
  if (state.serverBatchId !== null && state.serverBatchId !== batch.id) {
    return requireReconciliation(state, "batch_mismatch");
  }

  let next: GenerationBatchTracker = {
    ...state,
    serverBatchId: batch.id,
    admission: {
      ...state.admission,
      phase: "confirmed",
      lookup: "found",
      error: null,
    },
    reconciliation: { required: false, reason: null },
  };
  for (const child of batch.children) {
    next = mergeJob(
      next,
      lifecycleJob(next, batch.instance_id, batch.id, child, revision),
    );
  }
  return next;
}

export function reduceGenerationLifecycle(
  state: GenerationBatchTracker,
  action: GenerationLifecycleAction,
): GenerationBatchTracker {
  switch (action.type) {
    case "admission_uncertain":
      if (
        state.admission.phase === "confirmed" ||
        state.admission.phase === "rejected"
      ) {
        return state;
      }
      return {
        ...state,
        admission: {
          ...state.admission,
          phase: "uncertain",
          lookup: "unchecked",
          error: action.error,
        },
      };
    case "admission_rejected":
      if (
        state.admission.phase === "confirmed" ||
        state.admission.phase === "rejected"
      ) {
        return state;
      }
      return {
        ...state,
        admission: {
          ...state.admission,
          phase: "rejected",
          error: action.error,
        },
      };
    case "batch_snapshot":
      return attachBatchSnapshot(state, action.batch, action.revision ?? null);
    case "event_gap":
      return requireReconciliation(
        state,
        action.instanceId === state.expectedInstanceId
          ? "event_gap"
          : "instance_mismatch",
      );
    case "lookup_missing": {
      const next = {
        ...state,
        admission: { ...state.admission, lookup: "missing" as const },
      };
      if (
        state.serverBatchId !== null ||
        (action.batchId != null && action.batchId === state.serverBatchId)
      ) {
        return requireReconciliation(next, "missing");
      }
      return next;
    }
    case "job_event": {
      if (action.instanceId !== state.expectedInstanceId) {
        return requireReconciliation(state, "instance_mismatch");
      }
      if (
        action.clientBatchId !== state.clientBatchId ||
        state.serverBatchId === null ||
        action.batchId !== state.serverBatchId
      ) {
        return requireReconciliation(state, "batch_mismatch");
      }
      // Events after a known gap cannot repair the missing interval. Only an
      // authoritative snapshot clears this fence.
      if (state.reconciliation.required) return state;
      return mergeJob(
        state,
        lifecycleJob(
          state,
          action.instanceId,
          action.batchId,
          action.child,
          action.revision ?? null,
        ),
      );
    }
  }
}

export interface BulkGenerationMergeResult {
  trackers: GenerationBatchTracker[];
  missingClientBatchIds: string[];
  missingBatchIds: string[];
}

/**
 * Build the one bulk reconciliation request for a host. Known batches use
 * their server IDs; ambiguous admissions use the client idempotency IDs.
 * Input order is retained. Callers split trackers with
 * `chunkGenerationBatchTrackers` before sending each bounded request.
 */
export function buildGenerationBatchStatusRequest(
  trackers: readonly GenerationBatchTracker[],
  hostId: string,
): GenerationBatchStatusRequest {
  const clientBatchIds: string[] = [];
  const batchIds: string[] = [];
  const seenClients = new Set<string>();
  const seenBatches = new Set<string>();
  for (const tracker of trackers) {
    if (tracker.hostId !== hostId) continue;
    if (tracker.serverBatchId !== null) {
      if (!seenBatches.has(tracker.serverBatchId)) {
        seenBatches.add(tracker.serverBatchId);
        batchIds.push(tracker.serverBatchId);
      }
    } else if (!seenClients.has(tracker.clientBatchId)) {
      seenClients.add(tracker.clientBatchId);
      clientBatchIds.push(tracker.clientBatchId);
    }
  }
  return {
    client_batch_ids: clientBatchIds,
    ...(batchIds.length === 0 ? {} : { batch_ids: batchIds }),
  };
}

/** Server status reconciliation is deliberately bounded so one client cannot
 * monopolize queue persistence. Surfaces process every chunk in order. */
export const GENERATION_BATCH_STATUS_IDENTITY_LIMIT = 256;

export function chunkGenerationBatchTrackers(
  trackers: readonly GenerationBatchTracker[],
  hostId: string,
  limit = GENERATION_BATCH_STATUS_IDENTITY_LIMIT,
): GenerationBatchTracker[][] {
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new Error("generation batch status limit must be a positive integer");
  }
  const matching = trackers.filter((tracker) => tracker.hostId === hostId);
  const chunks: GenerationBatchTracker[][] = [];
  for (let offset = 0; offset < matching.length; offset += limit) {
    chunks.push(matching.slice(offset, offset + limit));
  }
  return chunks;
}

/**
 * Merge a successful bulk response while preserving the caller's tracker
 * order. Transport errors never enter this function, so they cannot be
 * confused with the server's explicit `missing` lists.
 */
export function mergeBulkGenerationBatchResponse(
  trackers: readonly GenerationBatchTracker[],
  hostId: string,
  response: GenerationBatchStatusResponse,
): BulkGenerationMergeResult {
  const byClient = new Map(
    response.batches.map((batch) => [batch.client_batch_id, batch]),
  );
  const byServer = new Map(response.batches.map((batch) => [batch.id, batch]));
  const missingClients = new Set(response.missing.client_batch_ids);
  const missingServers = new Set(response.missing.batch_ids);

  return {
    trackers: trackers.map((tracker) => {
      if (tracker.hostId !== hostId) return tracker;
      if (response.instance_id !== tracker.expectedInstanceId) {
        return requireReconciliation(tracker, "instance_mismatch");
      }
      const batch =
        (tracker.serverBatchId
          ? byServer.get(tracker.serverBatchId)
          : undefined) ?? byClient.get(tracker.clientBatchId);
      if (batch) {
        return reduceGenerationLifecycle(tracker, {
          type: "batch_snapshot",
          batch,
        });
      }
      const explicitlyMissing =
        missingClients.has(tracker.clientBatchId) ||
        (tracker.serverBatchId !== null &&
          missingServers.has(tracker.serverBatchId));
      if (explicitlyMissing) {
        return reduceGenerationLifecycle(tracker, {
          type: "lookup_missing",
          batchId: tracker.serverBatchId,
        });
      }
      return requireReconciliation(tracker, "incomplete_response");
    }),
    missingClientBatchIds: [...response.missing.client_batch_ids],
    missingBatchIds: [...response.missing.batch_ids],
  };
}

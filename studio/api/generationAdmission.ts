import { ApiError, apiJsonTo, type ApiTarget } from "./client";

export type GenerationLifecyclePhase =
  | "accepted"
  | "queued"
  | "paused"
  | "cancelling"
  | "running"
  | "held"
  | "complete"
  | "failed"
  | "cancelled";

export interface GenerationBatchResult {
  filename?: string;
  original_filename?: string;
}

export interface GenerationBatchChild {
  index: number;
  job_id: string;
  state: GenerationLifecyclePhase;
  /** Server-owned fence for POST /api/queue/{job_id}/retry. */
  retryable?: boolean | null;
  /** Human-readable failure the lifecycle presents beside `terminal_error`. */
  error?: string | null;
  /**
   * Typed cause of a `held` child — the machine's own refusal code
   * (`MODEL_NOT_FOUND`, `UNKNOWN_MODEL`, …) beside its sentence. This, never
   * the prose, is what the missing-model pull offer classifies on.
   */
  error_code?: string | null;
  created_at_ms: number;
  updated_at_ms: number;
  /**
   * Monotonic per-child version, incremented by every authoritative state
   * transition. This is the ordering authority reducers compare;
   * `updated_at_ms` collides within a millisecond, and the retry route moves
   * a child backward through the forward-phase order, so a collision there
   * decides whether the retry is visible at all.
   *
   * Absent on hosts predating the column, and `0` also means "not yet
   * transitioned since the migration" — treat both as no revision authority
   * and fall back to the timestamp.
   */
  revision?: number | null;
  completed_at_ms?: number | null;
  /** Structured terminal error. Its schema is intentionally server-owned. */
  terminal_error?: unknown;
  result?: GenerationBatchResult | null;
}

export interface GenerationBatchStatus {
  id: string;
  client_batch_id: string;
  instance_id: string;
  durable: true;
  children: GenerationBatchChild[];
}

export interface GenerationBatchAdmissionRequest<TRequest = unknown> {
  client_batch_id: string;
  requests: TRequest[];
}

export interface GenerationBatchStatusRequest {
  client_batch_ids: string[];
  batch_ids?: string[];
}

export interface GenerationBatchStatusResponse {
  instance_id: string;
  batches: GenerationBatchStatus[];
  missing: {
    client_batch_ids: string[];
    batch_ids: string[];
  };
}

export type GenerationBatchLookup =
  { kind: "found"; batch: GenerationBatchStatus } | { kind: "missing" };

/**
 * Typed `503` refusals the host answers BEFORE anything is committed. A bare
 * 5xx is ambiguous (the POST may have landed), but these name the exact
 * pre-commit gate that refused — a host that cannot admit durably, a degraded
 * encrypted-media store, a full queue, or a restart in progress — so the
 * print is definitely not queued and the message (often naming the fix) must
 * reach the user rather than a silent "confirming admission" forever.
 */
const PRE_COMMIT_REFUSAL_CODES: ReadonlySet<string> = new Set([
  "DURABLE_ADMISSION_UNAVAILABLE",
  "DURABLE_MEDIA_UNAVAILABLE",
  "QUEUE_FULL",
  "SERVER_RESTARTING",
]);

function apiErrorCode(error: ApiError): string | null {
  const body = error.body;
  if (typeof body !== "object" || body === null) return null;
  const code = (body as { code?: unknown }).code;
  return typeof code === "string" ? code : null;
}

/** A response that proves the server rejected the operation before commit.
 * Timeout-style client responses, throttling, untyped server errors,
 * transport loss, and response-decode failures are all ambiguous and must be
 * reconciled by the caller's durable identity. */
export function isDefiniteGenerationAdmissionRejection(
  error: unknown,
): boolean {
  if (!(error instanceof ApiError)) return false;
  if (error.status === 503) {
    const code = apiErrorCode(error);
    return code !== null && PRE_COMMIT_REFUSAL_CODES.has(code);
  }
  return (
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 425 &&
    error.status !== 429
  );
}

export interface DurableGenerationQueueCapabilities {
  /** The batch chunk limit. Its presence IS the durable-queue contract. */
  heterogeneous_batch_max_outputs?: number | null;
}

/**
 * Exact additive wire shape of `mold_core::DurableMediaCapabilities`. Each
 * boolean is a capability the request traits are fenced against;
 * `protocol_version` is carried for parity with the server struct and is
 * deliberately not gated on — client and server ship together.
 */
export interface DurableMediaCapabilities {
  protocol_version: number;
  encrypted_at_rest: boolean;
  generate_request_media: boolean;
  identity: boolean;
  private_h3: boolean;
}

/**
 * The advertised batch chunk limit, or `null` when this machine does not
 * advertise the durable generation queue at all. Network responses are
 * untrusted even when their TypeScript projection is typed, so a malformed
 * or absent limit never enables durable admission.
 */
export function canonicalGenerationBatchLimit(
  queue: DurableGenerationQueueCapabilities | null | undefined,
): number | null {
  const limit = queue?.heterogeneous_batch_max_outputs;
  return Number.isSafeInteger(limit) && (limit as number) >= 1
    ? (limit as number)
    : null;
}

export function chunkGenerationBatchRequests<T>(
  requests: readonly T[],
  limit: number,
): T[][] {
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new Error("generation batch limit must be a positive integer");
  }
  const chunks: T[][] = [];
  for (let offset = 0; offset < requests.length; offset += limit) {
    chunks.push(requests.slice(offset, offset + limit));
  }
  return chunks;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isLifecyclePhase(value: unknown): value is GenerationLifecyclePhase {
  return (
    value === "accepted" ||
    value === "queued" ||
    value === "paused" ||
    value === "cancelling" ||
    value === "running" ||
    value === "held" ||
    value === "complete" ||
    value === "failed" ||
    value === "cancelled"
  );
}

function parseResult(
  value: unknown,
  path: string,
): GenerationBatchResult | null {
  if (value == null) return null;
  if (!isRecord(value)) throw new Error(`${path}.result is incompatible`);
  if (value.filename !== undefined && !nonEmptyString(value.filename)) {
    throw new Error(`${path}.result.filename is incompatible`);
  }
  if (
    value.original_filename !== undefined &&
    !nonEmptyString(value.original_filename)
  ) {
    throw new Error(`${path}.result.original_filename is incompatible`);
  }
  return {
    ...(value.filename === undefined ? {} : { filename: value.filename }),
    ...(value.original_filename === undefined
      ? {}
      : { original_filename: value.original_filename }),
  };
}

export function parseGenerationBatchStatus(
  value: unknown,
  path = "generation batch",
): GenerationBatchStatus {
  if (
    !isRecord(value) ||
    !nonEmptyString(value.id) ||
    !nonEmptyString(value.client_batch_id) ||
    !nonEmptyString(value.instance_id) ||
    value.durable !== true ||
    !Array.isArray(value.children)
  ) {
    throw new Error(`${path} is incompatible`);
  }

  const seenIndexes = new Set<number>();
  const seenJobIds = new Set<string>();
  const children = value.children.map((raw, offset) => {
    const childPath = `${path}.children[${offset}]`;
    if (
      !isRecord(raw) ||
      !Number.isInteger(raw.index) ||
      (raw.index as number) < 1 ||
      !nonEmptyString(raw.job_id) ||
      !isLifecyclePhase(raw.state) ||
      !finiteNumber(raw.created_at_ms) ||
      !finiteNumber(raw.updated_at_ms) ||
      (raw.completed_at_ms !== undefined &&
        raw.completed_at_ms !== null &&
        !finiteNumber(raw.completed_at_ms)) ||
      (raw.error !== undefined &&
        raw.error !== null &&
        typeof raw.error !== "string") ||
      (raw.retryable !== undefined &&
        raw.retryable !== null &&
        typeof raw.retryable !== "boolean") ||
      (raw.revision !== undefined &&
        raw.revision !== null &&
        (!finiteNumber(raw.revision) || (raw.revision as number) < 0))
    ) {
      throw new Error(`${childPath} is incompatible`);
    }
    if (seenIndexes.has(raw.index as number) || seenJobIds.has(raw.job_id)) {
      throw new Error(`${childPath} duplicates a child identity`);
    }
    seenIndexes.add(raw.index as number);
    seenJobIds.add(raw.job_id);
    return {
      index: raw.index as number,
      job_id: raw.job_id,
      state: raw.state,
      ...(raw.retryable === undefined
        ? {}
        : { retryable: raw.retryable as boolean | null }),
      created_at_ms: raw.created_at_ms,
      updated_at_ms: raw.updated_at_ms,
      ...(raw.revision === undefined
        ? {}
        : { revision: raw.revision as number | null }),
      ...(raw.completed_at_ms === undefined
        ? {}
        : { completed_at_ms: raw.completed_at_ms as number | null }),
      ...(raw.error === undefined ? {} : { error: raw.error as string | null }),
      ...(typeof raw.error_code === "string"
        ? { error_code: raw.error_code }
        : {}),
      ...(raw.terminal_error === undefined
        ? {}
        : { terminal_error: raw.terminal_error }),
      ...(raw.result === undefined
        ? {}
        : { result: parseResult(raw.result, childPath) }),
    } satisfies GenerationBatchChild;
  });

  return {
    id: value.id,
    client_batch_id: value.client_batch_id,
    instance_id: value.instance_id,
    durable: true,
    children,
  };
}

export function parseGenerationBatchStatusResponse(
  value: unknown,
): GenerationBatchStatusResponse {
  if (
    !isRecord(value) ||
    !nonEmptyString(value.instance_id) ||
    !Array.isArray(value.batches) ||
    !isRecord(value.missing) ||
    !Array.isArray(value.missing.client_batch_ids) ||
    !Array.isArray(value.missing.batch_ids) ||
    !value.missing.client_batch_ids.every(nonEmptyString) ||
    !value.missing.batch_ids.every(nonEmptyString)
  ) {
    throw new Error("generation batch status response is incompatible");
  }
  const batches = value.batches.map((batch, index) =>
    parseGenerationBatchStatus(batch, `generation batches[${index}]`),
  );
  const seen = new Set<string>();
  const seenClients = new Set<string>();
  for (const batch of batches) {
    if (batch.instance_id !== value.instance_id) {
      throw new Error(
        "generation batch status response mixes server instances",
      );
    }
    if (seen.has(batch.id)) {
      throw new Error("generation batch status response duplicates a batch");
    }
    if (seenClients.has(batch.client_batch_id)) {
      throw new Error(
        "generation batch status response duplicates a client batch",
      );
    }
    seen.add(batch.id);
    seenClients.add(batch.client_batch_id);
  }
  return {
    instance_id: value.instance_id,
    batches,
    missing: {
      client_batch_ids: [...value.missing.client_batch_ids],
      batch_ids: [...value.missing.batch_ids],
    },
  };
}

export async function admitGenerationBatch<TRequest>(
  target: ApiTarget,
  request: GenerationBatchAdmissionRequest<TRequest>,
  signal?: AbortSignal,
  extraHeaders?: HeadersInit,
): Promise<GenerationBatchStatus> {
  return parseGenerationBatchStatus(
    await apiJsonTo<unknown>(target, "/api/generation-batches", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...Object.fromEntries(new Headers(extraHeaders)) },
      body: JSON.stringify(request),
      ...(signal ? { signal } : {}),
    }),
  );
}

/** Authoritative idempotency lookup after a POST response was lost. */
export async function lookupGenerationBatchByClientId(
  target: ApiTarget,
  clientBatchId: string,
  signal?: AbortSignal,
): Promise<GenerationBatchLookup> {
  try {
    return {
      kind: "found",
      batch: parseGenerationBatchStatus(
        await apiJsonTo<unknown>(
          target,
          `/api/generation-batches/by-client/${encodeURIComponent(clientBatchId)}`,
          signal ? { signal } : {},
        ),
      ),
    };
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return { kind: "missing" };
    }
    throw error;
  }
}

export async function getGenerationBatch(
  target: ApiTarget,
  batchId: string,
  signal?: AbortSignal,
): Promise<GenerationBatchLookup> {
  try {
    return {
      kind: "found",
      batch: parseGenerationBatchStatus(
        await apiJsonTo<unknown>(
          target,
          `/api/generation-batches/${encodeURIComponent(batchId)}`,
          signal ? { signal } : {},
        ),
      ),
    };
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return { kind: "missing" };
    }
    throw error;
  }
}

/** One bounded HTTP exchange chosen by the caller's actual tracked set. */
export async function reconcileGenerationBatches(
  target: ApiTarget,
  request: GenerationBatchStatusRequest,
  signal?: AbortSignal,
): Promise<GenerationBatchStatusResponse> {
  return parseGenerationBatchStatusResponse(
    await apiJsonTo<unknown>(target, "/api/generation-batches/status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
      ...(signal ? { signal } : {}),
    }),
  );
}

import { reactive } from "vue";
import { defineStore } from "pinia";
import { apiFetchTo, apiJsonTo, currentTarget, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import {
  evictMedia,
  fetchGalleryMediaBytes,
  galleryMediaPath,
  streamableMediaUrl,
} from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { notifyGenerated, notifyGenerationFailed } from "../lib/notify";
import { describeTransportError, isTransportFailure } from "../lib/api/errors";
import {
  isInterruptedGenerationError,
  reconcileInterruptedGenerationJobs,
} from "../lib/generationRecovery";
import { useAppPrefsStore } from "./appPrefs";
import { useGalleryStore } from "./gallery";
import { useHostsStore } from "./hosts";
import type {
  ChainProgressEvent,
  CompleteEvent,
  GenerateRequest,
  PromptTransformProvenance,
  ProgressEvent,
  SseChainCompleteEvent,
} from "../lib/api/types";
import {
  buildAutoChainRequest,
  type AutoChainRoutingDecision,
  type ChainRoutingDecision,
} from "../lib/chainRouting";
import {
  applyChainProgress,
  applyCompletionWarnings,
  applyProgress,
  base64ToBlobUrl,
  chainCompleteToComplete,
  isCancelledError,
  markJobSettled,
  metadataOnlyResult,
  newJob,
  type Job,
} from "../lib/generationJob";
import {
  prepareReferenceUploads,
  requestNeedsReferenceUpload,
  type ReferenceUploadCapabilities,
  type ReferenceUploadLease,
} from "@studio/api/referenceUploads";
import { requestWarningsFromHeaders } from "@studio/lib/requestWarnings";
import { blobToBase64 } from "@studio/lib/base64";
import {
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
  type GenerationBatchStatus,
} from "@studio/api/generationAdmission";
import {
  buildGenerationBatchStatusRequest,
  createGenerationBatchTracker,
  isTerminalGenerationPhase,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
  type GenerationLifecycleJob,
} from "@studio/lib/generationLifecycle";
import {
  durableChildSummary,
  loadDurableGenerationRecovery,
  parseEventAuthority,
  parseEventResync,
  requestIsEligibleForDurableGeneration,
  saveDurableGenerationRecovery,
  type DurableGenerationRecoveryRecord,
} from "../lib/durableGeneration";
import { TargetStreamSlots } from "@studio/lib/targetStreamSlots";

export {
  applyChainProgress,
  applyCompletionWarnings,
  applyProgress,
  base64ToBlobUrl,
  chainCompleteToComplete,
  isCancelledError,
  jobPhase,
  jobProgress,
  jobStatusCode,
  metadataOnlyResult,
  newJob,
  type Job,
  type JobStatus,
} from "../lib/generationJob";

/** Settled prints retained for the activity rail's scrollable history. */
export const GENERATION_HISTORY_LIMIT = 50;
/** Only the freshest jobs retain decoded/encoded media in Create memory. */
export const GENERATION_RICH_HISTORY_LIMIT = 12;

/** Where a batch runs — mirrors `HostRoute` from the hosts store. */
export interface JobRoute {
  hostId: string;
  label: string;
  kind: "local" | "remote";
  target: ApiTarget;
  /** iPhone is remote-only and has no desktop filesystem gallery to mirror into. */
  mirrorRemoteOutput?: boolean;
  /** iPhone releases large base64 image/video payloads after decoding them. */
  retainEncodedResult?: boolean;
  /** iPhone asks the host for saved-file metadata instead of encoded media bytes. */
  metadataOnlyCompletion?: boolean;
  /** Exact server installation and upload protocol captured at submit time. */
  instanceId?: string | null;
  referenceUploads?: ReferenceUploadCapabilities | null;
  heterogeneousBatch?: boolean;
  heterogeneousBatchMaxOutputs?: number | null;
  durableBatchOutcomes?: boolean;
}

interface ReferenceUploadAuthority {
  target: ApiTarget;
  instanceId: string;
  capabilities: ReferenceUploadCapabilities | null;
}

/** Filesystem-safe local filename for a saved output. */
export function suggestOutputFilename(
  model: string,
  seed: number,
  format: string,
  nowMs: number = Date.now(),
  role?: "original" | "upscaled",
): string {
  const slug = model
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `mold-${slug}-${seed}-${nowMs}${role ? `-${role}` : ""}.${format}`;
}

/** The primary connection as a target, or `null` when nothing is connected —
 *  `currentTarget()` throws in that case, and a reconcile with no host to ask
 *  must simply not happen rather than reject the batch. */
function connectedTarget(): ApiTarget | null {
  try {
    return currentTarget();
  } catch {
    return null;
  }
}

/** Random 32-bit seed — small enough to stay an exact integer after `+ i`. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 0xffffffff);
}

/**
 * Whether a batch must resolve an explicit host route instead of relying on
 * the primary connection: multiple live hosts, or a primary that isn't ready
 * while some host is (local engine down, remote still serving). When nothing
 * is ready, stay unrouted so the submit surfaces the directed error. Pure —
 * shared by GenerateView's submit path and its estimate preflight.
 */
export function needsHostRoute(opts: {
  multiHost: boolean;
  primaryReady: boolean;
  anyHostReady: boolean;
}): boolean {
  return opts.multiHost || (!opts.primaryReady && opts.anyHostReady);
}

/**
 * Resolve the base seed for a batch: an explicit finite seed is honored,
 * otherwise a fresh random base is drawn so the run is reproducible from the
 * first sibling. Pure given `rng`.
 */
export function resolveBaseSeed(seed: number | undefined, rng: () => number = randomSeed): number {
  return seed !== undefined && Number.isFinite(seed) ? seed : rng();
}

export interface BatchRequestOptions {
  /** Ordered prompt override for each sibling. Must exactly match the normalized batch size. */
  prompts?: readonly string[];
  /** Shared source prompt retained as provenance on every sibling request. */
  originalPrompt?: string;
  /** Per-sibling prompt-transform provenance in the same order as prompts. */
  promptTransforms?: readonly import("../lib/api/types").PromptTransformProvenance[];
  /** Durable identity shared by prepared siblings and retained in gallery metadata. */
  batchId?: string;
  promptTransform?: PromptTransformProvenance;
}

/**
 * Expand one request into `batchSize` sibling requests with seeds
 * `baseSeed + i`, each forced to `batch_size: 1` (the client drives the
 * sequence, one job per server call). Optional per-item prompts preserve their
 * order and one shared source prompt. Pure — the seed and prompt decisions
 * live here so they can be tested without the store or the network.
 */
export function planBatchRequests(
  req: GenerateRequest,
  batchSize: number,
  baseSeed: number,
  options: BatchRequestOptions = {},
): GenerateRequest[] {
  const size = Math.max(1, Math.floor(batchSize));
  if (options.prompts !== undefined && options.prompts.length !== size) {
    throw new RangeError(
      `Per-item prompt count ${options.prompts.length} does not match batch size ${size}`,
    );
  }
  if (options.promptTransforms !== undefined && options.promptTransforms.length !== size) {
    throw new RangeError(
      `Per-item prompt transform count ${options.promptTransforms.length} does not match batch size ${size}`,
    );
  }
  return Array.from({ length: size }, (_, i) => ({
    ...req,
    ...(options.prompts !== undefined ? { prompt: options.prompts[i]! } : {}),
    ...(options.originalPrompt !== undefined ? { original_prompt: options.originalPrompt } : {}),
    ...(options.promptTransform !== undefined
      ? {
          prompt_transform: {
            ...options.promptTransform,
            dimensions: [...(options.promptTransform.dimensions ?? [])],
          },
        }
      : {}),
    ...(options.promptTransforms !== undefined
      ? { prompt_transform: options.promptTransforms[i] }
      : {}),
    ...(options.batchId !== undefined
      ? { batch_id: options.batchId, batch_index: i + 1, batch_count: size }
      : {}),
    seed: baseSeed + i,
    batch_size: 1,
  }));
}

/**
 * Display order for the jobs rail: actively developing first, then the
 * server's real queue order. Concurrent batch submissions race to the
 * server, so submission order and queue position can disagree — the rail
 * must show the order the engine will actually run.
 */
export function railOrder(jobs: Job[]): Job[] {
  const rank = (j: Job): number =>
    j.status === "denoising" || j.status === "finishing" ? 0 : j.status === "loading" ? 1 : 2;
  return [...jobs].sort((a, b) => {
    const r = rank(a) - rank(b);
    if (r !== 0) return r;
    const pa = a.queuePosition ?? Number.MAX_SAFE_INTEGER;
    const pb = b.queuePosition ?? Number.MAX_SAFE_INTEGER;
    if (pa !== pb) return pa - pb;
    return a.clientId - b.clientId;
  });
}

/**
 * Run `tasks` with at most `limit` in flight at once, resolving with each
 * task's result in task order. A rejected task is swallowed (its slot resolves
 * to `undefined`) so one failure never stalls the pool — batch siblings
 * surface failures through their own job status, not this promise.
 * Pure/exported for tests.
 */
export async function runWithConcurrency<T>(
  tasks: Array<() => Promise<T>>,
  limit = 2,
): Promise<Array<T | undefined>> {
  const results = new Array<T | undefined>(tasks.length);
  let cursor = 0;
  const runner = async (): Promise<void> => {
    while (cursor < tasks.length) {
      const index = cursor++;
      try {
        results[index] = await tasks[index]!();
      } catch {
        /* swallow — a failed task must not stall its siblings */
      }
    }
  };
  const width = Math.min(Math.max(1, limit), tasks.length);
  await Promise.all(Array.from({ length: width }, runner));
  return results;
}

const MIME: Record<string, string> = {
  png: "image/png",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
  apng: "image/apng",
  mp4: "video/mp4",
};

/**
 * Held generation SSE requests share the browser's per-origin HTTP pool with
 * gallery reads and queue cancellation. Four slots match the web surface and
 * let a four-worker host stay busy while retaining connection headroom. The
 * limit is global per host, not merely per batch.
 */
const MAX_STREAMS_PER_TARGET = 4;

const streamSlots = new TargetStreamSlots(MAX_STREAMS_PER_TARGET);

interface DurableSettlement {
  promise: Promise<Job[]>;
  resolve: (jobs: Job[]) => void;
}

interface DurableHostStream {
  abort: AbortController;
  target: ApiTarget;
}

/** Durable recovery authority is intentionally outside Pinia: it contains no
 * render state and must never make API keys serializable through devtools. */
const durableRecords = new Map<string, DurableGenerationRecoveryRecord>();
const durableJobIds = new Map<string, Map<number, number>>();
const durableSettlements = new Map<string, DurableSettlement>();
const durableHostStreams = new Map<string, DurableHostStream>();
const durableReconciles = new Map<string, Promise<void>>();
const sharedDurableEventHosts = new Set<string>();
let durableRecoveryLoaded = false;

function persistDurableRecords(): void {
  saveDurableGenerationRecovery(durableRecords.values());
}

function durableClientBatchId(): string {
  return crypto.randomUUID();
}

function createDurableSettlement(): DurableSettlement {
  let resolve!: (jobs: Job[]) => void;
  const promise = new Promise<Job[]>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function recordIsTerminal(record: DurableGenerationRecoveryRecord): boolean {
  const jobs = Object.values(record.tracker.jobs);
  return (
    jobs.length === record.children.length &&
    jobs.every((job) => isTerminalGenerationPhase(job.phase))
  );
}

function requestFormat(
  filename: string | undefined,
  fallback: CompleteEvent["format"],
): CompleteEvent["format"] {
  const extension = filename?.split(".").pop()?.toLowerCase();
  return extension === "jpg"
    ? "jpeg"
    : extension === "jpeg" ||
        extension === "png" ||
        extension === "webp" ||
        extension === "gif" ||
        extension === "apng" ||
        extension === "mp4" ||
        extension === "wav"
      ? extension
      : fallback;
}

function jobHasSettled(job: Job): boolean {
  return job.status === "complete" || job.status === "error";
}

/** Move a job to a terminal status and stamp when it got there. Every
 *  terminal transition goes through here so the Create strip's attention-row
 *  age rule (and future history ordering) can trust `settledAtMs`. */
function settleJob(job: Job, status: "complete" | "error"): void {
  job.status = status;
  markJobSettled(job);
}

function resultUrlExpiry(url: string): number | null {
  try {
    const expires = new URL(url).searchParams.get("expires");
    if (!expires) return null;
    const seconds = Number(expires);
    return Number.isSafeInteger(seconds) ? seconds * 1000 : null;
  } catch {
    return null;
  }
}

export const useGenerationStore = defineStore("generation", {
  state: () => ({
    /**
     * Every job of this session, submission order. The server queue is the
     * scheduler for submitted jobs. A small per-host stream pool keeps later
     * jobs local until a connection slot opens, so queue/cancel/gallery HTTP
     * requests stay responsive. Every job snapshots its model + params.
     */
    jobs: [] as Job[],
    nextClientId: 1,
    nextBatchId: 1,
    /** Explicit canvas/job selection; null follows the automatic active job. */
    selectedClientId: null as number | null,
    /** Batches whose settled consumers have not yet had a microtask turn. */
    pendingConsumerBatchIds: [] as number[],
  }),
  getters: {
    /**
     * The job the Generate canvas tracks: the most recent actively
     * developing job, else the most recent queued one, else the most
     * recent job overall.
     */
    active(state): Job | null {
      const jobs = state.jobs;
      if (state.selectedClientId !== null) {
        const selected = jobs.find((job) => job.clientId === state.selectedClientId);
        if (selected) return selected;
      }
      const latest = (pred: (j: Job) => boolean) => {
        for (let i = jobs.length - 1; i >= 0; i--) {
          if (pred(jobs[i]!)) return jobs[i]!;
        }
        return null;
      };
      return (
        latest(
          (j) => j.status === "denoising" || j.status === "finishing" || j.status === "loading",
        ) ??
        latest((j) => j.status === "queued") ??
        (jobs.length > 0 ? jobs[jobs.length - 1]! : null)
      );
    },
    /** The active job's batch — drives the sibling dots under the canvas. */
    siblings(): Job[] {
      const active = this.active;
      if (!active) return [];
      return this.jobs.filter((j) => j.batchId === active.batchId);
    },
    /** Jobs still queued or developing, submission order. */
    pending(state): Job[] {
      return state.jobs.filter((j) => j.status !== "complete" && j.status !== "error");
    },
    /** Seed of the most recent finished print — powers "lock last seed". */
    lastSeedUsed(state): number | null {
      for (let i = state.jobs.length - 1; i >= 0; i--) {
        const result = state.jobs[i]!.result;
        if (result) return result.seed_used;
      }
      return null;
    },
  },
  actions: {
    /** Restore secret/media-free durable authority after a desktop reload. */
    resumeDurableGenerations(): void {
      if (!durableRecoveryLoaded) {
        durableRecoveryLoaded = true;
        for (const record of loadDurableGenerationRecovery()) {
          durableRecords.set(record.tracker.clientBatchId, record);
        }
      }
      for (const record of durableRecords.values()) {
        if (durableJobIds.has(record.tracker.clientBatchId)) continue;
        const mapping = new Map<number, number>();
        const localBatchId = this.nextBatchId++;
        const restoredTerminal = recordIsTerminal(record);
        const recoveredHost = useHostsStore().all.find(
          (candidate) =>
            candidate.id === record.tracker.hostId &&
            candidate.instanceId === record.tracker.expectedInstanceId &&
            candidate.baseUrl,
        );
        for (const summary of record.children) {
          const job = this.startJob({
            prompt: "Recovered generation",
            model: summary.model,
            width: summary.width,
            height: summary.height,
            steps: summary.steps,
            guidance: summary.guidance,
            ...(summary.seed === null ? {} : { seed: summary.seed }),
            output_format: summary.format,
          });
          job.batchId = localBatchId;
          job.hostId = record.tracker.hostId;
          job.hostLabel = record.hostLabel;
          job.remote = record.hostKind === "remote";
          job.mirrorRemoteOutput = record.mirrorRemoteOutput;
          job.streamStarted = true;
          job.suppressFreshCompletion = restoredTerminal;
          mapping.set(summary.index, job.clientId);
          summary.clientId = job.clientId;
          if (recoveredHost?.baseUrl) {
            targets.set(job.clientId, {
              baseUrl: recoveredHost.baseUrl,
              apiKey: recoveredHost.apiKey,
            });
          }
        }
        durableJobIds.set(record.tracker.clientBatchId, mapping);
        durableSettlements.set(record.tracker.clientBatchId, createDurableSettlement());
        this.applyDurableRecord(record);
        this.ensureDurableHostStream(record.tracker.hostId);
      }
      persistDurableRecords();
      void this.reconcileDurableAll();
    },
    /** The primary events store calls this before opening its shared stream. */
    attachSharedDurableEventHost(hostId: string): void {
      sharedDurableEventHosts.add(hostId);
      durableHostStreams.get(hostId)?.abort.abort();
      durableHostStreams.delete(hostId);
    },
    detachSharedDurableEventHost(hostId: string): void {
      sharedDurableEventHosts.delete(hostId);
      this.ensureDurableHostStream(hostId);
    },
    onDurableEventClose(hostId: string): void {
      let changed = false;
      for (const record of durableRecords.values()) {
        if (record.tracker.hostId !== hostId || recordIsTerminal(record)) continue;
        record.tracker = reduceGenerationLifecycle(record.tracker, {
          type: "event_gap",
          instanceId: record.tracker.expectedInstanceId,
        });
        this.applyDurableRecord(record);
        changed = true;
      }
      if (!changed) return;
      persistDurableRecords();
      void this.reconcileDurableHost(hostId);
    },
    onDurableEvent(hostId: string, event: string, data: string): void {
      const records = [...durableRecords.values()].filter(
        (record) => record.tracker.hostId === hostId && !recordIsTerminal(record),
      );
      if (records.length === 0) return;
      if (event === "authority") {
        const authority = parseEventAuthority(data);
        if (!authority) {
          for (const record of records) {
            record.tracker = reduceGenerationLifecycle(record.tracker, {
              type: "event_gap",
              instanceId: "",
            });
            this.applyDurableRecord(record);
          }
          persistDurableRecords();
          void this.reconcileDurableHost(hostId);
          return;
        }
        for (const record of records) {
          if (authority.instanceId !== record.tracker.expectedInstanceId) {
            record.tracker = reduceGenerationLifecycle(record.tracker, {
              type: "event_gap",
              instanceId: authority.instanceId,
            });
            this.applyDurableRecord(record);
          }
        }
        persistDurableRecords();
        if (records.some((record) => authority.instanceId === record.tracker.expectedInstanceId)) {
          void this.reconcileDurableHost(hostId);
        }
        return;
      }
      if (event === "resync_required") {
        const gap = parseEventResync(data);
        for (const record of records) {
          record.tracker = reduceGenerationLifecycle(record.tracker, {
            type: "event_gap",
            instanceId: gap?.instanceId ?? "",
          });
          this.applyDurableRecord(record);
        }
        persistDurableRecords();
        void this.reconcileDurableHost(hostId);
        return;
      }
      if (event !== "event") {
        void this.reconcileDurableHost(hostId);
        return;
      }
      try {
        const frame = JSON.parse(data) as { type?: unknown; id?: unknown };
        if (
          (frame.type === "job_queued" ||
            frame.type === "job_started" ||
            frame.type === "job_ended" ||
            frame.type === "gallery_added") &&
          (frame.type === "gallery_added" ||
            records.some((record) =>
              Object.values(record.tracker.jobs).some((job) => job.authority.jobId === frame.id),
            ))
        ) {
          void this.reconcileDurableHost(hostId);
        }
      } catch {
        void this.reconcileDurableHost(hostId);
      }
    },
    ensureDurableHostStream(hostId: string): void {
      if (sharedDurableEventHosts.has(hostId)) return;
      const host = useHostsStore().all.find((candidate) => candidate.id === hostId);
      const record = [...durableRecords.values()].find(
        (candidate) =>
          candidate.tracker.hostId === hostId &&
          candidate.tracker.expectedInstanceId === host?.instanceId &&
          !recordIsTerminal(candidate),
      );
      if (!record) return;
      if (!host?.baseUrl || host.status !== "ready") return;
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
      const existing = durableHostStreams.get(hostId);
      if (
        existing &&
        existing.target.baseUrl === target.baseUrl &&
        existing.target.apiKey === target.apiKey &&
        !existing.abort.signal.aborted
      ) {
        return;
      }
      existing?.abort.abort();
      const stream: DurableHostStream = {
        abort: new AbortController(),
        target,
      };
      durableHostStreams.set(hostId, stream);
      void sseStream("/api/events", {
        target,
        signal: stream.abort.signal,
        retry: true,
        terminalHttpStatuses: [401, 403, 404],
        onEvent: (event, data) => this.onDurableEvent(hostId, event, data),
        onClose: () => {
          if (stream.abort.signal.aborted) return;
          this.onDurableEventClose(hostId);
        },
      });
    },
    async reconcileDurableAll(): Promise<void> {
      const hostIds = new Set(
        [...durableRecords.values()]
          .filter((record) => !recordIsTerminal(record))
          .map((record) => record.tracker.hostId),
      );
      await Promise.all([...hostIds].map((hostId) => this.reconcileDurableHost(hostId)));
      await Promise.all(
        [...durableRecords.values()].filter(recordIsTerminal).map((record) => {
          const mapping = durableJobIds.get(record.tracker.clientBatchId);
          const jobs = [...(mapping?.values() ?? [])]
            .map((clientId) => this.jobs.find((candidate) => candidate.clientId === clientId))
            .filter((job): job is Job => !!job);
          return this.finishDurableBatchEffects(record, jobs);
        }),
      );
    },
    async reconcileDurableHost(hostId: string): Promise<void> {
      const existing = durableReconciles.get(hostId);
      if (existing) return existing;
      const operation = (async () => {
        const records = [...durableRecords.values()].filter(
          (record) => record.tracker.hostId === hostId && !recordIsTerminal(record),
        );
        if (records.length === 0) return;
        const host = useHostsStore().all.find((candidate) => candidate.id === hostId);
        if (!host?.baseUrl || host.status !== "ready") return;
        const matching = records.filter(
          (record) => host.instanceId === record.tracker.expectedInstanceId,
        );
        for (const record of records) {
          if (host.instanceId === record.tracker.expectedInstanceId) continue;
          record.tracker = reduceGenerationLifecycle(record.tracker, {
            type: "event_gap",
            instanceId: host.instanceId ?? "",
          });
          this.applyDurableRecord(record);
        }
        if (matching.length === 0) {
          persistDurableRecords();
          return;
        }
        const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
        const trackers = matching.map((record) => record.tracker);
        const request = buildGenerationBatchStatusRequest(trackers, hostId);
        if (request.client_batch_ids.length === 0 && !request.batch_ids?.length) return;
        const response = await reconcileGenerationBatches(target, request);
        const merged = mergeBulkGenerationBatchResponse(trackers, hostId, response);
        merged.trackers.forEach((tracker, index) => {
          const record = matching[index]!;
          record.tracker = tracker;
          this.applyDurableRecord(record);
        });
        persistDurableRecords();
      })();
      durableReconciles.set(hostId, operation);
      try {
        await operation;
      } catch {
        // Last-good authority remains visible; the next event/reconnect/wake retries.
      } finally {
        if (durableReconciles.get(hostId) === operation) durableReconciles.delete(hostId);
      }
    },
    applyDurableRecord(record: DurableGenerationRecoveryRecord): void {
      const mapping = durableJobIds.get(record.tracker.clientBatchId);
      if (!mapping) return;
      if (record.tracker.reconciliation.reason === "instance_mismatch") {
        for (const clientId of mapping.values()) {
          const job = this.jobs.find((candidate) => candidate.clientId === clientId);
          if (!job || jobHasSettled(job)) continue;
          job.stage = "Original machine identity changed — outcome unknown";
          job.interrupted = true;
        }
        return;
      }
      for (const lifecycle of Object.values(record.tracker.jobs)) {
        const clientId = mapping.get(lifecycle.childIndex);
        const job = this.jobs.find((candidate) => candidate.clientId === clientId);
        if (!job) continue;
        job.id = lifecycle.authority.jobId;
        job.streamStarted = true;
        switch (lifecycle.phase) {
          case "accepted":
          case "queued":
            if (!jobHasSettled(job)) {
              job.status = "queued";
              job.stage = null;
            }
            break;
          case "held":
            if (!jobHasSettled(job)) {
              job.status = "queued";
              job.stage = "Held by host — open Jobs for details";
            }
            break;
          case "running":
            if (!jobHasSettled(job)) {
              job.status = "loading";
              job.stage = "Developing";
            }
            break;
          case "complete":
          case "failed":
          case "cancelled":
            this.applyDurableTerminal(record, lifecycle, job);
            break;
        }
      }
      if (!recordIsTerminal(record)) return;
      const jobs = [...mapping.values()]
        .map((clientId) => this.jobs.find((candidate) => candidate.clientId === clientId))
        .filter((job): job is Job => !!job);
      durableSettlements.get(record.tracker.clientBatchId)?.resolve(jobs);
      durableSettlements.delete(record.tracker.clientBatchId);
      void this.finishDurableBatchEffects(record, jobs);
    },
    applyDurableTerminal(
      record: DurableGenerationRecoveryRecord,
      lifecycle: GenerationLifecycleJob,
      job: Job,
    ): void {
      if (jobHasSettled(job)) return;
      if (lifecycle.phase === "complete" && lifecycle.result?.filename) {
        const summary = record.children.find((child) => child.index === lifecycle.childIndex)!;
        const format = requestFormat(lifecycle.result.filename, summary.format);
        job.result = {
          image: "",
          format,
          width: summary.width,
          height: summary.height,
          seed_used: summary.seed ?? 0,
          generation_time_ms: 0,
          model: summary.model,
          filename: lifecycle.result.filename,
          ...(lifecycle.result.originalFilename
            ? { original_filename: lifecycle.result.originalFilename }
            : {}),
          metadata: null,
        };
        job.visualSeed = String(summary.seed ?? 0);
        settleJob(job, "complete");
        void this.refreshRemoteResultUrl(job.clientId).catch(() => undefined);
      } else {
        settleJob(job, "error");
        job.error =
          lifecycle.phase === "cancelled"
            ? "Cancelled"
            : (lifecycle.error ?? "Generation failed on the host.");
      }
      job.cancelling = false;
      if (job.previewUrl) {
        URL.revokeObjectURL(job.previewUrl);
        job.previewUrl = null;
      }
    },
    async finishDurableBatchEffects(
      record: DurableGenerationRecoveryRecord,
      jobs: Job[],
    ): Promise<void> {
      const claim = (key: string): boolean => {
        if (record.effectReceipts.includes(key)) return false;
        record.effectReceipts.push(key);
        persistDurableRecords();
        return true;
      };
      const completed = jobs.filter((job) => job.status === "complete" && job.result?.filename);
      if (completed.length > 0 && claim("native-notification")) {
        notifyGenerated(completed[0]!.prompt, completed[0]!.result?.filename);
      } else {
        const failed = jobs.find(
          (job) => job.status === "error" && job.error && !isCancelledError(job.error),
        );
        if (failed && claim("native-failure")) {
          notifyGenerationFailed(describeTransportError(failed.error!, failed.hostLabel));
        }
      }
      const host = useHostsStore().all.find(
        (candidate) =>
          candidate.id === record.tracker.hostId &&
          candidate.instanceId === record.tracker.expectedInstanceId,
      );
      const saveRemoteOutputs = useAppPrefsStore().settings?.saveRemoteOutputs ?? true;
      const hasPendingMirror = completed.some(
        (job) =>
          job.remote &&
          job.mirrorRemoteOutput &&
          saveRemoteOutputs &&
          [job.result?.original_filename, job.result?.filename]
            .filter((value): value is string => !!value)
            .some((filename) => !record.effectReceipts.includes(`mirror:${filename}`)),
      );
      // A terminal filename is still bound to the frozen server instance. If
      // that authority is temporarily absent, retain the record until the
      // exact host returns; never mirror from a replacement at the same URL.
      if (hasPendingMirror && !host?.baseUrl) return;
      if (host?.baseUrl) {
        const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
        for (const job of completed) {
          const result = job.result!;
          for (const filename of [result.original_filename, result.filename].filter(
            (value): value is string => !!value,
          )) {
            const effect = `mirror:${filename}`;
            if (!job.remote || !job.mirrorRemoteOutput || !saveRemoteOutputs || !claim(effect)) {
              continue;
            }
            try {
              const bytes = await fetchGalleryMediaBytes(
                galleryMediaPath(filename, "host"),
                target,
              );
              const buffer = Uint8Array.from(bytes).buffer;
              await ipc.saveOutputBytes(filename, await blobToBase64(new Blob([buffer])), null);
              void useGalleryStore().refreshHost("local");
            } catch (error) {
              console.warn("local save of remote durable output failed:", error);
            }
          }
        }
      }
      if (completed.length > 0) void useGalleryStore().refreshHost(record.tracker.hostId);
      durableRecords.delete(record.tracker.clientBatchId);
      durableJobIds.delete(record.tracker.clientBatchId);
      persistDurableRecords();
      if (
        ![...durableRecords.values()].some((item) => item.tracker.hostId === record.tracker.hostId)
      ) {
        durableHostStreams.get(record.tracker.hostId)?.abort.abort();
        durableHostStreams.delete(record.tracker.hostId);
      }
    },
    async admitDurableRecord(
      record: DurableGenerationRecoveryRecord,
      requests: GenerateRequest[],
    ): Promise<void> {
      const host = useHostsStore().all.find(
        (candidate) =>
          candidate.id === record.tracker.hostId &&
          candidate.instanceId === record.tracker.expectedInstanceId,
      );
      const target = host?.baseUrl
        ? { baseUrl: host.baseUrl, apiKey: host.apiKey }
        : (() => {
            const mapping = durableJobIds.get(record.tracker.clientBatchId);
            const firstClientId = mapping?.values().next().value as number | undefined;
            return firstClientId === undefined ? null : (targets.get(firstClientId) ?? null);
          })();
      if (!target) return;
      const body = { client_batch_id: record.tracker.clientBatchId, requests };
      const attach = (batch: GenerationBatchStatus) => {
        record.tracker = reduceGenerationLifecycle(record.tracker, {
          type: "batch_snapshot",
          batch,
        });
        this.applyDurableRecord(record);
        persistDurableRecords();
        this.ensureDurableHostStream(record.tracker.hostId);
      };
      try {
        attach(await admitGenerationBatch(target, body));
        return;
      } catch (error) {
        if (!isTransportFailure(error)) {
          record.tracker = reduceGenerationLifecycle(record.tracker, {
            type: "admission_rejected",
            error: error instanceof Error ? error.message : String(error),
          });
          for (const clientId of durableJobIds.get(record.tracker.clientBatchId)?.values() ?? []) {
            const job = this.jobs.find((candidate) => candidate.clientId === clientId);
            if (!job || jobHasSettled(job)) continue;
            settleJob(job, "error");
            job.error = record.tracker.admission.error;
          }
          persistDurableRecords();
          const jobs = [...(durableJobIds.get(record.tracker.clientBatchId)?.values() ?? [])]
            .map((clientId) => this.jobs.find((candidate) => candidate.clientId === clientId))
            .filter((job): job is Job => !!job);
          durableSettlements.get(record.tracker.clientBatchId)?.resolve(jobs);
          durableSettlements.delete(record.tracker.clientBatchId);
          void this.finishDurableBatchEffects(record, jobs);
          return;
        }
        record.tracker = reduceGenerationLifecycle(record.tracker, {
          type: "admission_uncertain",
          error: error instanceof Error ? error.message : String(error),
        });
        persistDurableRecords();
      }

      // A lost POST response is never permission to use the legacy endpoint.
      // Recover by idempotency key, then (only when the server explicitly says
      // it is missing) repeat the same durable admission.
      try {
        const lookup = await lookupGenerationBatchByClientId(target, record.tracker.clientBatchId);
        if (lookup.kind === "found") {
          attach(lookup.batch);
          return;
        }
        attach(await admitGenerationBatch(target, body));
      } catch {
        // Authority remains uncertain and persisted. Reconnect/wake performs
        // the bulk by-client reconciliation; no duplicate endpoint is used.
        this.ensureDurableHostStream(record.tracker.hostId);
      }
    },
    select(clientId: number | null) {
      this.selectedClientId =
        clientId !== null && this.jobs.some((job) => job.clientId === clientId) ? clientId : null;
    },
    /**
     * Submit a batch: every sibling is created with seeds `base + i`, but at
     * most two hold an SSE stream open at once. A browser's per-host HTTP/1.1
     * budget is ~6 connections; uncapped, a large batch would exhaust it and
     * starve the gallery/download requests behind the held-open streams. Later
     * siblings simply wait their turn in the pool. Returns the created jobs
     * plus a promise resolving when every sibling settles.
     */
    submitBatch(
      req: GenerateRequest,
      batchSize: number,
      route: JobRoute | null = null,
      chainRouting: ChainRoutingDecision | null = null,
      requestOptions: BatchRequestOptions = {},
    ): { jobs: Job[]; settled: Promise<Job[]> } {
      this.selectedClientId = null;
      if (chainRouting?.kind === "reject") throw new Error(chainRouting.reason);
      const size = Math.max(1, Math.floor(batchSize));
      const baseSeed = resolveBaseSeed(req.seed);
      const plans = planBatchRequests(req, size, baseSeed, requestOptions);
      const batchId = this.nextBatchId++;
      this.pendingConsumerBatchIds.push(batchId);
      const jobs = plans.map((plan) => {
        const job = this.startJob(plan);
        job.batchId = batchId;
        if (route) {
          job.hostId = route.hostId;
          job.hostLabel = route.label;
          job.remote = route.kind === "remote";
          job.mirrorRemoteOutput = route.mirrorRemoteOutput ?? true;
          job.retainEncodedResult = route.retainEncodedResult ?? true;
          job.metadataOnlyCompletion = route.metadataOnlyCompletion ?? false;
          targets.set(job.clientId, route.target);
          if (requestNeedsReferenceUpload(plan)) {
            referenceUploadAuthorities.set(job.clientId, {
              target: { ...route.target },
              instanceId: route.instanceId ?? "",
              capabilities: route.referenceUploads ?? null,
            });
          }
        } else {
          // Unrouted = the local primary engine — its prints are already in
          // this device's gallery, so they never trigger the remote auto-save.
          job.remote = false;
          // When the primary isn't ready but another host is (local engine
          // failed to start, remote still serving), snapshot that host
          // instead of the dead primary so the batch isn't dead on arrival.
          const hosts = useHostsStore();
          const primaryReady = hosts.primaryHost?.status === "ready";
          const fallback = primaryReady
            ? undefined
            : hosts.all.find((h) => h.status === "ready" && h.baseUrl);
          if (fallback?.baseUrl) {
            job.hostId = fallback.id;
            job.hostLabel = fallback.label;
            job.remote = fallback.kind === "remote";
            targets.set(job.clientId, { baseUrl: fallback.baseUrl, apiKey: fallback.apiKey });
          } else {
            // And snapshot the PRIMARY target at submit time: queued batch
            // siblings open their streams later, and cancels resolve later
            // still — both must hit the host the job was submitted to, not
            // whatever the primary happens to be then.
            try {
              targets.set(job.clientId, currentTarget());
            } catch {
              // No live connection — the stream will fail with the same
              // directed error the old path produced.
            }
          }
        }
        if (chainRouting?.kind === "chain") chainRoutes.set(job.clientId, chainRouting);
        return job;
      });
      const durableAdmission =
        route?.heterogeneousBatch === true &&
        route.durableBatchOutcomes === true &&
        !!route.instanceId &&
        (route.heterogeneousBatchMaxOutputs == null ||
          size <= route.heterogeneousBatchMaxOutputs) &&
        chainRouting?.kind !== "chain" &&
        plans.every(requestIsEligibleForDurableGeneration);
      if (durableAdmission) {
        const clientBatchId = durableClientBatchId();
        const record: DurableGenerationRecoveryRecord = {
          tracker: createGenerationBatchTracker({
            hostId: route.hostId,
            expectedInstanceId: route.instanceId!,
            clientBatchId,
            submittedAtMs: Date.now(),
          }),
          hostLabel: route.label,
          hostKind: route.kind,
          mirrorRemoteOutput: route.mirrorRemoteOutput ?? true,
          children: plans.map((plan, index) =>
            durableChildSummary(plan, index + 1, jobs[index]!.clientId),
          ),
          effectReceipts: [],
        };
        durableRecords.set(clientBatchId, record);
        durableJobIds.set(
          clientBatchId,
          new Map(jobs.map((job, index) => [index + 1, job.clientId])),
        );
        const settlement = createDurableSettlement();
        durableSettlements.set(clientBatchId, settlement);
        // Crash authority exists before the first byte of the POST leaves.
        try {
          persistDurableRecords();
        } catch (error) {
          durableRecords.delete(clientBatchId);
          durableJobIds.delete(clientBatchId);
          durableSettlements.delete(clientBatchId);
          for (const job of jobs) {
            settleJob(job, "error");
            job.error = "Could not save durable recovery authority before submission.";
          }
          throw error;
        }
        void this.admitDurableRecord(record, plans);
        const settled = settlement.promise.then((settledJobs) => {
          this.pendingConsumerBatchIds = this.pendingConsumerBatchIds.filter(
            (pendingBatchId) => pendingBatchId !== batchId,
          );
          const pendingBatches = new Set(this.pendingConsumerBatchIds);
          this.prune(
            GENERATION_HISTORY_LIMIT,
            settledJobs.map((job) => job.clientId),
            this.jobs.filter((job) => !pendingBatches.has(job.batchId)).map((job) => job.clientId),
          );
          return settledJobs;
        });
        return { jobs, settled };
      }
      const tasks = jobs.map((job, i) => () => {
        // A sibling cancelled while it waited its turn never opens a stream.
        if (job.status === "error") return Promise.resolve();
        return this.streamJob(job, plans[i]!);
      });
      const legacyServerAdmission =
        size > 1 &&
        requestOptions.batchId !== undefined &&
        route?.heterogeneousBatch === true &&
        route.durableBatchOutcomes !== true &&
        (route.heterogeneousBatchMaxOutputs == null ||
          size <= route.heterogeneousBatchMaxOutputs) &&
        chainRouting?.kind !== "chain" &&
        !plans.some(requestNeedsReferenceUpload);
      const submit = legacyServerAdmission
        ? (async () => {
            const target = route!.target;
            const body = {
              client_batch_id: requestOptions.batchId!,
              requests: plans,
            };
            let admitted: GenerationBatchStatus;
            try {
              admitted = await apiJsonTo<GenerationBatchStatus>(target, "/api/generation-batches", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              });
            } catch (error) {
              if (!isTransportFailure(error)) throw error;
              admitted = await apiJsonTo<GenerationBatchStatus>(target, "/api/generation-batches", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
              });
            }
            if (admitted.children.length !== jobs.length) {
              throw new Error(
                `The host admitted ${admitted.children.length} of ${jobs.length} prepared variations.`,
              );
            }
            for (const [index, job] of jobs.entries()) {
              const child = admitted.children[index]!;
              if (child.index !== index + 1 || !child.job_id) {
                throw new Error("The host returned an invalid prepared-batch child mapping.");
              }
              job.id = child.job_id;
              job.streamStarted = true;
              job.status = "error";
              job.error = "The generation stream closed before completion.";
              job.interrupted = true;
              job.retainedByHost = true;
            }
            await this.reconcileInterrupted(jobs);
          })()
        : runWithConcurrency(tasks, MAX_STREAMS_PER_TARGET);
      const settled = submit
        .catch((error) => {
          for (const job of jobs) {
            if (job.status === "complete" || job.status === "error") continue;
            job.status = "error";
            markJobSettled(job);
            job.error = error instanceof Error ? error.message : String(error);
          }
        })
        // A stream that died while the host kept working is not an outcome.
        // Reconcile against the frozen route BEFORE anyone reads these jobs:
        // every shell renders `status: "error"` as a failure (desktop Create
        // also hides the matching live fleet row by id), so settling here is
        // what decides whether a retained generation reads as failed or as the
        // work it still is. Part of `settled`, so consumers see final state.
        .then(() => this.reconcileInterrupted(jobs))
        .then(() => {
          // Background notification (the view toasts in the foreground).
          const failed = jobs.find((s) => s.status === "error");
          const completed = jobs.find((s) => s.status === "complete");
          if (completed) notifyGenerated(completed.prompt, completed.result?.filename);
          else if (failed?.error && !failed.interrupted && !isCancelledError(failed.error)) {
            notifyGenerationFailed(describeTransportError(failed.error, failed.hostLabel));
          }
          // Consumers such as the iPhone UI promote the returned result in
          // their own promise callback. Defer housekeeping until that callback
          // has run and protect this batch if older jobs happened to settle
          // after newer ones.
          setTimeout(() => {
            this.pendingConsumerBatchIds = this.pendingConsumerBatchIds.filter(
              (pendingBatchId) => pendingBatchId !== batchId,
            );
            const pendingBatches = new Set(this.pendingConsumerBatchIds);
            this.prune(
              GENERATION_HISTORY_LIMIT,
              jobs.map((job) => job.clientId),
              this.jobs
                .filter((job) => !pendingBatches.has(job.batchId))
                .map((job) => job.clientId),
            );
          }, 0);
          return jobs;
        });
      return { jobs, settled };
    },
    /**
     * Settle every job whose stream died while the host kept going, by asking
     * the exact host it was submitted to.
     *
     * Both shells need this and neither can do it alone: the iPhone loses its
     * sockets to iOS suspension, and any surface loses them when a
     * durable-queue host restarts and retains the job. The job is reclaimed as
     * live work synchronously — before any consumer of `settled` runs — and
     * then settled by what the host actually did: the finished print, a
     * re-attached running job, or a directed human failure.
     */
    async reconcileInterrupted(jobs: readonly Job[]): Promise<void> {
      const candidates = jobs.filter(
        (job) =>
          job.status === "error" &&
          (job.interrupted || isInterruptedGenerationError(job.error)) &&
          !isCancelledError(job.error),
      );
      if (candidates.length === 0) return;
      // Group by the FROZEN route each job was submitted to — never a
      // re-resolved current host. Siblings normally share one.
      const groups = new Map<string, { target: ApiTarget; label: string; jobs: Job[] }>();
      for (const job of candidates) {
        // ONLY the frozen route. A job with none was never accepted anywhere
        // this client can name, and resolving "the primary" now could point at
        // a machine that never ran it — which recovery would then read prints
        // from, or DELETE a matching queued row on.
        const target = targets.get(job.clientId);
        if (!target) continue;
        const key = `${target.baseUrl}|${job.hostLabel ?? ""}`;
        const group = groups.get(key) ?? {
          target,
          label: job.hostLabel ?? new URL(target.baseUrl).hostname,
          jobs: [],
        };
        group.jobs.push(job);
        groups.set(key, group);
      }
      const hostStore = useHostsStore();
      await Promise.all(
        [...groups.values()].map((group) =>
          reconcileInterruptedGenerationJobs(group.jobs, {
            target: { ...group.target },
            hostLabel: group.label,
            queueCapacity:
              hostStore.telemetry[group.jobs[0]?.hostId ?? hostStore.primaryHost?.id ?? ""]
                ?.queueCapacity,
            chain: group.jobs.some((job) => chainRoutes.has(job.clientId)),
            refreshResultUrl: (clientId) =>
              void this.refreshRemoteResultUrl(clientId).catch(() => {
                // The reactive job carries the directed, user-visible error.
              }),
          }),
        ),
      );
    },
    /** Submit and wait for the whole batch (menu Generate, tests). */
    async generateBatch(req: GenerateRequest, batchSize: number): Promise<Job[]> {
      return this.submitBatch(req, batchSize).settled;
    },
    /**
     * Cancel one job (default: the canvas job). Queued and running ordinary jobs leave
     * the server queue via DELETE /api/queue/:id; automatic chains use their
     * durable shim id with POST /api/chain-jobs/:id/cancel. The stream is
     * aborted only after the server confirms cancellation. The `cancelling`
     * flag repaints every desktop/iPhone consumer on the initiating tap.
     */
    async cancel(clientId?: number): Promise<boolean> {
      const job =
        clientId !== undefined
          ? (this.jobs.find((j) => j.clientId === clientId) ?? null)
          : this.active;
      if (!job || job.status === "complete" || job.status === "error") return false;
      if (job.cancelling) return false;
      job.cancelling = true;
      const durableRecord = [...durableRecords.values()].find((record) =>
        [...(durableJobIds.get(record.tracker.clientBatchId)?.values() ?? [])].includes(
          job.clientId,
        ),
      );
      if (job.id) {
        try {
          const chainRoute = chainRoutes.get(job.clientId);
          await apiFetchTo(
            targets.get(job.clientId) ?? currentTarget(),
            chainRoute
              ? `/api/chain-jobs/${encodeURIComponent(job.id)}/cancel`
              : `/api/queue/${encodeURIComponent(job.id)}`,
            { method: chainRoute ? "POST" : "DELETE" },
          );
        } catch (err) {
          // A terminal SSE frame may win while DELETE is in flight. Preserve
          // that authoritative outcome; otherwise the failed DELETE means the
          // server still owns the job and the live stream must remain open.
          if (jobHasSettled(job)) return false;
          job.cancelling = false;
          throw err;
        }
      } else if (job.streamStarted) {
        job.cancelling = false;
        throw new Error("Remote cancellation was not confirmed before the queue ID arrived.");
      }
      if (durableRecord) {
        // DELETE is only a cancellation request. A concurrent completion can
        // win; only the exact host's durable snapshot decides the terminal.
        await this.reconcileDurableHost(durableRecord.tracker.hostId);
        job.cancelling = false;
        return job.error === "Cancelled";
      }
      // A terminal SSE frame may win while DELETE is in flight. Preserve that
      // authoritative outcome, even if the DELETE request itself then fails.
      if (jobHasSettled(job)) {
        job.cancelling = false;
        return false;
      }
      aborts.get(job.clientId)?.abort();
      aborts.delete(job.clientId);
      settleJob(job, "error");
      job.error = "Cancelled";
      job.cancelling = false;
      return true;
    },
    /** Single generation — a batch of one. */
    async generate(req: GenerateRequest): Promise<Job> {
      const [job] = await this.generateBatch(req, 1);
      return job!;
    },
    /** Drop finished jobs beyond the most recent few, releasing their URLs. */
    prune(
      keep = 12,
      preserveClientIds: number | Iterable<number> | null = null,
      eligibleClientIds: Iterable<number> | null = null,
    ) {
      const preserve = new Set<number>(
        preserveClientIds === null
          ? []
          : typeof preserveClientIds === "number"
            ? [preserveClientIds]
            : preserveClientIds,
      );
      const eligible = eligibleClientIds === null ? null : new Set(eligibleClientIds);
      const finished = this.jobs.filter(
        (job) =>
          (job.status === "complete" || job.status === "error") &&
          (!eligible || eligible.has(job.clientId)),
      );
      const rich = new Set(
        finished.slice(-Math.min(keep, GENERATION_RICH_HISTORY_LIMIT)).map((job) => job.clientId),
      );
      for (const clientId of preserve) rich.add(clientId);
      for (const job of finished) {
        if (rich.has(job.clientId)) continue;
        if (job.result) job.result = metadataOnlyResult(job.result);
        if (job.resultUrlIsObjectUrl && job.resultUrl) {
          URL.revokeObjectURL(job.resultUrl);
          job.resultUrl = null;
          job.resultUrlIsObjectUrl = false;
        }
        if (job.previewUrl) {
          URL.revokeObjectURL(job.previewUrl);
          job.previewUrl = null;
        }
      }
      const excess = finished.length - keep;
      if (excess <= 0) return;
      const drop = new Set(
        finished
          .filter((job) => !preserve.has(job.clientId))
          .slice(0, excess)
          .map((job) => job.clientId),
      );
      for (const job of this.jobs) {
        if (!drop.has(job.clientId)) continue;
        if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
        if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
        targets.delete(job.clientId);
        referenceUploadAuthorities.delete(job.clientId);
        chainRoutes.delete(job.clientId);
      }
      this.jobs = this.jobs.filter((j) => !drop.has(j.clientId));
      if (
        this.selectedClientId !== null &&
        !this.jobs.some((job) => job.clientId === this.selectedClientId)
      ) {
        this.selectedClientId = null;
      }
    },
    /** Acquire or renew the filename-backed URL for an opted-in mobile result. */
    async refreshRemoteResultUrl(clientId: number, force = false): Promise<void> {
      const job = this.jobs.find((candidate) => candidate.clientId === clientId);
      if (!job || job.status !== "complete" || !job.result) return;
      if (job.resultUrlLoading) return;
      if (
        !force &&
        job.resultUrl &&
        (job.resultUrlExpiresAt === null || job.resultUrlExpiresAt > Date.now() + 60_000)
      ) {
        return;
      }

      const filename = job.result.filename;
      const target = targets.get(job.clientId);
      if (!filename || !target) {
        job.resultUrl = null;
        job.resultUrlExpiresAt = null;
        job.resultError = !filename
          ? "This host did not provide a saved result URL. Update the host and try again."
          : "The result host is no longer available.";
        throw new Error(job.resultError);
      }

      const path = galleryMediaPath(filename, "host");
      const cacheKey = job.hostId ?? `job-${job.clientId}`;
      // Older authenticated hosts fall back to a cached Blob URL for images.
      // A forced retry must discard that entry first; otherwise a revoked or
      // failed Blob is returned forever and the manual retry cannot recover.
      if (force) evictMedia(path, cacheKey);

      job.resultUrlLoading = true;
      job.resultError = null;
      try {
        const url = await streamableMediaUrl(path, {
          target,
          cacheKey,
          allowLegacyBlob: job.result.format !== "mp4",
        });
        if (
          job.status === "complete" &&
          this.jobs.some((candidate) => candidate.clientId === job.clientId)
        ) {
          const previousUrl = job.resultUrl;
          const previousWasObjectUrl = job.resultUrlIsObjectUrl;
          job.resultUrl = url;
          job.resultUrlIsObjectUrl = url.startsWith("blob:");
          job.resultUrlExpiresAt = resultUrlExpiry(url);
          if (previousUrl && previousWasObjectUrl && previousUrl !== url) {
            URL.revokeObjectURL(previousUrl);
          }
        }
      } catch (error) {
        if (this.jobs.some((candidate) => candidate.clientId === job.clientId)) {
          if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
          job.resultUrl = null;
          job.resultUrlIsObjectUrl = false;
          job.resultUrlExpiresAt = null;
          job.resultError = error instanceof Error ? error.message : String(error);
        }
        throw error;
      } finally {
        job.resultUrlLoading = false;
      }
    },
    /** Revoke every held object URL and clear all jobs (teardown/tests). */
    targetForJob(clientId: number): ApiTarget | null {
      return targets.get(clientId) ?? null;
    },
    resetJobs() {
      for (const job of this.jobs) {
        if (!jobHasSettled(job)) {
          settleJob(job, "error");
          job.error = "Cancelled";
        }
        aborts.get(job.clientId)?.abort();
        if (job.resultUrl && job.resultUrlIsObjectUrl) URL.revokeObjectURL(job.resultUrl);
        if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      }
      aborts.clear();
      for (const stream of durableHostStreams.values()) stream.abort.abort();
      durableHostStreams.clear();
      durableReconciles.clear();
      durableJobIds.clear();
      durableSettlements.clear();
      durableRecords.clear();
      sharedDurableEventHosts.clear();
      durableRecoveryLoaded = false;
      targets.clear();
      referenceUploadAuthorities.clear();
      chainRoutes.clear();
      this.pendingConsumerBatchIds = [];
      this.selectedClientId = null;
      this.jobs = [];
    },
    startJob(req: GenerateRequest): Job {
      // reactive() here is load-bearing: the SSE handlers below mutate the
      // returned reference from a closure. A raw object would update the
      // data without firing Vue's proxy traps — the canvas, edge code, and
      // job chips would sit frozen at "Queued 0/N" for the whole run.
      const job = reactive(newJob(req));
      job.clientId = this.nextClientId++;
      this.jobs.push(job);
      return job;
    },
    async streamJob(job: Job, req: GenerateRequest): Promise<void> {
      const abort = new AbortController();
      aborts.set(job.clientId, abort);
      const target = targets.get(job.clientId);
      const releaseStreamSlot = await streamSlots.acquire(
        target?.baseUrl ?? "__primary__",
        abort.signal,
      );
      if (!releaseStreamSlot || abort.signal.aborted || job.status === "error") {
        releaseStreamSlot?.();
        aborts.delete(job.clientId);
        return;
      }

      let streamError: unknown = null;
      let lease: ReferenceUploadLease<GenerateRequest> | null = null;
      const chainRoute = chainRoutes.get(job.clientId);
      const path = chainRoute ? "/api/generate/chain/stream" : "/api/generate/stream";
      let transportRequest = req;
      if (requestNeedsReferenceUpload(req)) {
        try {
          if (chainRoute) {
            throw new Error(
              "MiniMax H3 reference media cannot be submitted through a chain route.",
            );
          }
          const authority = referenceUploadAuthorities.get(job.clientId);
          if (!authority) {
            throw new Error(
              "MiniMax H3 reference uploads require a frozen authenticated host route.",
            );
          }
          const prepared = await prepareReferenceUploads({
            target: authority.target,
            expectedInstanceId: authority.instanceId,
            capabilities: authority.capabilities,
            request: req,
            signal: abort.signal,
          });
          lease = prepared;
          transportRequest = prepared.request;
        } catch (error) {
          if (!abort.signal.aborted && !jobHasSettled(job)) {
            settleJob(job, "error");
            job.interrupted = false;
            job.error = error instanceof Error ? error.message : String(error);
          }
          releaseStreamSlot();
          aborts.delete(job.clientId);
          return;
        }
      }
      job.streamStarted = true;
      const body = chainRoute ? buildAutoChainRequest(req, chainRoute) : transportRequest;
      // Freeze the exact host this stream opens against. Submit already
      // snapshots the primary, but a job admitted while nothing was connected
      // has no route recorded — and `sseStream` would then resolve the primary
      // AGAIN, later, possibly a different machine. Recovery and cancel key off
      // this map, and asking (or DELETEing on) a host that never ran the job is
      // the one outcome the frozen-route invariant exists to prevent.
      const streamTarget = target ?? connectedTarget();
      if (streamTarget && !targets.has(job.clientId)) {
        targets.set(job.clientId, streamTarget);
      }
      await sseStream(path, {
        method: "POST",
        body,
        signal: abort.signal,
        retry: false,
        ...(job.metadataOnlyCompletion
          ? { headers: { "X-Mold-SSE-Payload": "metadata-only" } }
          : {}),
        ...(streamTarget ? { target: streamTarget } : {}),
        onOpen: (response) => {
          job.requestWarnings = requestWarningsFromHeaders(response.headers);
        },
        onEvent: (event, data) => {
          const current = job;
          // Abort/reset/cancel and terminal frames are final. Some SSE
          // implementations can still deliver already-buffered callbacks;
          // ignoring them prevents a cancelled job from being resurrected.
          if (abort.signal.aborted || jobHasSettled(current)) return;
          try {
            if (event === "progress") {
              if (chainRoute) {
                applyChainProgress(current, JSON.parse(data) as ChainProgressEvent);
              } else {
                applyProgress(current, JSON.parse(data) as ProgressEvent);
              }
            } else if (event === "complete") {
              const complete = chainRoute
                ? chainCompleteToComplete(JSON.parse(data) as SseChainCompleteEvent, req)
                : (JSON.parse(data) as CompleteEvent);
              applyCompletionWarnings(current, complete);
              const useSavedResult =
                !complete.image ||
                (current.metadataOnlyCompletion &&
                  complete.format === "mp4" &&
                  !!complete.filename);
              if (complete.image && !useSavedResult) {
                current.resultUrl = base64ToBlobUrl(
                  complete.image,
                  MIME[complete.format] ?? "application/octet-stream",
                );
                current.resultUrlIsObjectUrl = true;
              }
              current.result = current.retainEncodedResult
                ? complete
                : metadataOnlyResult(complete);
              current.visualSeed = String(complete.seed_used);
              settleJob(current, "complete");
              if (useSavedResult) {
                void this.refreshRemoteResultUrl(current.clientId).catch(() => {
                  // The reactive job carries the directed, user-visible error.
                });
              }
              if (current.previewUrl) {
                URL.revokeObjectURL(current.previewUrl);
                current.previewUrl = null;
              }
              // Remote prints also land in this Mac's gallery (pref-gated):
              // the SSE payload is the encoded output file, so no extra
              // download is needed. Newer servers also send the gallery
              // filename and recorded metadata — keeping the origin's name
              // makes the copy and the original one logical print in the
              // merged gallery, and the metadata gives video copies (which
              // embed nothing) their true dimensions and provenance.
              if (
                current.remote &&
                current.mirrorRemoteOutput &&
                complete.image &&
                (useAppPrefsStore().settings?.saveRemoteOutputs ?? true)
              ) {
                const now = Date.now();
                const meta = complete.metadata ?? null;
                const originalMeta =
                  meta && complete.original_width && complete.original_height
                    ? {
                        ...meta,
                        width: complete.original_width,
                        height: complete.original_height,
                      }
                    : meta;
                const saves = complete.original_image
                  ? [
                      ipc.saveOutputBytes(
                        complete.original_filename ??
                          suggestOutputFilename(
                            complete.model,
                            complete.seed_used,
                            complete.format,
                            now,
                            "original",
                          ),
                        complete.original_image,
                        originalMeta,
                      ),
                      ipc.saveOutputBytes(
                        complete.filename ??
                          suggestOutputFilename(
                            complete.model,
                            complete.seed_used,
                            complete.format,
                            now,
                            "upscaled",
                          ),
                        complete.image,
                        meta,
                      ),
                    ]
                  : [
                      ipc.saveOutputBytes(
                        complete.filename ??
                          suggestOutputFilename(
                            complete.model,
                            complete.seed_used,
                            complete.format,
                            now,
                          ),
                        complete.image,
                        meta,
                      ),
                    ];
                Promise.allSettled(saves).then((results) => {
                  for (const result of results) {
                    if (result.status === "rejected") {
                      console.warn("local save of remote output failed:", result.reason);
                    }
                  }
                  if (results.some((result) => result.status === "fulfilled")) {
                    void useGalleryStore().refreshHost("local");
                  }
                });
              }
              // Nudge the unified gallery's bucket for the host this print
              // landed on. refreshHost only refetches already-loaded buckets
              // — a background completion must not force-load a gallery
              // bucket the user never opened.
              const originHostId = current.hostId ?? useHostsStore().primaryHost?.id ?? null;
              if (originHostId) void useGalleryStore().refreshHost(originHostId);
              abort.abort();
            } else if (event === "error") {
              settleJob(current, "error");
              try {
                const parsed = JSON.parse(data) as {
                  error?: string;
                  message?: string;
                  retained?: boolean;
                };
                const message = parsed.error ?? parsed.message ?? data;
                current.error = isCancelledError(message) ? "Cancelled" : message;
                // A durable-queue host ends a retained job's stream with an
                // error frame carrying `retained` while keeping the work: it
                // will run and land in that host's gallery. Treating it as an
                // interruption suppresses the "failed" notification and hands
                // the job to reconciliation, exactly like a dead socket —
                // `retainedByHost` additionally tells reconciliation to wait
                // out the restart instead of the much shorter suspension budget.
                current.interrupted = parsed.retained === true;
                current.retainedByHost = parsed.retained === true;
              } catch {
                current.error = isCancelledError(data) ? "Cancelled" : data;
              }
              abort.abort();
            }
          } catch {
            if (current.status !== "complete" && current.status !== "error") {
              settleJob(current, "error");
              current.error = "The host returned an invalid generation update.";
              abort.abort();
            }
          }
        },
        onClose: (err) => {
          if (err && !abort.signal.aborted && !jobHasSettled(job)) {
            settleJob(job, "error");
            job.error = err.message;
            // fetch-event-source reports network/transport loss as TypeError.
            // HTTP/auth failures are deterministic and must remain final —
            // suppressing their notification and retrying on foreground would
            // only hide the actual server response.
            job.interrupted = err instanceof TypeError;
          }
        },
      }).catch((error: unknown) => {
        streamError = error;
      });
      if (lease) void lease.cancel().catch(() => undefined);
      if (!abort.signal.aborted && !jobHasSettled(job)) {
        settleJob(job, "error");
        job.interrupted = streamError === null || streamError instanceof TypeError;
        job.error = streamError
          ? streamError instanceof Error
            ? streamError.message
            : String(streamError)
          : "The generation stream closed before completion.";
      }
      releaseStreamSlot();
      aborts.delete(job.clientId);
    },
  },
});

/**
 * Per-job stream handles, outside Pinia state — AbortControllers are
 * process-local plumbing, not renderable state.
 */
const aborts = new Map<number, AbortController>();

/**
 * Per-job API targets for multi-host routing, keyed by clientId. Kept out of
 * Pinia state (they carry API keys and are plumbing, not renderable state);
 * jobs without an entry use the primary connection.
 */
const targets = new Map<number, ApiTarget>();

/** Per-attempt authority for one-use H3 ingress, never Pinia/persistence. */
const referenceUploadAuthorities = new Map<number, ReferenceUploadAuthority>();

/** Automatic-chain routing snapshot for endpoint selection and cancellation. */
const chainRoutes = new Map<number, AutoChainRoutingDecision>();

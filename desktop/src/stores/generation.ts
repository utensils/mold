import { reactive } from "vue";
import { defineStore } from "pinia";
import { apiFetchTo, currentTarget, type ApiTarget, apiJsonTo } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import {
  evictMedia,
  fetchGalleryMediaBytes,
  galleryMediaPath,
  streamableMediaUrl,
} from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { notifyGenerated, notifyGenerationFailed } from "../lib/notify";
import { describeTransportError } from "../lib/api/errors";
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
  OutputMetadata,
  GalleryImage,
} from "../lib/api/types";
import {
  buildAutoChainRequest,
  type AutoChainRoutingDecision,
  type ChainRoutingDecision,
} from "../lib/chainRouting";
import {
  applyChainProgress,
  isCancelledError,
  markJobSettled,
  metadataOnlyResult,
  newJob,
  type Job,
} from "../lib/generationJob";
import { type ReferenceUploadCapabilities } from "@studio/api/referenceUploads";
import { requestWarningsFromHeaders } from "@studio/lib/requestWarnings";
import { emptyChainJobLive, reduceChainJobFrame } from "@studio/lib/chainJobProgress";
import { OwnPrintPreviewWatchers, previewDataUrl } from "@studio/api/ownPrintPreview";
import type { ChainJobEvent, CreateChainJobResponse } from "@studio/lib/api/chainTypes";
import { blobToBase64 } from "@studio/lib/base64";
import {
  admitGenerationBatch,
  canonicalGenerationBatchLimit,
  chunkGenerationBatchRequests,
  isDefiniteGenerationAdmissionRejection,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
  type DurableGenerationQueueCapabilities,
  type DurableMediaCapabilities,
  type GenerationBatchStatus,
} from "@studio/api/generationAdmission";
import {
  buildGenerationBatchStatusRequest,
  chunkGenerationBatchTrackers,
  createGenerationBatchTracker,
  isTerminalGenerationPhase,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
} from "@studio/lib/generationLifecycle";
import {
  generationTrackerSettled,
  presentGenerationChild,
  reconciliationPresentation,
  type GenerationChildPresentation,
} from "@studio/lib/generationPresentation";
import { applyDurablePresentation } from "../lib/durableGenerationPresentation";
import {
  durableChildSummary,
  loadDurableGenerationRecovery,
  parseEventAuthority,
  parseEventResync,
  generationRefusalReason,
  saveDurableGenerationRecovery,
  type DurableGenerationRecoveryRecord,
} from "../lib/durableGeneration";
import { TargetStreamSlots } from "@studio/lib/targetStreamSlots";
import { useToastStore } from "./toasts";
import { retryQueueJobRecoveringAmbiguity } from "@studio/api/queuePlan";

export {
  applyChainProgress,
  isCancelledError,
  jobPhase,
  jobProgress,
  jobProgressCopy,
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
  /** The frozen host's durable batch chunk limit. Its presence IS the durable
   * generation contract. */
  heterogeneousBatchMaxOutputs?: number | null;
  durableMedia?: DurableMediaCapabilities | null;
  modelFamily?: string | null;
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
/** Stable batch membership survives presentation-row dismissal. */
const durableBatchJobs = new Map<string, Map<number, Job>>();
/** Effect-only recovered jobs participate in settlement but never load UI media. */
const durableHiddenJobIds = new Set<number>();
const durableSettlements = new Map<string, DurableSettlement>();
/** Live preview + step progress for our own running prints (see web). */
const ownPreviews = new OwnPrintPreviewWatchers();

/**
 * The origin host's own gallery row for one print. `GET /api/gallery` filters
 * by filename server-side, so this is a single-row read through the same
 * DB / archive / organization pipeline the listing uses. `null` when the row
 * is not there (yet) or the host cannot answer — the mirror still saves the
 * bytes, with the metadata the desktop synthesizes from the filename.
 */
async function originGalleryMetadata(
  target: ApiTarget,
  filename: string,
): Promise<OutputMetadata | null> {
  try {
    const rows = await apiJsonTo<GalleryImage[]>(
      target,
      `/api/gallery?filename=${encodeURIComponent(filename)}`,
    );
    const row = Array.isArray(rows) ? rows.find((entry) => entry.filename === filename) : null;
    return row?.metadata ?? null;
  } catch {
    return null;
  }
}
const durableHostStreams = new Map<string, DurableHostStream>();
const durableReconciles = new Map<string, Promise<void>>();
/** Missing = no follow-up; null = host-wide; Set = selected client batches. */
const durableReconcilePending = new Map<string, Set<string> | null>();
const durableCancelAttempts = new Map<string, Promise<boolean>>();
const sharedDurableEventHosts = new Set<string>();
let durableRecoveryLoaded = false;
let durableRecoveryStorageUnavailable = false;

function durableRecordForClientId(clientId: number): DurableGenerationRecoveryRecord | null {
  return (
    [...durableRecords.values()].find((record) =>
      [...(durableJobIds.get(record.tracker.clientBatchId)?.values() ?? [])].includes(clientId),
    ) ?? null
  );
}

const DURABLE_RECOVERY_STORAGE_WARNING =
  "Recovery storage is unavailable. This generation is still being submitted, but reloading before Mold confirms it may hide it from Create.";

function persistDurableRecords(): boolean {
  const saved = saveDurableGenerationRecovery(durableRecords.values());
  if (saved) {
    durableRecoveryStorageUnavailable = false;
    return true;
  }
  if (!durableRecoveryStorageUnavailable) {
    useToastStore().push(DURABLE_RECOVERY_STORAGE_WARNING, "warning");
  }
  durableRecoveryStorageUnavailable = true;
  return false;
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

function createStoreJob(req: GenerateRequest, clientId: number): Job {
  const job = reactive(newJob(req));
  job.clientId = clientId;
  return job;
}

function recordIsSettled(record: DurableGenerationRecoveryRecord): boolean {
  return generationTrackerSettled(record.tracker, record.children.length);
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

export function jobCanBeRemoved(job: Job): boolean {
  return jobHasSettled(job) && !job.interrupted && !job.retainedByHost;
}

function dismissalReceipt(childIndex: number): string {
  return `dismissed:${childIndex}`;
}

function recordIsFullyDismissed(record: DurableGenerationRecoveryRecord): boolean {
  return record.children.every((child) =>
    record.effectReceipts.includes(dismissalReceipt(child.index)),
  );
}

/** Release every client-local resource owned by one settled activity row. */
function releaseSettledJob(job: Job): void {
  if (job.resultUrl && job.resultUrlIsObjectUrl) {
    if (job.result?.filename) {
      evictMedia(
        galleryMediaPath(job.result.filename, "host"),
        job.hostId ?? `job-${job.clientId}`,
      );
    }
    URL.revokeObjectURL(job.resultUrl);
  }
  if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
  targets.delete(job.clientId);
  chainRoutes.delete(job.clientId);
  chainJobIds.delete(job.clientId);
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

/** The frozen route's queue capability, in the one shape the shared policy
 * reads. The chunk limit's presence IS the durable-generation contract. */
function routeQueueCapabilities(route: JobRoute | null): DurableGenerationQueueCapabilities {
  return { heterogeneous_batch_max_outputs: route?.heterogeneousBatchMaxOutputs ?? null };
}

/**
 * The machine this batch runs on. An unrouted submit is **This device**: the
 * app's own embedded server is a machine like any other, and durable
 * admission needs its instance identity and advertised limit, so the primary's
 * full route is resolved here rather than left as a bare target.
 */
function effectiveJobRoute(route: JobRoute | null, model: string | null): JobRoute | null {
  if (route) return route;
  const hosts = useHostsStore();
  const primaryId = hosts.primaryHost?.id ?? null;
  return (
    (primaryId ? hosts.resolveRoute(primaryId, model) : null) ?? hosts.resolveRoute(null, model)
  );
}

/** The named reason this batch cannot be queued on its frozen machine, or
 * `null` when it can. Host-level: the durable protocol carries every request
 * trait, so the server's typed admission refusal is the only authority for
 * what it cannot take. */
function generationRouteRefusal(route: JobRoute | null): string | null {
  if (!route) return "No machine is selected for this print.";
  if (!route.instanceId) {
    return `${route.label} has not reported its server instance yet.`;
  }
  const reason = generationRefusalReason(routeQueueCapabilities(route), route.durableMedia);
  return reason === null ? null : `${route.label} cannot queue this print: ${reason}.`;
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
        const batchJobs = new Map<number, Job>();
        const localBatchId = this.nextBatchId++;
        const recoveredHost = useHostsStore().all.find(
          (candidate) =>
            candidate.id === record.tracker.hostId &&
            candidate.instanceId === record.tracker.expectedInstanceId &&
            candidate.baseUrl,
        );
        for (const summary of record.children) {
          const dismissed = record.effectReceipts.includes(dismissalReceipt(summary.index));
          const job = createStoreJob(
            {
              prompt: "Recovered generation",
              model: summary.model,
              width: summary.width,
              height: summary.height,
              steps: summary.steps,
              guidance: summary.guidance,
              ...(summary.seed === null ? {} : { seed: summary.seed }),
              output_format: summary.format,
            },
            this.nextClientId++,
          );
          if (dismissed) durableHiddenJobIds.add(job.clientId);
          else this.jobs.push(job);
          job.batchId = localBatchId;
          job.hostId = record.tracker.hostId;
          job.hostLabel = record.hostLabel;
          job.remote = record.hostKind === "remote";
          job.mirrorRemoteOutput = record.mirrorRemoteOutput;
          job.streamStarted = true;
          const restoredLifecycle = Object.values(record.tracker.jobs).find(
            (candidate) => candidate.childIndex === summary.index,
          );
          job.suppressFreshCompletion = restoredLifecycle
            ? isTerminalGenerationPhase(restoredLifecycle.phase)
            : false;
          job.cancelling = record.cancelRequestedChildIndexes.includes(summary.index);
          mapping.set(summary.index, job.clientId);
          batchJobs.set(summary.index, job);
          summary.clientId = job.clientId;
          if (recoveredHost?.baseUrl) {
            targets.set(job.clientId, {
              baseUrl: recoveredHost.baseUrl,
              apiKey: recoveredHost.apiKey,
            });
          }
        }
        durableJobIds.set(record.tracker.clientBatchId, mapping);
        durableBatchJobs.set(record.tracker.clientBatchId, batchJobs);
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
        if (record.tracker.hostId !== hostId || recordIsSettled(record)) continue;
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
        (record) => record.tracker.hostId === hostId && !recordIsSettled(record),
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
        const owner = records.find((record) =>
          Object.values(record.tracker.jobs).some((job) => job.authority.jobId === frame.id),
        );
        if (frame.type === "job_state_committed") {
          void this.reconcileDurableHost(
            hostId,
            owner ? new Set([owner.tracker.clientBatchId]) : undefined,
          );
        } else if (frame.type === "generation_states_committed") {
          void this.reconcileDurableHost(hostId);
        } else if (owner && (frame.type === "job_queued" || frame.type === "job_started")) {
          // The running transition emits no commit hint; `job_ended` and
          // `gallery_added` precede the one that follows every settlement.
          void this.reconcileDurableHost(hostId, new Set([owner.tracker.clientBatchId]));
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
          !recordIsSettled(candidate),
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
          .filter((record) => !recordIsSettled(record))
          .map((record) => record.tracker.hostId),
      );
      await Promise.all([...hostIds].map((hostId) => this.reconcileDurableHost(hostId)));
      await Promise.all(
        [...durableRecords.values()].filter(recordIsSettled).map((record) => {
          const jobs = [...(durableBatchJobs.get(record.tracker.clientBatchId)?.values() ?? [])];
          return this.finishDurableBatchEffects(record, jobs);
        }),
      );
    },
    async reconcileDurableHost(
      hostId: string,
      clientBatchIds?: ReadonlySet<string>,
    ): Promise<void> {
      const existing = durableReconciles.get(hostId);
      if (existing) {
        const pending = durableReconcilePending.get(hostId);
        if (clientBatchIds === undefined) {
          durableReconcilePending.set(hostId, null);
        } else if (pending !== null) {
          const next = pending ?? new Set<string>();
          for (const clientBatchId of clientBatchIds) next.add(clientBatchId);
          durableReconcilePending.set(hostId, next);
        }
        return existing;
      }
      const operation = (async () => {
        const records = [...durableRecords.values()].filter(
          (record) =>
            record.tracker.hostId === hostId &&
            !recordIsSettled(record) &&
            (!clientBatchIds || clientBatchIds.has(record.tracker.clientBatchId)),
        );
        const host = useHostsStore().all.find((candidate) => candidate.id === hostId);
        if (records.length > 0 && host?.baseUrl && host.status === "ready") {
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
          if (matching.length > 0) {
            const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
            const trackers = matching.map((record) => record.tracker);
            for (const trackerChunk of chunkGenerationBatchTrackers(trackers, hostId)) {
              const request = buildGenerationBatchStatusRequest(trackerChunk, hostId);
              if (request.client_batch_ids.length === 0 && !request.batch_ids?.length) continue;
              const response = await reconcileGenerationBatches(target, request);
              const merged = mergeBulkGenerationBatchResponse(trackerChunk, hostId, response);
              for (const tracker of merged.trackers) {
                const record = matching.find(
                  (candidate) => candidate.tracker.clientBatchId === tracker.clientBatchId,
                );
                if (!record) continue;
                record.tracker = tracker;
                this.applyDurableRecord(record);
              }
            }
          }
          persistDurableRecords();
        }
      })();
      durableReconciles.set(hostId, operation);
      try {
        await operation;
      } catch {
        // Last-good authority remains visible; the next event/reconnect/wake retries.
      } finally {
        if (durableReconciles.get(hostId) === operation) durableReconciles.delete(hostId);
        const pending = durableReconcilePending.get(hostId);
        if (pending !== undefined) {
          durableReconcilePending.delete(hostId);
          void this.reconcileDurableHost(hostId, pending ?? undefined);
        }
      }
    },
    applyDurableRecord(record: DurableGenerationRecoveryRecord): void {
      const batchJobs = durableBatchJobs.get(record.tracker.clientBatchId);
      if (!durableJobIds.has(record.tracker.clientBatchId) || !batchJobs) return;
      const now = Date.now();
      for (const [childIndex, job] of batchJobs) {
        const lifecycle =
          Object.values(record.tracker.jobs).find(
            (candidate) => candidate.childIndex === childIndex,
          ) ?? null;
        if (lifecycle) {
          job.id = lifecycle.authority.jobId;
          job.streamStarted = true;
        }
        const p = presentGenerationChild({
          tracker: record.tracker,
          childIndex,
          hostLabel: record.hostLabel,
          now,
        });
        // The durable child carries no denoise preview or step count; poll
        // the host's `/api/queue/{id}/preview` for our own running print
        // exactly as an inspected queue row does, and stop when it leaves
        // `running`.
        const previewTarget = targets.get(job.clientId) ?? null;
        if (p.kind === "running" && lifecycle && previewTarget && !jobHasSettled(job)) {
          ownPreviews.ensure(
            String(job.clientId),
            previewTarget,
            lifecycle.authority.jobId,
            (preview) => {
              if (jobHasSettled(job)) return;
              job.previewUrl = previewDataUrl(preview);
              job.step = preview.step;
              job.total = preview.total;
            },
          );
        } else {
          ownPreviews.stop(String(job.clientId));
        }
        if (p.kind === "complete") this.applyDurableCompletion(record, childIndex, p, job);
        else applyDurablePresentation(job, p);
      }
      this.scheduleDurableCancelIntents(record);
      if (!recordIsSettled(record)) return;
      const jobs = [...batchJobs.values()];
      durableSettlements.get(record.tracker.clientBatchId)?.resolve(jobs);
      durableSettlements.delete(record.tracker.clientBatchId);
      void this.finishDurableBatchEffects(record, jobs);
    },
    async retryHeld(clientId: number): Promise<void> {
      const job = this.jobs.find((candidate) => candidate.clientId === clientId);
      if (!job || !job.id || !job.retryable || job.retrying) {
        throw new Error("This held generation is not retryable yet.");
      }
      const record = durableRecordForClientId(clientId);
      if (!record || record.tracker.reconciliation.reason === "instance_mismatch") {
        throw new Error("The original machine identity changed; Retry is unavailable.");
      }
      if (!record.tracker.serverBatchId) {
        throw new Error("The durable batch identity is unavailable; Retry is unavailable.");
      }
      const host = useHostsStore().all.find((candidate) => candidate.id === record.tracker.hostId);
      if (
        !host?.baseUrl ||
        host.status !== "ready" ||
        host.instanceId !== record.tracker.expectedInstanceId
      ) {
        job.retryable = false;
        throw new Error("The original machine is not connected with the same identity.");
      }
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
      job.retrying = true;
      job.retryable = false;
      try {
        const outcome = await retryQueueJobRecoveringAmbiguity(target, {
          instanceId: record.tracker.expectedInstanceId,
          batchId: record.tracker.serverBatchId,
          clientBatchId: record.tracker.clientBatchId,
          jobId: job.id,
        });
        if (outcome.kind === "reconciled") {
          record.tracker = reduceGenerationLifecycle(record.tracker, {
            type: "batch_snapshot",
            batch: outcome.batch,
          });
          this.applyDurableRecord(record);
          persistDurableRecords();
          return;
        }
        if (outcome.kind === "uncertain") {
          job.holdError = outcome.error;
          void this.reconcileDurableHost(
            record.tracker.hostId,
            new Set([record.tracker.clientBatchId]),
          );
          throw new Error(outcome.error);
        }
        job.holdError = null;
        job.holdCode = null;
        job.stage = null;
        void this.reconcileDurableHost(
          record.tracker.hostId,
          new Set([record.tracker.clientBatchId]),
        );
      } catch (error) {
        void this.reconcileDurableHost(
          record.tracker.hostId,
          new Set([record.tracker.clientBatchId]),
        );
        throw error;
      } finally {
        job.retrying = false;
      }
    },
    async fulfillDurableCancelIntent(
      record: DurableGenerationRecoveryRecord,
      childIndex: number,
    ): Promise<boolean> {
      const attemptKey = `${record.tracker.clientBatchId}:${childIndex}`;
      const existing = durableCancelAttempts.get(attemptKey);
      if (existing) return existing;
      const attempt = (async () => {
        const current = durableRecords.get(record.tracker.clientBatchId);
        if (!current?.cancelRequestedChildIndexes.includes(childIndex)) return false;
        const clientId = durableJobIds.get(current.tracker.clientBatchId)?.get(childIndex);
        const job = this.jobs.find((candidate) => candidate.clientId === clientId);
        if (!job) return false;
        if (jobHasSettled(job)) {
          current.cancelRequestedChildIndexes = current.cancelRequestedChildIndexes.filter(
            (index) => index !== childIndex,
          );
          job.cancelling = false;
          persistDurableRecords();
          return isCancelledError(job.error);
        }
        if (!job.id) return false;
        const host = useHostsStore().all.find(
          (candidate) =>
            candidate.id === current.tracker.hostId &&
            candidate.instanceId === current.tracker.expectedInstanceId &&
            candidate.baseUrl &&
            candidate.status === "ready",
        );
        if (!host?.baseUrl) return false;
        try {
          await apiFetchTo(
            { baseUrl: host.baseUrl, apiKey: host.apiKey },
            `/api/queue/${encodeURIComponent(job.id)}`,
            { method: "DELETE" },
          );
        } catch (error) {
          await this.reconcileDurableHost(current.tracker.hostId);
          if (jobHasSettled(job)) return isCancelledError(job.error);
          throw error;
        }
        await this.reconcileDurableHost(current.tracker.hostId);
        if (jobHasSettled(job)) return isCancelledError(job.error);
        return false;
      })().finally(() => {
        durableCancelAttempts.delete(attemptKey);
      });
      durableCancelAttempts.set(attemptKey, attempt);
      return attempt;
    },
    scheduleDurableCancelIntents(record: DurableGenerationRecoveryRecord): void {
      for (const childIndex of record.cancelRequestedChildIndexes) {
        void this.fulfillDurableCancelIntent(record, childIndex).catch(() => {
          // The persisted intent remains authoritative. A later event, wake,
          // or reconnect retries against the same host instance and job id.
        });
      }
    },
    /** The one arm the store maps itself: the result is built from the
     * recovery record's child summary, and its media URL is host-exact. */
    applyDurableCompletion(
      record: DurableGenerationRecoveryRecord,
      childIndex: number,
      p: Extract<GenerationChildPresentation, { kind: "complete" }>,
      job: Job,
    ): void {
      if (jobHasSettled(job)) return;
      const summary = record.children.find((child) => child.index === childIndex)!;
      job.result = {
        image: "",
        format: requestFormat(p.filename, summary.format),
        width: summary.width,
        height: summary.height,
        seed_used: summary.seed ?? 0,
        generation_time_ms: p.generationTimeMs,
        model: summary.model,
        filename: p.filename,
        ...(p.originalFilename ? { original_filename: p.originalFilename } : {}),
        metadata: null,
      };
      job.visualSeed = String(summary.seed ?? 0);
      job.cancelling = false;
      job.settledAtMs ??= p.settledAtMs;
      settleJob(job, "complete");
      if (job.previewUrl) {
        URL.revokeObjectURL(job.previewUrl);
        job.previewUrl = null;
      }
      if (!durableHiddenJobIds.has(job.clientId)) {
        void this.refreshRemoteResultUrl(job.clientId).catch(() => undefined);
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
      // An unknown outcome is advisory: no notification, no mirror, only the
      // dismissal bookkeeping below.
      const unknown =
        reconciliationPresentation(record.tracker.reconciliation, null).kind === "unknown";
      const completed = unknown
        ? []
        : jobs.filter((job) => job.status === "complete" && job.result?.filename);
      if (completed.length > 0 && claim("native-notification")) {
        notifyGenerated(completed[0]!.prompt, completed[0]!.result?.filename);
      } else if (!unknown) {
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
              // The durable child names only the file; the origin's gallery
              // row is where its prompt, seed, dimensions, and timing live.
              // Mirror the print WITH that metadata so This device's copy is
              // a real print (reuse, cross-host collapse) rather than a file
              // whose metadata was synthesized from its name.
              const metadata = await originGalleryMetadata(target, filename);
              if (metadata && result.filename === filename) result.metadata = metadata;
              const bytes = await fetchGalleryMediaBytes(
                galleryMediaPath(filename, "host"),
                target,
              );
              const buffer = Uint8Array.from(bytes).buffer;
              await ipc.saveOutputBytes(filename, await blobToBase64(new Blob([buffer])), metadata);
              void useGalleryStore().refreshHost("local");
            } catch (error) {
              console.warn("local save of remote durable output failed:", error);
            }
          }
        }
      }
      if (completed.length > 0) void useGalleryStore().refreshHost(record.tracker.hostId);
      // Terminal recovery records double as persistent activity history. They
      // are execution-inert, but remain restorable until every child has been
      // explicitly dismissed (or aged out through the same removal action).
      if (!recordIsFullyDismissed(record)) {
        persistDurableRecords();
        return;
      }
      durableRecords.delete(record.tracker.clientBatchId);
      durableJobIds.delete(record.tracker.clientBatchId);
      const retiredJobs = durableBatchJobs.get(record.tracker.clientBatchId);
      durableBatchJobs.delete(record.tracker.clientBatchId);
      for (const job of retiredJobs?.values() ?? []) {
        targets.delete(job.clientId);
        if (durableHiddenJobIds.delete(job.clientId)) releaseSettledJob(job);
      }
      persistDurableRecords();
      if (
        ![...durableRecords.values()].some(
          (item) => item.tracker.hostId === record.tracker.hostId && !recordIsSettled(item),
        )
      ) {
        durableHostStreams.get(record.tracker.hostId)?.abort.abort();
        durableHostStreams.delete(record.tracker.hostId);
      }
    },
    /** The host refused the batch by name: nothing is queued, every child
     * shows the host's own sentence, and the batch settles at once. */
    rejectDurableRecord(record: DurableGenerationRecoveryRecord, error: unknown): void {
      record.tracker = reduceGenerationLifecycle(record.tracker, {
        type: "admission_rejected",
        error: error instanceof Error ? error.message : String(error),
      });
      this.applyDurableRecord(record);
      persistDurableRecords();
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
        if (isDefiniteGenerationAdmissionRejection(error)) {
          this.rejectDurableRecord(record, error);
          return;
        }
        record.tracker = reduceGenerationLifecycle(record.tracker, {
          type: "admission_uncertain",
          error: error instanceof Error ? error.message : String(error),
        });
        persistDurableRecords();
      }

      // A lost POST response is never permission to submit a second time.
      // Recover by idempotency key, then (only when the server explicitly says
      // it is missing) repeat the same durable admission.
      try {
        const lookup = await lookupGenerationBatchByClientId(target, record.tracker.clientBatchId);
        if (lookup.kind === "found") {
          attach(lookup.batch);
          return;
        }
        attach(await admitGenerationBatch(target, body));
      } catch (retryError) {
        if (isDefiniteGenerationAdmissionRejection(retryError)) {
          // The second, identical POST was refused by name: nothing is queued
          // and the host's own sentence is the answer, not a silent wait.
          this.rejectDurableRecord(record, retryError);
          return;
        }
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
    ): { jobs: Job[]; admitted?: Promise<Job[]>; settled: Promise<Job[]> } {
      this.selectedClientId = null;
      if (chainRouting?.kind === "reject") throw new Error(chainRouting.reason);
      const size = Math.max(1, Math.floor(batchSize));
      const baseSeed = resolveBaseSeed(req.seed);
      const plans = planBatchRequests(req, size, baseSeed, requestOptions);
      // An unrouted submit means This device; resolve its own route so the
      // durable contract is read from the machine that will run the print.
      route = effectiveJobRoute(route, req.model || null);
      if (chainRouting?.kind !== "chain") {
        // Every print is admitted through the durable queue. A machine that
        // cannot carry this request is refused BY NAME with nothing queued —
        // there is no attached stream left to fall back to.
        const refusal = generationRouteRefusal(route);
        if (refusal !== null) throw new Error(refusal);
      }
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
      if (chainRouting?.kind !== "chain") {
        // `generationRouteRefusal` above already proved the machine and its
        // durable contract for every plan in this batch.
        const host = route!;
        const limit = canonicalGenerationBatchLimit(routeQueueCapabilities(host))!;
        const chunks = chunkGenerationBatchRequests(plans, limit).map(
          (requestChunk, chunkIndex) => {
            const offset = chunkIndex * limit;
            const jobChunk = jobs.slice(offset, offset + requestChunk.length);
            const clientBatchId = durableClientBatchId();
            const record: DurableGenerationRecoveryRecord = {
              tracker: createGenerationBatchTracker({
                hostId: host.hostId,
                expectedInstanceId: host.instanceId!,
                clientBatchId,
                submittedAtMs: Date.now(),
              }),
              hostLabel: host.label,
              hostKind: host.kind,
              mirrorRemoteOutput: host.mirrorRemoteOutput ?? true,
              children: requestChunk.map((plan, index) =>
                durableChildSummary(plan, index + 1, jobChunk[index]!.clientId),
              ),
              cancelRequestedChildIndexes: [],
              effectReceipts: [],
            };
            durableRecords.set(clientBatchId, record);
            durableJobIds.set(
              clientBatchId,
              new Map(jobChunk.map((job, index) => [index + 1, job.clientId])),
            );
            durableBatchJobs.set(
              clientBatchId,
              new Map(jobChunk.map((job, index) => [index + 1, job])),
            );
            const settlement = createDurableSettlement();
            durableSettlements.set(clientBatchId, settlement);
            return { record, requestChunk, settlement };
          },
        );
        // Prefer putting crash authority on disk before the first byte of the
        // first chunk POST leaves. If Web Storage rejects the write, retain
        // every UUID and instance fence in memory and continue through this durable path;
        // a client-side quota/privacy failure cannot veto valid host work or
        // redirect it into a second submission.
        persistDurableRecords();
        const admitted = Promise.all(
          chunks.map(({ record, requestChunk }) => this.admitDurableRecord(record, requestChunk)),
        ).then(() => jobs);
        const settled = Promise.all(chunks.map(({ settlement }) => settlement.promise)).then(() => {
          this.pendingConsumerBatchIds = this.pendingConsumerBatchIds.filter(
            (pendingBatchId) => pendingBatchId !== batchId,
          );
          const pendingBatches = new Set(this.pendingConsumerBatchIds);
          this.prune(
            GENERATION_HISTORY_LIMIT,
            jobs.map((job) => job.clientId),
            this.jobs.filter((job) => !pendingBatches.has(job.batchId)).map((job) => job.clientId),
          );
          return jobs;
        });
        return { jobs, admitted, settled };
      }
      const admissionResolvers: Array<() => void> = [];
      const admitted = Promise.all(
        jobs.map(
          () =>
            new Promise<void>((resolve) => {
              admissionResolvers.push(resolve);
            }),
        ),
      ).then(() => jobs);
      const tasks = jobs.map((job, i) => () => {
        // A sibling cancelled while it waited its turn never opens a stream.
        if (job.status === "error") {
          admissionResolvers[i]?.();
          return Promise.resolve();
        }
        return this.streamJob(job, plans[i]!, admissionResolvers[i]);
      });
      const submit = runWithConcurrency(tasks, MAX_STREAMS_PER_TARGET);
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
      return { jobs, admitted, settled };
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
            queueCapacity: hostStore.telemetry[group.jobs[0]?.hostId ?? ""]?.queueCapacity,
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
      if (durableRecord) {
        const childIndex = [...(durableJobIds.get(durableRecord.tracker.clientBatchId) ?? [])].find(
          ([, durableClientId]) => durableClientId === job.clientId,
        )?.[0];
        if (childIndex === undefined) {
          job.cancelling = false;
          return false;
        }
        if (!durableRecord.cancelRequestedChildIndexes.includes(childIndex)) {
          durableRecord.cancelRequestedChildIndexes.push(childIndex);
          // This write is the cancellation authority boundary: it precedes
          // the by-client reconciliation and any id-keyed DELETE.
          persistDurableRecords();
        }
        try {
          return await this.fulfillDurableCancelIntent(durableRecord, childIndex);
        } catch (error) {
          if (jobHasSettled(job)) return isCancelledError(job.error);
          // Keep the persisted intent and cancelling affordance. The same
          // one tap is retried by event/wake/reconnect reconciliation.
          throw error;
        }
      }
      // A sequence is a chain job and is cancelled on its own route; the
      // generation queue has no row for it. Its id comes from the create
      // response, not from a progress frame.
      const chainJobId = chainJobIds.get(job.clientId);
      if (chainJobId || job.id) {
        try {
          await apiFetchTo(
            targets.get(job.clientId) ?? currentTarget(),
            chainJobId
              ? `/api/chain-jobs/${encodeURIComponent(chainJobId)}/cancel`
              : `/api/queue/${encodeURIComponent(job.id!)}`,
            { method: chainJobId ? "POST" : "DELETE" },
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
      job.retainedByHost = false;
      return true;
    },
    /** Single generation — a batch of one. */
    async generate(req: GenerateRequest): Promise<Job> {
      const [job] = await this.generateBatch(req, 1);
      return job!;
    },
    /** Dismiss one terminal activity row without mutating server queue authority. */
    removeSettled(clientId: number): boolean {
      const index = this.jobs.findIndex((job) => job.clientId === clientId);
      const job = this.jobs[index];
      if (!job || !jobCanBeRemoved(job)) return false;
      const record = durableRecordForClientId(clientId);
      if (record) {
        const childIndex = [
          ...(durableJobIds.get(record.tracker.clientBatchId)?.entries() ?? []),
        ].find(([, mappedClientId]) => mappedClientId === clientId)?.[0];
        if (childIndex !== undefined) {
          const receipt = dismissalReceipt(childIndex);
          if (!record.effectReceipts.includes(receipt)) record.effectReceipts.push(receipt);
          persistDurableRecords();
        }
      }
      releaseSettledJob(job);
      this.jobs.splice(index, 1);
      if (this.selectedClientId === clientId) this.selectedClientId = null;
      if (record && recordIsSettled(record)) {
        const jobs = [...(durableBatchJobs.get(record.tracker.clientBatchId)?.values() ?? [])];
        void this.finishDurableBatchEffects(record, jobs);
      }
      return true;
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
      for (const clientId of drop) this.removeSettled(clientId);
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
      ownPreviews.stopAll();
      durableReconciles.clear();
      durableReconcilePending.clear();
      durableCancelAttempts.clear();
      durableJobIds.clear();
      durableBatchJobs.clear();
      durableHiddenJobIds.clear();
      durableSettlements.clear();
      durableRecords.clear();
      durableRecoveryStorageUnavailable = false;
      sharedDurableEventHosts.clear();
      durableRecoveryLoaded = false;
      targets.clear();
      chainRoutes.clear();
      chainJobIds.clear();
      this.pendingConsumerBatchIds = [];
      this.selectedClientId = null;
      this.jobs = [];
    },
    startJob(req: GenerateRequest): Job {
      // reactive() here is load-bearing: the SSE handlers below mutate the
      // returned reference from a closure. A raw object would update the
      // data without firing Vue's proxy traps — the canvas, edge code, and
      // job chips would sit frozen at "Queued 0/N" for the whole run.
      const job = createStoreJob(req, this.nextClientId++);
      this.jobs.push(job);
      return job;
    },
    /**
     * A finished sequence is one saved print. `finalized { output }` carries
     * the FILENAME the machine saved it under — never inline bytes — so the
     * media is fetched from that machine's gallery exactly as a durable
     * print's is, including the mirror to this Mac.
     */
    async completeChainJob(
      job: Job,
      request: GenerateRequest,
      filename: string,
      target?: ApiTarget,
    ): Promise<void> {
      if (jobHasSettled(job)) return;
      const format = requestFormat(filename, request.output_format ?? "mp4");
      job.result = {
        image: "",
        format,
        width: request.width,
        height: request.height,
        seed_used: request.seed ?? 0,
        generation_time_ms: 0,
        model: request.model,
        filename,
        metadata: null,
      };
      job.visualSeed = String(request.seed ?? 0);
      settleJob(job, "complete");
      if (job.previewUrl) {
        URL.revokeObjectURL(job.previewUrl);
        job.previewUrl = null;
      }
      void this.refreshRemoteResultUrl(job.clientId).catch(() => {
        // The reactive job carries the directed, user-visible error.
      });
      const originHostId = job.hostId ?? useHostsStore().primaryHost?.id ?? null;
      if (originHostId) void useGalleryStore().refreshHost(originHostId);
      if (
        !job.remote ||
        !job.mirrorRemoteOutput ||
        !target ||
        !(useAppPrefsStore().settings?.saveRemoteOutputs ?? true)
      ) {
        return;
      }
      try {
        const metadata = await originGalleryMetadata(target, filename);
        const bytes = await fetchGalleryMediaBytes(galleryMediaPath(filename, "host"), target);
        const buffer = Uint8Array.from(bytes).buffer;
        await ipc.saveOutputBytes(filename, await blobToBase64(new Blob([buffer])), metadata);
        void useGalleryStore().refreshHost("local");
      } catch (error) {
        console.warn("local save of remote sequence output failed:", error);
      }
    },
    async streamJob(
      job: Job,
      req: GenerateRequest,
      onAdmitted: () => void = () => {},
    ): Promise<void> {
      const abort = new AbortController();
      aborts.set(job.clientId, abort);
      const target = targets.get(job.clientId);
      const releaseStreamSlot = await streamSlots.acquire(
        target?.baseUrl ?? "__primary__",
        abort.signal,
      );
      if (!releaseStreamSlot || abort.signal.aborted || job.status === "error") {
        onAdmitted();
        releaseStreamSlot?.();
        aborts.delete(job.clientId);
        return;
      }

      let streamError: unknown = null;
      const chainRoute = chainRoutes.get(job.clientId);
      if (!chainRoute) {
        // Only a sequence opens a held stream. A print is admitted through the
        // durable queue by `submitBatch`, which never reaches this path.
        onAdmitted();
        if (!abort.signal.aborted && !jobHasSettled(job)) {
          settleJob(job, "error");
          job.interrupted = false;
          job.error = "internal: a print reached the sequence stream path.";
        }
        releaseStreamSlot();
        aborts.delete(job.clientId);
        return;
      }
      // A sequence is a durable chain job: created through
      // `POST /api/chain-jobs` with additive `ephemeral: true` — the machine
      // stitches it, records the print with stage seeds but no chain job id,
      // and deletes the job's artifacts — then followed on its OWN event
      // stream. `chainJobs.watch` is a singleton driving the sequence rail and
      // must not be taken over by an auto-chained one-shot.
      job.streamStarted = true;
      const body = { ...buildAutoChainRequest(req, chainRoute), ephemeral: true };
      // Freeze the exact host this job is created on. Submit already snapshots
      // the primary, but a job admitted while nothing was connected has no
      // route recorded — and the stream would then resolve the primary AGAIN,
      // later, possibly a different machine. Recovery and cancel key off this
      // map, and asking (or DELETEing on) a host that never ran the job is the
      // one outcome the frozen-route invariant exists to prevent.
      const streamTarget = target ?? connectedTarget();
      if (streamTarget && !targets.has(job.clientId)) {
        targets.set(job.clientId, streamTarget);
      }
      let chainJobId: string;
      try {
        const created = await apiFetchTo(
          streamTarget ?? currentTarget(),
          "/api/chain-jobs",
          chainJobInit(body, crypto.randomUUID()),
        );
        job.requestWarnings = requestWarningsFromHeaders(created.headers);
        ({ job_id: chainJobId } = (await created.json()) as CreateChainJobResponse);
      } catch (error) {
        onAdmitted();
        if (!abort.signal.aborted && !jobHasSettled(job)) {
          settleJob(job, "error");
          job.interrupted = false;
          job.error = describeTransportError(error, job.hostLabel);
        }
        releaseStreamSlot();
        aborts.delete(job.clientId);
        return;
      }
      onAdmitted();
      chainJobIds.set(job.clientId, chainJobId);
      // For a sequence the chain job IS its server identity: recovery looks it
      // up by this id, and the shim used to deliver the same value on its
      // synthesized first frame.
      job.id = chainJobId;
      let live = emptyChainJobLive();
      await sseStream(`/api/chain-jobs/${encodeURIComponent(chainJobId)}/events`, {
        signal: abort.signal,
        retry: true,
        ...(streamTarget ? { target: streamTarget } : {}),
        onEvent: (_event, data) => {
          // Abort/reset/cancel and terminal frames are final. Some SSE
          // implementations can still deliver already-buffered callbacks;
          // ignoring them prevents a cancelled job from being resurrected.
          if (abort.signal.aborted || jobHasSettled(job)) return;
          let event: ChainJobEvent;
          try {
            event = JSON.parse(data) as ChainJobEvent;
          } catch {
            // A malformed frame carries no authority; the terminal one settles.
            return;
          }
          const reduced = reduceChainJobFrame(live, event);
          live = reduced.live;
          for (const frame of reduced.progress) {
            applyChainProgress(job, frame as ChainProgressEvent);
          }
          const output = reduced.finalized?.output;
          if (output) {
            void this.completeChainJob(job, req, output, streamTarget ?? undefined);
            abort.abort();
            return;
          }
          const terminal = reduced.terminal;
          if (!terminal || terminal.state === "completed") return;
          settleJob(job, "error");
          job.error =
            terminal.state === "cancelled"
              ? "Cancelled"
              : (terminal.error ?? "The sequence failed on the host.");
          abort.abort();
        },
        onClose: (err) => {
          if (err && !abort.signal.aborted && !jobHasSettled(job)) {
            settleJob(job, "error");
            job.error = err.message;
            // fetch-event-source reports network/transport loss as TypeError.
            // HTTP/auth failures are deterministic and must remain final.
            job.interrupted = err instanceof TypeError;
          }
        },
      }).catch((error: unknown) => {
        streamError = error;
      });
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

/** Automatic-chain routing snapshot for endpoint selection and cancellation. */
const chainRoutes = new Map<number, AutoChainRoutingDecision>();
/** The durable chain job an auto-chained sequence became. Cancel routes on
 * this: a chain is cancelled through its own route, never the queue. */
const chainJobIds = new Map<number, string>();

const chainJobInit = (body: unknown, operationId: string): RequestInit => ({
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-mold-operation-id": operationId,
  },
  body: JSON.stringify(body),
});

import { reactive } from "vue";
import { defineStore } from "pinia";
import { apiFetchTo, currentTarget, ApiError, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { evictMedia, galleryMediaPath, streamableMediaUrl } from "../lib/gallery/media";
import { ipc } from "../lib/ipc";
import { notifyGenerated, notifyGenerationFailed } from "../lib/notify";
import { describeTransportError } from "../lib/api/errors";
import { useAppPrefsStore } from "./appPrefs";
import { useGalleryStore } from "./gallery";
import { useHostsStore } from "./hosts";
import type {
  ChainProgressEvent,
  CompleteEvent,
  GenerateRequest,
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
  applyProgress,
  base64ToBlobUrl,
  chainCompleteToComplete,
  isCancelledError,
  markJobSettled,
  metadataOnlyResult,
  newJob,
  type Job,
} from "../lib/generationJob";

export {
  applyChainProgress,
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
  /** Durable identity shared by prepared siblings and retained in gallery metadata. */
  batchId?: string;
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
  return Array.from({ length: size }, (_, i) => ({
    ...req,
    ...(options.prompts !== undefined ? { prompt: options.prompts[i]! } : {}),
    ...(options.originalPrompt !== undefined ? { original_prompt: options.originalPrompt } : {}),
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
 * gallery reads and queue cancellation. Keep two slots per host globally —
 * not merely per batch — so repeated Generate taps cannot starve those other
 * requests. Later jobs remain visible as locally queued until a slot opens.
 */
const MAX_STREAMS_PER_TARGET = 2;

interface StreamSlotWaiter {
  signal: AbortSignal;
  resolve: (release: (() => void) | null) => void;
  onAbort?: () => void;
}

interface StreamSlotPool {
  active: number;
  waiters: StreamSlotWaiter[];
}

const streamSlotPools = new Map<string, StreamSlotPool>();

function cleanStreamSlotPool(key: string, pool: StreamSlotPool): void {
  if (pool.active === 0 && pool.waiters.length === 0) streamSlotPools.delete(key);
}

function drainStreamSlotPool(key: string, pool: StreamSlotPool): void {
  while (pool.active < MAX_STREAMS_PER_TARGET && pool.waiters.length > 0) {
    const waiter = pool.waiters.shift()!;
    if (waiter.onAbort) waiter.signal.removeEventListener("abort", waiter.onAbort);
    if (waiter.signal.aborted) {
      waiter.resolve(null);
      continue;
    }

    pool.active += 1;
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      waiter.signal.removeEventListener("abort", release);
      pool.active -= 1;
      drainStreamSlotPool(key, pool);
      cleanStreamSlotPool(key, pool);
    };
    waiter.signal.addEventListener("abort", release, { once: true });
    waiter.resolve(release);
    if (waiter.signal.aborted) release();
  }
  cleanStreamSlotPool(key, pool);
}

function acquireStreamSlot(key: string, signal: AbortSignal): Promise<(() => void) | null> {
  if (signal.aborted) return Promise.resolve(null);
  let pool = streamSlotPools.get(key);
  if (!pool) {
    pool = { active: 0, waiters: [] };
    streamSlotPools.set(key, pool);
  }
  return new Promise((resolve) => {
    const waiter: StreamSlotWaiter = { signal, resolve };
    waiter.onAbort = () => {
      const index = pool.waiters.indexOf(waiter);
      if (index >= 0) pool.waiters.splice(index, 1);
      signal.removeEventListener("abort", waiter.onAbort!);
      resolve(null);
      cleanStreamSlotPool(key, pool);
    };
    signal.addEventListener("abort", waiter.onAbort, { once: true });
    pool.waiters.push(waiter);
    drainStreamSlotPool(key, pool);
  });
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
      const tasks = jobs.map((job, i) => () => {
        // A sibling cancelled while it waited its turn never opens a stream.
        if (job.status === "error") return Promise.resolve();
        return this.streamJob(job, plans[i]!);
      });
      const settled = runWithConcurrency(tasks, 2).then(() => {
        // Background notification (the view toasts in the foreground).
        const failed = jobs.find((s) => s.status === "error");
        if (jobs.some((s) => s.status === "complete")) notifyGenerated(req.prompt);
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
            12,
            jobs.map((job) => job.clientId),
            this.jobs.filter((job) => !pendingBatches.has(job.batchId)).map((job) => job.clientId),
          );
        }, 0);
        return jobs;
      });
      return { jobs, settled };
    },
    /** Submit and wait for the whole batch (menu Generate, tests). */
    async generateBatch(req: GenerateRequest, batchSize: number): Promise<Job[]> {
      return this.submitBatch(req, batchSize).settled;
    },
    /**
     * Cancel one job (default: the canvas job). Ordinary queued jobs leave
     * the server queue via DELETE /api/queue/:id; automatic chains use their
     * durable shim id with POST /api/chain-jobs/:id/cancel. The stream is
     * aborted either way and the job is marked cancelled.
     */
    async cancel(clientId?: number): Promise<void> {
      const job =
        clientId !== undefined
          ? (this.jobs.find((j) => j.clientId === clientId) ?? null)
          : this.active;
      if (!job || job.status === "complete" || job.status === "error") return;
      let cancellationError: unknown = null;
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
          if (!(err instanceof ApiError && (err.status === 409 || err.status === 404))) {
            cancellationError = err;
          }
        }
      } else if (job.streamStarted) {
        cancellationError = new Error(
          "Remote cancellation was not confirmed before the queue ID arrived.",
        );
      }
      // A terminal SSE frame may win while DELETE is in flight. Preserve that
      // authoritative outcome, even if the DELETE request itself then fails.
      if (jobHasSettled(job)) return;
      aborts.get(job.clientId)?.abort();
      aborts.delete(job.clientId);
      settleJob(job, "error");
      if (cancellationError) {
        // Release the local stream permit even when the host did not confirm
        // cancellation. The server may still finish the queued work.
        job.error = "Cancelled locally; remote cancellation was not confirmed.";
        throw cancellationError;
      }
      job.error = "Cancelled";
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
        chainRoutes.delete(job.clientId);
      }
      this.jobs = this.jobs.filter((j) => !drop.has(j.clientId));
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
      targets.clear();
      chainRoutes.clear();
      this.pendingConsumerBatchIds = [];
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
      const releaseStreamSlot = await acquireStreamSlot(
        target?.baseUrl ?? "__primary__",
        abort.signal,
      );
      if (!releaseStreamSlot || abort.signal.aborted || job.status === "error") {
        releaseStreamSlot?.();
        aborts.delete(job.clientId);
        return;
      }

      let streamError: unknown = null;
      job.streamStarted = true;
      const chainRoute = chainRoutes.get(job.clientId);
      const path = chainRoute ? "/api/generate/chain/stream" : "/api/generate/stream";
      const body = chainRoute ? buildAutoChainRequest(req, chainRoute) : req;
      await sseStream(path, {
        method: "POST",
        body,
        signal: abort.signal,
        retry: false,
        ...(job.metadataOnlyCompletion
          ? { headers: { "X-Mold-SSE-Payload": "metadata-only" } }
          : {}),
        ...(target ? { target } : {}),
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
                const parsed = JSON.parse(data) as { error?: string; message?: string };
                const message = parsed.error ?? parsed.message ?? data;
                current.error = isCancelledError(message) ? "Cancelled" : message;
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

/** Automatic-chain routing snapshot for endpoint selection and cancellation. */
const chainRoutes = new Map<number, AutoChainRoutingDecision>();

import { reactive } from "vue";
import { defineStore } from "pinia";
import { apiFetchTo, currentTarget, ApiError, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { ipc } from "../lib/ipc";
import { notifyGenerated, notifyGenerationFailed } from "../lib/notify";
import { useAppPrefsStore } from "./appPrefs";
import { useConnectionStore } from "./connection";
import type { CompleteEvent, GenerateRequest, ProgressEvent } from "../lib/api/types";
import type { DevelopPhase } from "../lib/develop/grain";

/** Where a batch runs — mirrors `HostRoute` from the hosts store. */
export interface JobRoute {
  hostId: string;
  label: string;
  kind: "local" | "remote";
  target: ApiTarget;
}

/** Whether the primary connection points at a remote host. */
function primaryIsRemote(): boolean {
  return useConnectionStore().mode === "remote";
}

/** Filesystem-safe local filename for a saved output. */
export function suggestOutputFilename(
  model: string,
  seed: number,
  format: string,
  nowMs: number = Date.now(),
): string {
  const slug = model
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `mold-${slug}-${seed}-${nowMs}.${format}`;
}

export type JobStatus = "queued" | "loading" | "denoising" | "finishing" | "complete" | "error";

export interface Job {
  /** Client-side identity — stable across the job's life; keys cancel/menus. */
  clientId: number;
  /** Groups sibling jobs submitted together as one batch. */
  batchId: number;
  /** Server-assigned id from the Queued event (empty until it arrives). */
  id: string;
  prompt: string;
  model: string;
  width: number;
  height: number;
  /** Seed driving the Develop grain — requested seed or a stand-in until seed_used arrives. */
  visualSeed: string;
  status: JobStatus;
  queuePosition: number | null;
  step: number;
  total: number;
  stage: string | null;
  error: string | null;
  /** Object URL of the decoded result. */
  resultUrl: string | null;
  /** Object URL of the latest live latent preview (small PNG, upscaled by CSS). */
  previewUrl: string | null;
  result: CompleteEvent | null;
  /** Host this job queued on; null = the primary connection. */
  hostId: string | null;
  hostLabel: string | null;
  /** True when the job runs on a remote host (drives the auto local-save). */
  remote: boolean;
}

export function newJob(req: GenerateRequest): Job {
  return {
    clientId: 0,
    batchId: 0,
    id: "",
    prompt: req.prompt,
    model: req.model,
    width: req.width,
    height: req.height,
    visualSeed: req.seed !== undefined ? String(req.seed) : `${req.model}·${req.prompt}`,
    status: "queued",
    queuePosition: null,
    step: 0,
    total: req.steps,
    stage: null,
    error: null,
    resultUrl: null,
    previewUrl: null,
    result: null,
    hostId: null,
    hostLabel: null,
    remote: false,
  };
}

/** Pure SSE reducer — exported for tests. Mutates and returns the job. */
export function applyProgress(job: Job, event: ProgressEvent): Job {
  switch (event.type) {
    case "queued":
      job.status = "queued";
      job.queuePosition = event.position;
      if (event.id) job.id = event.id;
      break;
    case "weight_load":
    case "stage_start":
      // Stages after the denoise loop (transformer drop, VAE decode, encode)
      // are the fixer bath: the steps read N/N but the print isn't done.
      if (job.status === "denoising" || job.status === "finishing") {
        if (event.type === "stage_start") {
          job.status = "finishing";
          job.stage = event.name;
        }
      } else {
        job.status = "loading";
        job.stage = event.type === "stage_start" ? event.name : "Loading weights";
      }
      break;
    case "denoise_step":
      job.status = "denoising";
      job.queuePosition = null;
      job.step = event.step;
      job.total = event.total;
      break;
    case "preview": {
      job.status = "denoising";
      job.queuePosition = null;
      const previous = job.previewUrl;
      job.previewUrl = base64ToBlobUrl(event.image, "image/png");
      if (previous) URL.revokeObjectURL(previous);
      break;
    }
    default:
      break;
  }
  return job;
}

export function jobPhase(job: Job): DevelopPhase {
  switch (job.status) {
    case "complete":
      return "fixed";
    case "error":
      return "stopped";
    case "denoising":
    case "finishing":
      return "developing";
    default:
      return "latent";
  }
}

export function jobProgress(job: Job): number {
  if (job.status === "complete" || job.status === "finishing") return 1;
  if (job.total <= 0) return 0;
  return job.step / job.total;
}

export function base64ToBlobUrl(b64: string, mime: string): string {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

/** Random 32-bit seed — small enough to stay an exact integer after `+ i`. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 0xffffffff);
}

/**
 * Resolve the base seed for a batch: an explicit finite seed is honored,
 * otherwise a fresh random base is drawn so the run is reproducible from the
 * first sibling. Pure given `rng`.
 */
export function resolveBaseSeed(seed: number | undefined, rng: () => number = randomSeed): number {
  return seed !== undefined && Number.isFinite(seed) ? seed : rng();
}

/**
 * Expand one request into `batchSize` sibling requests with seeds
 * `baseSeed + i`, each forced to `batch_size: 1` (the client drives the
 * sequence, one job per server call). Pure — the seed decision lives here so
 * it can be tested without the store or the network.
 */
export function planBatchRequests(
  req: GenerateRequest,
  batchSize: number,
  baseSeed: number,
): GenerateRequest[] {
  const size = Math.max(1, Math.floor(batchSize));
  return Array.from({ length: size }, (_, i) => ({
    ...req,
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

export const useGenerationStore = defineStore("generation", {
  state: () => ({
    /**
     * Every job of this session, submission order. The server queue is the
     * scheduler — each job holds its own SSE stream, so submitting while
     * another develops simply queues behind it (each job snapshots its own
     * model + params at submit time).
     */
    jobs: [] as Job[],
    nextClientId: 1,
    nextBatchId: 1,
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
    ): { jobs: Job[]; settled: Promise<Job[]> } {
      const size = Math.max(1, Math.floor(batchSize));
      const baseSeed = resolveBaseSeed(req.seed);
      const plans = planBatchRequests(req, size, baseSeed);
      const batchId = this.nextBatchId++;
      const jobs = plans.map((plan) => {
        const job = this.startJob(plan);
        job.batchId = batchId;
        if (route) {
          job.hostId = route.hostId;
          job.hostLabel = route.label;
          job.remote = route.kind === "remote";
          targets.set(job.clientId, route.target);
        } else {
          // Unrouted = the primary connection, which may itself be remote
          // (single-host remote mode) — those prints get saved locally too.
          job.remote = primaryIsRemote();
        }
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
        else if (failed?.error && failed.error !== "Cancelled") {
          notifyGenerationFailed(failed.error);
        }
        this.prune();
        return jobs;
      });
      return { jobs, settled };
    },
    /** Submit and wait for the whole batch (menu Generate, tests). */
    async generateBatch(req: GenerateRequest, batchSize: number): Promise<Job[]> {
      return this.submitBatch(req, batchSize).settled;
    },
    /**
     * Cancel one job (default: the canvas job). Queued jobs leave the server
     * queue via DELETE /api/queue/:id (409 = already running: the server
     * finishes the compute; we stop listening); the stream is aborted either
     * way and the job is marked cancelled.
     */
    async cancel(clientId?: number): Promise<void> {
      const job =
        clientId !== undefined
          ? (this.jobs.find((j) => j.clientId === clientId) ?? null)
          : this.active;
      if (!job || job.status === "complete" || job.status === "error") return;
      if (job.id) {
        try {
          await apiFetchTo(
            targets.get(job.clientId) ?? currentTarget(),
            `/api/queue/${encodeURIComponent(job.id)}`,
            { method: "DELETE" },
          );
        } catch (err) {
          if (!(err instanceof ApiError && (err.status === 409 || err.status === 404))) throw err;
        }
      }
      aborts.get(job.clientId)?.abort();
      aborts.delete(job.clientId);
      // The await above may have let the stream finish; only stamp live jobs.
      const status = job.status as JobStatus;
      if (status !== "complete" && status !== "error") {
        job.status = "error";
        job.error = "Cancelled";
      }
    },
    /** Single generation — a batch of one. */
    async generate(req: GenerateRequest): Promise<Job> {
      const [job] = await this.generateBatch(req, 1);
      return job!;
    },
    /** Drop finished jobs beyond the most recent few, releasing their URLs. */
    prune(keep = 12) {
      const finished = this.jobs.filter((j) => j.status === "complete" || j.status === "error");
      const excess = finished.length - keep;
      if (excess <= 0) return;
      const drop = new Set(finished.slice(0, excess).map((j) => j.clientId));
      for (const job of this.jobs) {
        if (!drop.has(job.clientId)) continue;
        if (job.resultUrl) URL.revokeObjectURL(job.resultUrl);
        if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
        targets.delete(job.clientId);
      }
      this.jobs = this.jobs.filter((j) => !drop.has(j.clientId));
    },
    /** Revoke every held object URL and clear all jobs (teardown/tests). */
    resetJobs() {
      for (const job of this.jobs) {
        aborts.get(job.clientId)?.abort();
        if (job.resultUrl) URL.revokeObjectURL(job.resultUrl);
        if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      }
      aborts.clear();
      targets.clear();
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
      await sseStream("/api/generate/stream", {
        method: "POST",
        body: req,
        signal: abort.signal,
        retry: false,
        ...(target ? { target } : {}),
        onEvent: (event, data) => {
          const current = job;
          try {
            if (event === "progress") {
              applyProgress(current, JSON.parse(data) as ProgressEvent);
            } else if (event === "complete") {
              const complete = JSON.parse(data) as CompleteEvent;
              current.result = complete;
              current.resultUrl = base64ToBlobUrl(
                complete.image,
                MIME[complete.format] ?? "application/octet-stream",
              );
              current.visualSeed = String(complete.seed_used);
              current.status = "complete";
              if (current.previewUrl) {
                URL.revokeObjectURL(current.previewUrl);
                current.previewUrl = null;
              }
              // Remote prints also land in this Mac's gallery (pref-gated):
              // the SSE payload is the encoded output file, metadata included,
              // so no extra download is needed.
              if (current.remote && (useAppPrefsStore().settings?.saveRemoteOutputs ?? true)) {
                const filename = suggestOutputFilename(
                  complete.model,
                  complete.seed_used,
                  complete.format,
                );
                ipc.saveOutputBytes(filename, complete.image).catch((err) => {
                  console.warn("local save of remote output failed:", err);
                });
              }
            } else if (event === "error") {
              current.status = "error";
              try {
                const parsed = JSON.parse(data) as { error?: string; message?: string };
                current.error = parsed.error ?? parsed.message ?? data;
              } catch {
                current.error = data;
              }
            }
          } catch {
            /* skip malformed frame */
          }
        },
        onClose: (err) => {
          if (err && job.status !== "complete" && job.status !== "error") {
            job.status = "error";
            job.error = err.message;
          }
        },
      });
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

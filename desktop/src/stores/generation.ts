import { defineStore } from "pinia";
import { sseStream } from "../lib/api/sse";
import type { CompleteEvent, GenerateRequest, ProgressEvent } from "../lib/api/types";
import type { DevelopPhase } from "../lib/develop/grain";

export type JobStatus = "queued" | "loading" | "denoising" | "complete" | "error";

export interface Job {
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
  result: CompleteEvent | null;
}

export function newJob(req: GenerateRequest): Job {
  return {
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
    result: null,
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
      if (job.status !== "denoising") job.status = "loading";
      job.stage = event.type === "stage_start" ? event.name : "Loading weights";
      break;
    case "denoise_step":
      job.status = "denoising";
      job.queuePosition = null;
      job.step = event.step;
      job.total = event.total;
      break;
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
      return "developing";
    default:
      return "latent";
  }
}

export function jobProgress(job: Job): number {
  if (job.status === "complete") return 1;
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
    /** The job shown in the Generate canvas — the most recent sibling. */
    active: null as Job | null,
    /** Every job in the current batch (length 1 for a single generation). */
    siblings: [] as Job[],
    abort: null as AbortController | null,
  }),
  actions: {
    /**
     * Run a batch of `batchSize` generations sequentially (single GPU, one job
     * at a time). Seeds are `base + i`; the canvas tracks the most recent
     * sibling. `batchSize <= 1` is exactly the old single-flight path.
     */
    async generateBatch(req: GenerateRequest, batchSize: number): Promise<Job[]> {
      const size = Math.max(1, Math.floor(batchSize));
      const baseSeed = resolveBaseSeed(req.seed);
      const plans = planBatchRequests(req, size, baseSeed);
      this.resetJobs();
      for (const plan of plans) {
        const job = this.startJob(plan);
        this.siblings.push(job);
        await this.streamJob(job, plan);
      }
      return this.siblings;
    },
    /** Single generation — a batch of one. */
    async generate(req: GenerateRequest): Promise<Job> {
      await this.generateBatch(req, 1);
      return this.active!;
    },
    /** Revoke every held object URL and clear the batch. */
    resetJobs() {
      this.cancelStream();
      if (this.active?.resultUrl) URL.revokeObjectURL(this.active.resultUrl);
      for (const s of this.siblings) {
        if (s.resultUrl && s.resultUrl !== this.active?.resultUrl) {
          URL.revokeObjectURL(s.resultUrl);
        }
      }
      this.siblings = [];
      this.active = null;
    },
    startJob(req: GenerateRequest): Job {
      const job = newJob(req);
      this.active = job;
      return job;
    },
    async streamJob(job: Job, req: GenerateRequest): Promise<void> {
      const abort = new AbortController();
      this.abort = abort;
      await sseStream("/api/generate/stream", {
        method: "POST",
        body: req,
        signal: abort.signal,
        retry: false,
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
    },
    cancelStream() {
      this.abort?.abort();
      this.abort = null;
    },
  },
});

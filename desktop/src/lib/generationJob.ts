import type {
  ChainProgressEvent,
  CompleteEvent,
  GenerateRequest,
  ProgressEvent,
  SseChainCompleteEvent,
} from "./api/types";
import type { DevelopPhase } from "@ui/lib/grain";

export type JobStatus = "queued" | "loading" | "denoising" | "finishing" | "complete" | "error";

/**
 * One client-owned generation stream. Desktop and mobile both keep every
 * submitted request open while the engine schedules it in the server queue.
 */
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
  /** Guidance the job was submitted with — reuse must not rewrite it. */
  guidance: number;
  /** Seed driving the Develop grain — requested seed or a stand-in until seed_used arrives. */
  visualSeed: string;
  status: JobStatus;
  queuePosition: number | null;
  step: number;
  total: number;
  stage: string | null;
  /** Current zero-based clip when this job uses automatic chain routing. */
  chainStageIndex: number | null;
  /** Total clips reported by the chain stream. */
  chainStageCount: number | null;
  error: string | null;
  /** Structured transport-close marker used by iOS resume reconciliation. */
  interrupted: boolean;
  /** Client submission time used to avoid joining a later fixed-seed duplicate.
   *  Also the print's `createdAtMs` in the shared activity merge — a real wall
   *  clock, never the clientId counter. */
  submittedAtUnixMs: number;
  /** Wall clock at the first terminal status; null while the job is in flight.
   *  The Create strip's attention rows expire against this. */
  settledAtMs: number | null;
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
  /** Whether a remote result should also be copied into this device's local gallery. */
  mirrorRemoteOutput: boolean;
  /** Whether to retain encoded output bytes after creating the renderable Blob URL. */
  retainEncodedResult: boolean;
  /** Ask the host for filename-backed completion metadata instead of media bytes. */
  metadataOnlyCompletion: boolean;
  /** True once this job has started its HTTP request to the host. */
  streamStarted: boolean;
  /** True when resultUrl must be released with URL.revokeObjectURL. */
  resultUrlIsObjectUrl: boolean;
  /** Expiry of a ticketed result URL in milliseconds since epoch; null = no expiry. */
  resultUrlExpiresAt: number | null;
  /** True while a filename-backed result URL is being acquired or renewed. */
  resultUrlLoading: boolean;
  /** A generation can succeed even when its saved result URL cannot be loaded. */
  resultError: string | null;
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
    guidance: req.guidance ?? 1.0,
    visualSeed: req.seed !== undefined ? String(req.seed) : `${req.model}·${req.prompt}`,
    status: "queued",
    queuePosition: null,
    step: 0,
    total: req.steps,
    stage: null,
    chainStageIndex: null,
    chainStageCount: null,
    error: null,
    interrupted: false,
    submittedAtUnixMs: Date.now(),
    settledAtMs: null,
    resultUrl: null,
    previewUrl: null,
    result: null,
    hostId: null,
    hostLabel: null,
    remote: false,
    mirrorRemoteOutput: true,
    retainEncodedResult: true,
    metadataOnlyCompletion: false,
    streamStarted: false,
    resultUrlIsObjectUrl: false,
    resultUrlExpiresAt: null,
    resultUrlLoading: false,
    resultError: null,
  };
}

/**
 * Stamp the wall clock the first time a job reaches a terminal status. Idempotent
 * on purpose: a late transport frame must not restart the attention-row clock.
 */
export function markJobSettled(job: Job): void {
  if (job.settledAtMs !== null) return;
  if (job.status === "complete" || job.status === "error") job.settledAtMs = Date.now();
}

/** Pure SSE reducer shared by desktop and mobile. Mutates and returns the job. */
export function applyProgress(job: Job, event: ProgressEvent): Job {
  switch (event.type) {
    case "queued":
      job.status = "queued";
      job.queuePosition = event.position;
      if (event.id) job.id = event.id;
      break;
    case "dependency_wait":
      job.status = "queued";
      job.stage = `Waiting for ${event.dependency}`;
      break;
    case "download_progress": {
      job.status = "queued";
      const percent =
        event.bytes_total > 0 ? Math.round((event.bytes_downloaded / event.bytes_total) * 100) : 0;
      job.stage = `Downloading ${event.filename} (${percent}%)`;
      break;
    }
    case "download_done":
      job.status = "queued";
      job.stage = `Dependency ready: ${event.filename}`;
      break;
    case "pull_complete":
      job.status = "queued";
      job.stage = `Dependency ready: ${event.model}`;
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

/** Apply one chain-specific SSE progress frame to the shared Job shape. */
export function applyChainProgress(job: Job, event: ChainProgressEvent): Job {
  if (event.job_id) job.id = event.job_id;
  job.queuePosition = null;

  switch (event.type) {
    case "chain_start":
      job.status = "loading";
      job.chainStageIndex = null;
      job.chainStageCount = event.stage_count;
      job.step = 0;
      job.stage = `Preparing ${event.stage_count} clips`;
      break;
    case "stage_start": {
      job.status = "loading";
      job.chainStageIndex = event.stage_idx;
      const count = job.chainStageCount;
      job.stage = count ? `Clip ${event.stage_idx + 1} of ${count}` : `Clip ${event.stage_idx + 1}`;
      break;
    }
    case "denoise_step": {
      job.status = "denoising";
      job.chainStageIndex = event.stage_idx;
      const count = job.chainStageCount ?? event.stage_idx + 1;
      job.step = event.stage_idx * event.total + event.step;
      job.total = count * event.total;
      job.stage = `Clip ${event.stage_idx + 1} of ${count}`;
      break;
    }
    case "stage_done":
      job.status = "loading";
      job.chainStageIndex = event.stage_idx;
      if (job.chainStageCount) {
        job.stage = `Clip ${event.stage_idx + 1} of ${job.chainStageCount} complete`;
      }
      break;
    case "stitching":
      job.status = "finishing";
      job.step = job.total;
      job.stage = `Stitching ${event.total_frames} frames`;
      break;
  }
  return job;
}

/** Adapt a chain completion to the result shape consumed by every Job UI. */
export function chainCompleteToComplete(
  event: SseChainCompleteEvent,
  req: GenerateRequest,
): CompleteEvent {
  return {
    image: event.video,
    format: event.format,
    width: event.width,
    height: event.height,
    original_image: null,
    original_width: null,
    original_height: null,
    seed_used: event.metadata?.seed ?? req.seed ?? 0,
    generation_time_ms: event.generation_time_ms ?? 0,
    model: event.metadata?.model ?? req.model,
    video_frames: event.frames,
    video_fps: event.fps,
    video_thumbnail: event.thumbnail ?? null,
    video_gif_preview: event.gif_preview ?? null,
    video_has_audio: event.has_audio ?? false,
    video_duration_ms: event.duration_ms ?? null,
    video_audio_sample_rate: event.audio_sample_rate ?? null,
    video_audio_channels: event.audio_channels ?? null,
    gpu: event.gpu ?? null,
    filename: event.filename ?? null,
    original_filename: null,
    metadata: event.metadata ?? null,
  };
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

/** Normalize cancellation emitted by the server or initiated by this client. */
export function isCancelledError(error: string | null | undefined): boolean {
  return error != null && /\bcancell?ed\b/i.test(error);
}

/** Keep result metadata while dropping every encoded media field on memory-constrained clients. */
export function metadataOnlyResult(result: CompleteEvent): CompleteEvent {
  return {
    image: "",
    format: result.format,
    width: result.width,
    height: result.height,
    original_image: null,
    original_width: result.original_width ?? null,
    original_height: result.original_height ?? null,
    seed_used: result.seed_used,
    generation_time_ms: result.generation_time_ms,
    model: result.model,
    video_frames: result.video_frames ?? null,
    video_fps: result.video_fps ?? null,
    video_thumbnail: null,
    video_gif_preview: null,
    video_has_audio: result.video_has_audio ?? false,
    video_duration_ms: result.video_duration_ms ?? null,
    video_audio_sample_rate: result.video_audio_sample_rate ?? null,
    video_audio_channels: result.video_audio_channels ?? null,
    gpu: result.gpu ?? null,
    filename: result.filename ?? null,
    original_filename: result.original_filename ?? null,
    metadata: result.metadata ?? null,
  };
}

/** Compact, plain-language status shared by every jobs surface. */
export function jobStatusCode(job: Job): string {
  switch (job.status) {
    case "denoising":
      return `${job.step}/${job.total}`;
    case "finishing":
      return "FINALIZING";
    case "loading":
      return "LOADING";
    case "queued":
      return job.queuePosition && job.queuePosition > 0 ? `QUEUED #${job.queuePosition}` : "QUEUED";
    case "complete":
      return "DONE";
    case "error":
      return isCancelledError(job.error) ? "CANCELLED" : "FAILED";
  }
  return "UNKNOWN";
}

export function base64ToBlobUrl(b64: string, mime: string): string {
  const bytes = Uint8Array.from(atob(b64), (character) => character.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

import { reactive, ref, type Ref } from "vue";
import { generateChainStream, generateStream } from "../api";
import type {
  ChainProgressEvent,
  ChainRequestWire,
  GenerateRequestWire,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
} from "../types";
import type { ChainRoutingDecision } from "../lib/chainRouting";

export interface JobProgress {
  stage: string;
  step: number | null;
  totalSteps: number | null;
  weightBytesLoaded: number | null;
  weightBytesTotal: number | null;
  queuePosition: number | null;
  gpu: number | null;
  elapsedMs: number | null;
}

export interface Job {
  id: string;
  request: GenerateRequestWire;
  startedAt: number;
  controller: AbortController;
  progress: JobProgress;
  result: SseCompleteEvent | null;
  error: string | null;
  state: "running" | "done" | "error" | "canceled";
  /** When the job was auto-promoted to the chain endpoint. `null` for a
   * normal single-clip submission. */
  chain: ChainJobMeta | null;
}

export interface ChainJobMeta {
  stageCount: number;
  currentStage: number;
  estimatedTotalFrames: number | null;
}

function emptyProgress(): JobProgress {
  return {
    stage: "Starting",
    step: null,
    totalSteps: null,
    weightBytesLoaded: null,
    weightBytesTotal: null,
    queuePosition: null,
    gpu: null,
    elapsedMs: null,
  };
}

function applyProgress(job: Job, evt: SseProgressEvent) {
  const p = job.progress;
  switch (evt.type) {
    case "stage_start":
      p.stage = evt.name;
      break;
    case "stage_done":
      p.stage = `${evt.name} (done)`;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "info":
      p.stage = evt.message;
      break;
    case "denoise_step":
      p.stage = "Denoising";
      p.step = evt.step;
      p.totalSteps = evt.total;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "queued":
      p.stage = `Queued (position ${evt.position})`;
      p.queuePosition = evt.position;
      break;
    case "weight_load":
      p.stage = `Loading ${evt.component}`;
      p.weightBytesLoaded = evt.bytes_loaded;
      p.weightBytesTotal = evt.bytes_total;
      break;
    case "cache_hit":
      p.stage = `Cache hit: ${evt.resource}`;
      break;
  }
}

/** Chain progress events come from a separate SSE stream shape than the
 * single-clip path; we fold them into the same `JobProgress` so the
 * `RunningJobCard` UI renders a familiar "Denoising clip K/N · step X/Y"
 * readout without the per-event UI layer needing to know about chaining. */
function applyChainProgress(job: Job, evt: ChainProgressEvent) {
  const p = job.progress;
  const meta = job.chain;
  switch (evt.type) {
    case "chain_start":
      if (meta) {
        meta.stageCount = evt.stage_count;
        meta.estimatedTotalFrames = evt.estimated_total_frames;
      }
      p.stage = `Chain · ${evt.stage_count} clips · ~${evt.estimated_total_frames} frames`;
      break;
    case "stage_start":
      if (meta) meta.currentStage = evt.stage_idx;
      p.stage = chainStageLabel(meta, evt.stage_idx, "Starting");
      p.step = null;
      p.totalSteps = null;
      break;
    case "denoise_step":
      if (meta) meta.currentStage = evt.stage_idx;
      p.stage = chainStageLabel(meta, evt.stage_idx, "Denoising");
      p.step = evt.step;
      p.totalSteps = evt.total;
      break;
    case "stage_done":
      p.stage = chainStageLabel(meta, evt.stage_idx, "Done");
      p.step = null;
      p.totalSteps = null;
      break;
    case "stitching":
      p.stage = `Stitching ${evt.total_frames} frames…`;
      p.step = null;
      p.totalSteps = null;
      break;
  }
}

function chainStageLabel(
  meta: ChainJobMeta | null,
  stageIdx: number,
  action: string,
): string {
  const total = meta?.stageCount ?? null;
  const human = stageIdx + 1;
  return total !== null
    ? `${action} clip ${human}/${total}`
    : `${action} clip ${human}`;
}

/** Chain complete events carry a `video` payload instead of `image`, no
 * single seed, and separate thumbnail/gif_preview fields. Shape-shift into
 * `SseCompleteEvent` so `GeneratePage.openJob` + `RunningJobCard` stay
 * unchanged. `seed_used` falls back to the request seed (or 0) — the
 * gallery match will miss but the refresh-on-complete still surfaces the
 * new item. */
function chainCompleteToSingle(
  req: GenerateRequestWire,
  evt: SseChainCompleteEvent,
): SseCompleteEvent {
  return {
    image: evt.video,
    format: evt.format,
    width: evt.width,
    height: evt.height,
    seed_used: req.seed ?? 0,
    generation_time_ms: evt.generation_time_ms ?? 0,
    model: req.model,
    video_frames: evt.frames,
    video_fps: evt.fps,
    video_thumbnail: evt.thumbnail ?? null,
    video_gif_preview: evt.gif_preview ?? null,
    video_has_audio: evt.has_audio ?? false,
    video_duration_ms: evt.duration_ms ?? null,
    video_audio_sample_rate: evt.audio_sample_rate ?? null,
    video_audio_channels: evt.audio_channels ?? null,
    gpu: evt.gpu ?? null,
  };
}

/** Translate a single-clip `GenerateRequestWire` + chain routing decision
 * into the auto-expand `ChainRequestWire` the server expects. */
function buildChainRequest(
  req: GenerateRequestWire,
  decision: Extract<ChainRoutingDecision, { kind: "chain" }>,
): ChainRequestWire {
  return {
    model: req.model,
    motion_tail_frames: decision.motionTail,
    width: req.width,
    height: req.height,
    fps: req.fps ?? 24,
    seed: req.seed ?? null,
    steps: req.steps,
    guidance: req.guidance,
    strength: req.strength ?? 1.0,
    output_format: req.output_format,
    placement: req.placement ?? null,
    prompt: req.prompt,
    total_frames: req.frames ?? undefined,
    clip_frames: decision.clipFrames,
    source_image: req.source_image ?? null,
  };
}

export interface UseGenerateStream {
  jobs: Ref<Job[]>;
  submit: (req: GenerateRequestWire, decision?: ChainRoutingDecision) => string;
  cancel: (id: string) => void;
  clearDone: () => void;
}

export function useGenerateStream(
  onComplete?: (job: Job) => void,
): UseGenerateStream {
  const jobs = ref<Job[]>([]);

  function submit(
    req: GenerateRequestWire,
    decision: ChainRoutingDecision = { kind: "single" },
  ): string {
    const id = crypto.randomUUID();
    const controller = new AbortController();
    const isChain = decision.kind === "chain";
    // Wrap in reactive() so that property mutations during SSE streaming
    // (stage, step, state, result) trigger RunningJobCard re-renders. The
    // closure must hold the proxy, not the raw object — mutations through
    // the raw target bypass the Proxy's set trap and skip dep notification.
    const job = reactive<Job>({
      id,
      request: req,
      startedAt: Date.now(),
      controller,
      progress: emptyProgress(),
      result: null,
      error: null,
      state: "running",
      chain: isChain
        ? {
            stageCount: (
              decision as Extract<ChainRoutingDecision, { kind: "chain" }>
            ).stageCount,
            currentStage: 0,
            estimatedTotalFrames: null,
          }
        : null,
    }) as Job;
    jobs.value = [job, ...jobs.value];

    const onErrorCommon = (err: {
      kind: "http" | "network";
      status?: number;
      retryAfter?: number;
      body?: string;
      message?: string;
    }) => {
      if (err.kind === "http") {
        job.error =
          err.status === 503
            ? `Queue full (retry after ${err.retryAfter ?? "?"}s)`
            : `HTTP ${err.status}: ${err.body ?? ""}`;
      } else {
        job.error = err.message ?? "network error";
      }
      job.state = "error";
    };

    if (decision.kind === "chain") {
      const chainReq = buildChainRequest(req, decision);
      generateChainStream(
        chainReq,
        {
          onProgress: (evt) => applyChainProgress(job, evt),
          onComplete: (evt) => {
            job.result = chainCompleteToSingle(req, evt);
            job.state = "done";
            if (evt.gpu !== null && evt.gpu !== undefined)
              job.progress.gpu = evt.gpu;
            onComplete?.(job);
          },
          onError: onErrorCommon,
        },
        controller.signal,
      );
    } else {
      generateStream(
        req,
        {
          onProgress: (evt) => applyProgress(job, evt),
          onComplete: (evt) => {
            job.result = evt;
            job.state = "done";
            if (evt.gpu !== null && evt.gpu !== undefined)
              job.progress.gpu = evt.gpu;
            onComplete?.(job);
          },
          onError: onErrorCommon,
        },
        controller.signal,
      );
    }

    return id;
  }

  function cancel(id: string) {
    const job = jobs.value.find((j) => j.id === id);
    if (!job) return;
    job.controller.abort();
    job.state = "canceled";
  }

  function clearDone() {
    jobs.value = jobs.value.filter((j) => j.state === "running");
  }

  return { jobs, submit, cancel, clearDone };
}

import { reactive, ref, type Ref } from "vue";
import { generateStream } from "../api";
import type {
  GenerateRequestWire,
  SseCompleteEvent,
  SseProgressEvent,
} from "../types";

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

export interface UseGenerateStream {
  jobs: Ref<Job[]>;
  submit: (req: GenerateRequestWire) => string;
  cancel: (id: string) => void;
  clearDone: () => void;
}

/*
 * `crypto.randomUUID` is only available in a secure context — serving the
 * SPA over a bare LAN IP (`http://10.x.y.z:5173`) leaves it undefined and
 * the submit path throws. IDs here are purely client-side correlation
 * keys, so a non-cryptographic fallback is fine.
 */
function makeJobId(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function useGenerateStream(
  onComplete?: (job: Job) => void,
): UseGenerateStream {
  const jobs = ref<Job[]>([]);

  function submit(req: GenerateRequestWire): string {
    const id = makeJobId();
    const controller = new AbortController();
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
    }) as Job;
    jobs.value = [job, ...jobs.value];

    generateStream(
      req,
      {
        onProgress: (evt) => {
          applyProgress(job, evt);
        },
        onComplete: (evt) => {
          job.result = evt;
          job.state = "done";
          if (evt.gpu !== null && evt.gpu !== undefined)
            job.progress.gpu = evt.gpu;
          onComplete?.(job);
        },
        onError: (err) => {
          if (err.kind === "http") {
            job.error =
              err.status === 503
                ? `Queue full (retry after ${err.retryAfter ?? "?"}s)`
                : `HTTP ${err.status}: ${err.body}`;
          } else {
            job.error = err.message;
          }
          job.state = "error";
        },
      },
      controller.signal,
    );

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

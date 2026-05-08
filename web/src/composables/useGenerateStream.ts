import { onUnmounted, reactive, ref, watch, type Ref } from "vue";
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
  /** Either a single-clip `GenerateRequestWire` or a canonical
   * `ChainRequestWire` (Script mode on GeneratePage submits the latter
   * directly). Only `model` is read from this, so the union is safe. */
  request: GenerateRequestWire | ChainRequestWire;
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
  req: GenerateRequestWire | ChainRequestWire,
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
    // Forward the single-mode audio toggle into the auto-expand chain so
    // LTX-2.3 users with `frames > 97` still get audio. Omitted (undefined)
    // when the form's enableAudio is null — the chain endpoint then
    // defaults to off, matching the wire's omit-when-None semantics.
    enable_audio: req.enable_audio ?? undefined,
  };
}

/** Returns `true` when `req` is already a canonical `ChainRequestWire` with
 * at least one stage authored — i.e. Script-mode submissions that should be
 * sent verbatim instead of re-projected through `buildChainRequest`. */
export function isPrebuiltChainRequest(
  req: GenerateRequestWire | ChainRequestWire,
): req is ChainRequestWire {
  const stages = (req as ChainRequestWire).stages;
  return Array.isArray(stages) && stages.length > 0;
}

/** Decide which wire body to send for a chain submission. Script-mode
 * callers pass a `ChainRequestWire` with populated `stages` — that goes
 * through untouched. Single-prompt callers pass a `GenerateRequestWire`
 * whose `frames` crossed the per-clip cap; those get projected into the
 * auto-expand form.
 *
 * Exported so unit tests can cover the branching without mocking SSE. */
export function resolveChainRequest(
  req: GenerateRequestWire | ChainRequestWire,
  decision: Extract<ChainRoutingDecision, { kind: "chain" }>,
): ChainRequestWire {
  if (isPrebuiltChainRequest(req)) return req;
  return buildChainRequest(req, decision);
}

export interface UseGenerateStream {
  jobs: Ref<Job[]>;
  submit: (
    req: GenerateRequestWire | ChainRequestWire,
    decision?: ChainRoutingDecision,
  ) => string;
  cancel: (id: string) => void;
  clearDone: () => void;
  /** Remove a specific job from the list (used to dismiss persisted cards). */
  remove: (id: string) => void;
}

const STORAGE_KEY = "mold.generate.jobs";

/// Maximum number of *settled* (done/error/canceled) jobs we keep in
/// localStorage. Running jobs are never dropped — the user might be
/// watching them across navigations. Past this cap, oldest settled
/// jobs are forgotten on the next persist cycle. The completed image
/// itself lives in the gallery DB; the in-memory `jobs` list is just
/// the SPA's "recent activity" rail.
const SETTLED_HISTORY_CAP = 10;

/** Shape we persist to localStorage — everything in `Job` minus the
 * non-serializable `AbortController`, plus a marker so we can short-circuit
 * loads from a future schema bump.
 *
 * `result` is *trimmed*: we drop the base64 `image`/`video_thumbnail`/
 * `video_gif_preview` bytes before writing. They're megabytes each;
 * persisting them on every progress tick (200 ms debounced deep watch)
 * blocks the main thread on `JSON.stringify` + `localStorage.setItem`.
 * The live UI still holds the full result in memory; on reload the
 * image/video re-loads from `/api/gallery` which is the durable source
 * of truth anyway.
 */
type PersistedResult = Omit<
  SseCompleteEvent,
  "image" | "video_thumbnail" | "video_gif_preview"
>;

interface PersistedJob {
  id: string;
  request: GenerateRequestWire | ChainRequestWire;
  startedAt: number;
  progress: JobProgress;
  result: PersistedResult | null;
  error: string | null;
  state: Job["state"];
  chain: ChainJobMeta | null;
}

function stripHeavyResult(r: SseCompleteEvent | null): PersistedResult | null {
  if (!r) return null;
  // Discriminated drop — leave every metadata field intact so the
  // RunningJobCard can still render dimensions/timing on rehydrate.
  // The intentionally-omitted ones are exactly the base64 payloads.
  const { image: _i, video_thumbnail: _t, video_gif_preview: _g, ...rest } = r;
  void _i;
  void _t;
  void _g;
  return rest;
}

function loadPersistedJobs(): Job[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as PersistedJob[];
    if (!Array.isArray(parsed)) return [];
    return parsed.map((p) => {
      // Running jobs are kept as `running` after a route change — with the
      // singleton scope the SSE callback closures from `submit()` continue
      // mutating the same `jobs` ref across navigation, so a job that was
      // mid-generation when the user navigated away is still receiving
      // live progress events.
      //
      // Hard refreshes / server restarts are different: the SSE stream is
      // truly gone and no further events will arrive. We don't have a
      // way to tell those cases apart from in-progress-still-streaming
      // here — RunningJobCard renders a "stale"/"reconnecting" hint
      // after a timeout, and the user can dismiss / retry from the UI.
      //
      // `result` is null for running/error/cancelled jobs and a
      // metadata-only object for done jobs (the base64 image was
      // stripped at persist time — see `persistJobs`). The live result
      // is repopulated from `/api/gallery` if the user clicks back into
      // a completed job's preview.
      return {
        id: p.id,
        request: p.request,
        startedAt: p.startedAt,
        // Dangling controllers from prior sessions aren't used — cancel()
        // bails early for non-running jobs anyway.
        controller: new AbortController(),
        progress: p.progress ?? emptyProgress(),
        result: p.result as SseCompleteEvent | null,
        error: p.error,
        state: p.state,
        chain: p.chain,
      };
    });
  } catch {
    return [];
  }
}

function persistJobs(jobs: Job[]) {
  try {
    // Always keep running jobs (the user is watching them); cap the
    // number of settled jobs persisted so localStorage doesn't grow
    // unbounded across sessions. Order is preserved — `submit` prepends
    // new jobs, so the cap takes the most recent settled entries.
    const settledCount = { n: 0 };
    const filtered = jobs.filter((j) => {
      if (j.state === "running") return true;
      settledCount.n += 1;
      return settledCount.n <= SETTLED_HISTORY_CAP;
    });
    const serializable: PersistedJob[] = filtered.map((j) => ({
      id: j.id,
      request: j.request,
      startedAt: j.startedAt,
      progress: j.progress,
      result: stripHeavyResult(j.result),
      error: j.error,
      state: j.state,
      chain: j.chain,
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  } catch {
    /* quota / privacy mode — silently drop */
  }
}

// ── Module-level singleton state ─────────────────────────────────────────────
//
// Pre-singleton: `useGenerateStream()` was invoked inside `GeneratePage.vue`'s
// `setup()`, so each mount got its own `jobs` ref and watcher. Navigating
// away → back created a fresh instance whose `jobs` was loaded from
// localStorage; the SSE callbacks from the previous instance kept mutating
// the *old* (orphaned) ref, so live progress was invisible to the new view.
// Lifting `jobs` and `submit` to module scope makes the state survive route
// changes — the same ref is shared by every consumer.
//
// Per-mount concerns (the `onComplete` listener) move to a Set with
// register/unregister, so a stale toast handler from an unmounted component
// doesn't keep firing.

const jobs = ref<Job[]>(loadPersistedJobs());

// Persist whenever the list or any job's mutable state changes. 200 ms
// debounce keeps writes out of the SSE hot path (we get a progress event
// roughly every 50 ms during denoising).
let persistTimer: ReturnType<typeof setTimeout> | null = null;
watch(
  jobs,
  (v) => {
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => persistJobs(v), 200);
  },
  { deep: true },
);

type CompleteListener = (job: Job) => void;
const completeListeners = new Set<CompleteListener>();

function fireComplete(job: Job) {
  for (const cb of completeListeners) {
    try {
      cb(job);
    } catch (e) {
      console.error("generate onComplete listener threw", e);
    }
  }
}

/// Grace period before a successfully-completed job's running-strip card
/// is auto-dismissed. Keeps the thumbnail on screen long enough for the
/// user to register that "yes, the thing I asked for finished" before it
/// quietly drops away into the gallery feed below. Errored / canceled
/// jobs never auto-dismiss — they have nothing in the gallery to fall
/// back to and the user may want to re-read the error.
const AUTO_REMOVE_DONE_MS = 1500;

/** Schedule auto-removal of a successfully-completed job. Exported only
 * so the constant is reachable in tests; production code calls this
 * inside the SSE complete handlers. Safe to call for jobs that the user
 * has already manually dismissed — `removeJob` filters by id, so a
 * missing id is a no-op. */
function scheduleAutoRemoveOnDone(id: string) {
  setTimeout(() => removeJob(id), AUTO_REMOVE_DONE_MS);
}

export const __testing__ = {
  AUTO_REMOVE_DONE_MS,
};

function submitJob(
  req: GenerateRequestWire | ChainRequestWire,
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
    const chainReq = resolveChainRequest(req, decision);
    generateChainStream(
      chainReq,
      {
        onProgress: (evt) => applyChainProgress(job, evt),
        onComplete: (evt) => {
          job.result = chainCompleteToSingle(req, evt);
          job.state = "done";
          if (evt.gpu !== null && evt.gpu !== undefined)
            job.progress.gpu = evt.gpu;
          fireComplete(job);
          scheduleAutoRemoveOnDone(id);
        },
        onError: onErrorCommon,
      },
      controller.signal,
    );
  } else if (isPrebuiltChainRequest(req)) {
    // Caller bug: a stages-based ChainRequestWire was submitted with a
    // non-chain routing decision. The single-clip endpoint would reject
    // the unknown `stages` field, so bail early with a clear message
    // instead of producing an opaque 422/500.
    job.error =
      "internal: ChainRequestWire submitted with non-chain routing decision";
    job.state = "error";
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
          fireComplete(job);
          scheduleAutoRemoveOnDone(id);
        },
        onError: onErrorCommon,
      },
      controller.signal,
    );
  }

  return id;
}

function cancelJob(id: string) {
  const job = jobs.value.find((j) => j.id === id);
  if (!job) return;
  job.controller.abort();
  job.state = "canceled";
}

function clearDoneJobs() {
  jobs.value = jobs.value.filter((j) => j.state === "running");
}

function removeJob(id: string) {
  jobs.value = jobs.value.filter((j) => j.id !== id);
}

export function useGenerateStream(
  onComplete?: (job: Job) => void,
): UseGenerateStream {
  // Per-call: register the optional `onComplete` listener and tear it
  // down when the calling component unmounts so navigating away from
  // GeneratePage doesn't leak callbacks into module-level state.
  // `onUnmounted` is a no-op outside a component instance, which keeps
  // direct test invocations harmless.
  if (onComplete) {
    completeListeners.add(onComplete);
    onUnmounted(() => {
      completeListeners.delete(onComplete);
    });
  }

  return {
    jobs,
    submit: submitJob,
    cancel: cancelJob,
    clearDone: clearDoneJobs,
    remove: removeJob,
  };
}

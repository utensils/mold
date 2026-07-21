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
import type { HostRoute } from "../lib/hostRouting";

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
  /** `Date.now()` of the most recent SSE event delivered to this job.
   * Lets `RunningJobCard` flag a stale stream (no progress for >60 s) so
   * the user can dismiss / retry instead of staring at a frozen card
   * after the underlying connection silently dropped. */
  lastProgressAt: number;
  /** True after the job has moved past server queue bookkeeping and
   * produced progress from actual work (model download/load, cache hit,
   * denoising, chain orchestration, etc.). Queued jobs can legitimately sit
   * quiet for a long time, so stale-stream warnings must stay suppressed
   * until this flips. */
  workStarted: boolean;
  /** Registry id of the host this job was dispatched to, and that host's
   * label. `null` for submissions that were never routed (single-host web, or
   * jobs persisted before routing existed) — those ran on the origin. Carried
   * so the activity strip and the result caption can attribute the print to
   * the machine that actually rendered it. */
  hostId: string | null;
  hostLabel: string | null;
  /** Server-assigned UUID, captured from the first `queued` SSE event.
   * `null` until that event arrives (e.g. between submit and HTTP
   * handshake), and stays `null` against legacy servers that predate
   * L3. The reconciliation poller only sweeps cards whose `serverId`
   * is non-null — without an id we can't tell server-side reality from
   * legacy-server-pretending-it-knows-nothing. */
  serverId: string | null;
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

function markWorkStarted(job: Job) {
  job.workStarted = true;
  job.progress.queuePosition = null;
}

function applyProgress(job: Job, evt: SseProgressEvent) {
  job.lastProgressAt = Date.now();
  const p = job.progress;
  switch (evt.type) {
    case "stage_start":
      markWorkStarted(job);
      p.stage = evt.name;
      break;
    case "stage_done":
      markWorkStarted(job);
      p.stage = `${evt.name} (done)`;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "info":
      // Dimension warnings are emitted before the server queues the job, so
      // an info event only proves real work has started if the job has
      // already passed through a queued event.
      if (p.queuePosition !== null || job.workStarted) markWorkStarted(job);
      p.stage = evt.message;
      break;
    case "denoise_step":
      markWorkStarted(job);
      p.stage = "Denoising";
      p.step = evt.step;
      p.totalSteps = evt.total;
      p.elapsedMs = evt.elapsed_ms;
      break;
    case "queued":
      p.stage = `Queued (position ${evt.position})`;
      p.queuePosition = evt.position;
      // Capture the server-assigned id the first time it lands.
      // Legacy servers (pre-L3) omit `id`; we leave `serverId` null
      // and the reconciliation poller skips this card.
      if (evt.id && !job.serverId) {
        job.serverId = evt.id;
      }
      break;
    case "weight_load":
      markWorkStarted(job);
      p.stage = `Loading ${evt.component}`;
      p.weightBytesLoaded = evt.bytes_loaded;
      p.weightBytesTotal = evt.bytes_total;
      break;
    case "cache_hit":
      markWorkStarted(job);
      p.stage = `Cache hit: ${evt.resource}`;
      break;
  }
}

/** Chain progress events come from a separate SSE stream shape than the
 * single-clip path; we fold them into the same `JobProgress` so the
 * `RunningJobCard` UI renders a familiar "Denoising clip K/N · step X/Y"
 * readout without the per-event UI layer needing to know about chaining. */
function applyChainProgress(job: Job, evt: ChainProgressEvent) {
  job.lastProgressAt = Date.now();
  markWorkStarted(job);
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
    original_prompt: req.original_prompt ?? undefined,
    batch_id: req.batch_id ?? undefined,
    batch_index: req.batch_index ?? undefined,
    batch_count: req.batch_count ?? undefined,
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
    route?: HostRoute | null,
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
  /** May be missing on payloads persisted before lastProgressAt existed —
   * load path falls back to `startedAt`. */
  lastProgressAt?: number;
  /** Optional — pre-queue-stale-fix payloads don't carry workStarted. */
  workStarted?: boolean;
  /** Optional — pre-routing payloads don't carry a host. */
  hostId?: string | null;
  hostLabel?: string | null;
  /** Optional — pre-L3 payloads don't carry a serverId. */
  serverId?: string | null;
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

/** Reconstitute the persisted job rail. `raw` is the localStorage payload
 * (caller injects it so tests can drive the dead-letter logic without
 * touching `localStorage` directly).
 *
 * `loadPersistedJobs` runs exactly once per SPA boot — at module-import
 * time. Any row persisted as `running` therefore belongs to a session
 * whose SSE stream is now dead (the page was hard-reloaded, the user
 * landed in a new tab, or the server restarted). Reconnecting that
 * stream is impossible from this side, so we flip the row to `error`
 * with a load-bearing reason. This is the only mechanism that removes
 * zombie "running" cards left over from prior pile-ups when the queue
 * was deep and connections dropped silently.
 *
 * Within a single SPA session, route changes do NOT call this function —
 * the singleton `jobs` ref is preserved and the SSE callback closures
 * keep mutating it. So this dead-letter does not interfere with the
 * route-change-during-generation flow. */
function loadPersistedJobs(raw: string | null): Job[] {
  try {
    if (!raw) return [];
    const parsed = JSON.parse(raw) as PersistedJob[];
    if (!Array.isArray(parsed)) return [];
    return parsed.map((p) => {
      const wasZombie = p.state === "running";
      const state: Job["state"] = wasZombie ? "error" : p.state;
      const error = wasZombie
        ? (p.error ?? "page reloaded — server progress lost")
        : p.error;
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
        error,
        state,
        chain: p.chain,
        lastProgressAt: p.lastProgressAt ?? p.startedAt,
        workStarted: p.workStarted ?? state !== "running",
        hostId: p.hostId ?? null,
        hostLabel: p.hostLabel ?? null,
        serverId: p.serverId ?? null,
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
      lastProgressAt: j.lastProgressAt,
      workStarted: j.workStarted,
      hostId: j.hostId,
      hostLabel: j.hostLabel,
      serverId: j.serverId,
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

const jobs = ref<Job[]>(
  loadPersistedJobs(
    typeof localStorage !== "undefined"
      ? localStorage.getItem(STORAGE_KEY)
      : null,
  ),
);

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

/** Schedule auto-removal of a successfully-completed job. The timer
 * re-checks `state` at fire time and bails if the job has since flipped
 * to `canceled` (user clicked Cancel during the grace period) — without
 * this, the card would briefly flash "canceled" then auto-dismiss
 * anyway, losing the user's signal. Safe to call for jobs that have
 * already been manually dismissed: `removeJob` filters by id, so a
 * missing id is a no-op. */
function scheduleAutoRemoveOnDone(id: string) {
  setTimeout(() => {
    const job = jobs.value.find((j) => j.id === id);
    if (!job || job.state !== "done") return;
    removeJob(id);
  }, AUTO_REMOVE_DONE_MS);
}

/// How long a running job can go without a progress event before
/// `RunningJobCard` flags it as stale. Calibrated for the slowest
/// realistic path: a fresh model swap on a large quantized engine can
/// hold the load lock for ~30 s without an SSE event, and offload-mode
/// transformer-block streaming can be quiet for a similar stretch. 60 s
/// is a comfortable buffer past both — long enough to avoid false
/// positives during legitimate work, short enough that a truly dropped
/// stream surfaces within a minute instead of leaving the user staring
/// at a frozen card indefinitely.
export const STALE_THRESHOLD_MS = 60_000;

export const __testing__ = {
  AUTO_REMOVE_DONE_MS,
  STALE_THRESHOLD_MS,
  loadPersistedJobs,
  STORAGE_KEY,
};

function submitJob(
  req: GenerateRequestWire | ChainRequestWire,
  decision: ChainRoutingDecision = { kind: "single" },
  route: HostRoute | null = null,
): string {
  const id = crypto.randomUUID();
  const controller = new AbortController();
  const isChain = decision.kind === "chain";
  // Wrap in reactive() so that property mutations during SSE streaming
  // (stage, step, state, result) trigger RunningJobCard re-renders. The
  // closure must hold the proxy, not the raw object — mutations through
  // the raw target bypass the Proxy's set trap and skip dep notification.
  const now = Date.now();
  const job = reactive<Job>({
    id,
    request: req,
    startedAt: now,
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
    lastProgressAt: now,
    workStarted: false,
    hostId: route?.hostId ?? null,
    hostLabel: route?.label ?? null,
    serverId: null,
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
      route?.target,
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
      route?.target,
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

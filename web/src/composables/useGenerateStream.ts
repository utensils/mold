import { computed, onUnmounted, reactive, ref, watch, type Ref } from "vue";
import {
  cancelQueueJob,
  generateChainStream,
  generateStream,
  type StreamTarget,
} from "../api";
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
import { StreamSlotPool } from "../lib/streamSlots";
import { createUuid } from "@studio/lib/id";
import {
  prepareReferenceUploads,
  requestNeedsReferenceUpload,
  type ReferenceUploadLease,
} from "@studio/api/referenceUploads";
import { redactGenerationReference } from "@studio/lib/generationReferences";

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
   * `ChainRequestWire` (Script mode on CreatePage submits the latter
   * directly). Only `model` is read from this, so the union is safe. */
  request: GenerateRequestWire | ChainRequestWire;
  startedAt: number;
  controller: AbortController;
  progress: JobProgress;
  result: SseCompleteEvent | null;
  error: string | null;
  state: "running" | "done" | "error" | "canceled";
  /** Wall clock when the job stopped moving; `null` while it is running.
   * The Create activity strip expires settled-but-failed rows against this
   * (shared @studio partition rule) instead of keeping them forever. */
  settledAt: number | null;
  /** When the job was auto-promoted to the chain endpoint. `null` for a
   * normal single-clip submission. */
  chain: ChainJobMeta | null;
  /** `Date.now()` of the most recent SSE event delivered to this job.
   * Lets the activity strip flag a stale stream (no progress for >60 s) so
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
  /** Exact in-memory route used by cancel and reconciliation. Deliberately not
   * persisted because it may contain a host API key. */
  target: StreamTarget | null;
  /** Server-assigned UUID, captured from the first `queued` SSE event.
   * `null` until the required queued event arrives. The reconciliation poller
   * only sweeps cards whose `serverId` is known. */
  serverId: string | null;
  /** Latest live latent preview as a `data:image/png;base64,…` URI.
   * Deliberately a data URI rather than a blob URL: the module-singleton job
   * list is persisted and shared across consumers, so there is no sane place
   * to hang a `URL.revokeObjectURL` lifecycle. Cleared when the job settles;
   * never persisted. */
  previewUrl: string | null;
  /** Seed driving the Develop grain — the explicit request seed as a string,
   * or a stable `model·prompt` stand-in when the seed is random (desktop
   * `generationJob.ts` recipe). Recomputed from the request on rehydrate. */
  seedVisual: string;
}

/**
 * The running job the Create canvas should develop.
 *
 * The rail is newest-first, but with several jobs queued (prepared batch
 * variations) the server denoises the EARLIEST submission — a naive "first
 * running" pick binds the canvas to a job that sits previewless while another
 * is actively developing. Prefer the running job that holds a live preview
 * (proof the server is denoising it); otherwise the earliest-submitted
 * running job, which is next in line.
 */
export function activeCanvasJob(jobs: readonly Job[]): Job | undefined {
  let earliest: Job | undefined;
  for (const job of jobs) {
    if (job.state !== "running") continue;
    if (job.previewUrl !== null) return job;
    if (!earliest || job.startedAt < earliest.startedAt) earliest = job;
  }
  return earliest;
}

/**
 * Resolve the failure that should own the Create canvas.
 *
 * Failed rows remain in persisted Activity history so users can inspect or
 * dismiss them. They must not regain canvas authority after a newer print
 * succeeds and its short-lived completion card auto-removes from `jobs`.
 * An explicitly selected failure still wins because opening that row is a
 * deliberate request to inspect it.
 */
export function latestUnresolvedError(
  jobs: readonly Job[],
  canvasErrorJobId: string | null,
  selected: Job | null = null,
): Job | undefined {
  if (selected) return selected.state === "error" ? selected : undefined;
  if (canvasErrorJobId === null) return undefined;
  const job = jobs.find((candidate) => candidate.id === canvasErrorJobId);
  return job?.state === "error" ? job : undefined;
}

function seedVisualFor(req: GenerateRequestWire | ChainRequestWire): string {
  return req.seed != null
    ? String(req.seed)
    : `${req.model}·${(req as GenerateRequestWire).prompt ?? ""}`;
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

function serverErrorMessage(body: string | undefined): string | null {
  if (!body) return null;
  try {
    const parsed = JSON.parse(body) as {
      message?: unknown;
      error?: unknown;
    };
    if (typeof parsed.message === "string" && parsed.message.trim()) {
      return parsed.message;
    }
    if (typeof parsed.error === "string" && parsed.error.trim()) {
      return parsed.error;
    }
  } catch {
    // Plain-text HTTP errors are already suitable for display.
  }
  return body.trim() || null;
}

function markWorkStarted(job: Job) {
  job.workStarted = true;
  job.progress.queuePosition = null;
}

function applyProgress(job: Job, evt: SseProgressEvent) {
  job.lastProgressAt = Date.now();
  const p = job.progress;
  switch (evt.type) {
    case "dependency_wait":
      p.stage = `Waiting for ${evt.dependency}`;
      break;
    case "download_progress": {
      const percent =
        evt.bytes_total > 0
          ? Math.round((evt.bytes_downloaded / evt.bytes_total) * 100)
          : 0;
      p.stage = `Downloading ${evt.filename} (${percent}%)`;
      break;
    }
    case "download_done":
      p.stage = `Dependency ready: ${evt.filename}`;
      break;
    case "pull_complete":
      p.stage = `Dependency ready: ${evt.model}`;
      break;
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
    case "preview":
      // A latent preview only exists once denoising is underway.
      markWorkStarted(job);
      p.stage = "Denoising";
      p.step = evt.step;
      p.totalSteps = evt.total;
      job.previewUrl = `data:image/png;base64,${evt.image}`;
      break;
    case "queued":
      p.stage = `Queued (position ${evt.position})`;
      p.queuePosition = evt.position;
      if (!job.serverId) {
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
 * activity UI renders a familiar "Denoising clip K/N · step X/Y"
 * readout without the per-event UI layer needing to know about chaining. */
function applyChainProgress(job: Job, evt: ChainProgressEvent) {
  job.lastProgressAt = Date.now();
  const p = job.progress;
  const meta = job.chain;
  switch (evt.type) {
    case "chain_start":
      if (meta) {
        meta.stageCount = evt.stage_count;
        meta.estimatedTotalFrames = evt.estimated_total_frames;
      }
      // The compatibility endpoint synthesizes chain_start from its initial
      // snapshot even while the durable job is queued. StageStart is the
      // first proof that a scheduler lane is actually executing it.
      p.stage = `Queued · ${evt.stage_count} clips · ~${evt.estimated_total_frames} frames`;
      break;
    case "stage_start":
      markWorkStarted(job);
      if (meta) meta.currentStage = evt.stage_idx;
      p.stage = chainStageLabel(meta, evt.stage_idx, "Preparing");
      p.step = null;
      p.totalSteps = null;
      break;
    case "denoise_step":
      markWorkStarted(job);
      if (meta) meta.currentStage = evt.stage_idx;
      p.stage = chainStageLabel(meta, evt.stage_idx, "Denoising");
      p.step = evt.step;
      p.totalSteps = evt.total;
      break;
    case "stage_done": {
      // Durable stages release their lane independently. Until the next
      // stage_start arrives this is queued work, not model loading. The final
      // stage instead moves directly into final-output preparation.
      const finalStage =
        meta?.stageCount != null && evt.stage_idx + 1 >= meta.stageCount;
      job.workStarted = finalStage;
      p.queuePosition = null;
      p.stage = meta?.stageCount
        ? finalStage
          ? `Clip ${evt.stage_idx + 1}/${meta.stageCount} done · preparing final output`
          : `Clip ${evt.stage_idx + 1}/${meta.stageCount} done · next clip queued`
        : `Clip ${evt.stage_idx + 1} done · next clip queued`;
      p.step = null;
      p.totalSteps = null;
      break;
    }
    case "stitching":
      markWorkStarted(job);
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
 * `SseCompleteEvent` so `CreatePage.openJob` and the activity strip stay
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
  selectedJob: Ref<Job | null>;
  /** Exact terminal failure that currently owns the canvas. Historical
   * failures remain Activity rows unless explicitly opened. */
  canvasErrorJobId: Ref<string | null>;
  submit: (
    req: GenerateRequestWire | ChainRequestWire,
    decision?: ChainRoutingDecision,
    route?: HostRoute | null,
  ) => string;
  cancel: (id: string) => Promise<void>;
  /** Settle a still-running job as failed. Used by external liveness
   * authorities such as queue reconciliation so every failure updates the
   * canvas owner and terminal metadata through the same path. */
  failRunning: (id: string, error: string) => void;
  clearDone: () => void;
  /** Remove a specific job from the list (used to dismiss persisted cards). */
  remove: (id: string) => void;
  select: (id: string | null) => void;
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
  settledAt?: number | null;
  chain: ChainJobMeta | null;
  lastProgressAt: number;
  workStarted: boolean;
  hostId: string | null;
  hostLabel: string | null;
  serverId: string | null;
}

const JOB_STORAGE_VERSION = 1;

interface PersistedJobs {
  version: typeof JOB_STORAGE_VERSION;
  jobs: PersistedJob[];
}

function stripHeavyResult(r: SseCompleteEvent | null): PersistedResult | null {
  if (!r) return null;
  // Discriminated drop — leave every metadata field intact so the
  // The activity strip can still render dimensions/timing on rehydrate.
  // The intentionally-omitted ones are exactly the base64 payloads.
  const {
    image: _i,
    video_thumbnail: _t,
    video_gif_preview: _g,
    audio_thumbnail: _a,
    ...rest
  } = r;
  void _i;
  void _t;
  void _g;
  void _a;
  return rest;
}

/** Job recovery keeps only redacted reference descriptors. Inline bytes and
 * one-use upload handles are session memory, never localStorage state. */
function persistenceSafeRequest(
  request: GenerateRequestWire | ChainRequestWire,
): GenerateRequestWire | ChainRequestWire {
  if (isPrebuiltChainRequest(request) || !request.references?.length) {
    return request;
  }
  return {
    ...request,
    references: request.references.map(redactGenerationReference),
  };
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
interface LoadedJobsState {
  jobs: Job[];
  /** A row that was still running when this browser session ended is a
   * failure discovered by the current boot, not settled history. */
  canvasErrorJobId: string | null;
}

function loadPersistedState(raw: string | null): LoadedJobsState {
  try {
    if (!raw) return { jobs: [], canvasErrorJobId: null };
    const parsed = JSON.parse(raw) as PersistedJobs;
    if (parsed.version !== JOB_STORAGE_VERSION || !Array.isArray(parsed.jobs)) {
      return { jobs: [], canvasErrorJobId: null };
    }
    let newestZombie: PersistedJob | null = null;
    for (const persisted of parsed.jobs) {
      if (
        persisted.state === "running" &&
        (!newestZombie || persisted.startedAt > newestZombie.startedAt)
      ) {
        newestZombie = persisted;
      }
    }
    const canvasErrorJobId = newestZombie?.id ?? null;
    const loadedAt = Date.now();
    const loadedJobs = parsed.jobs.map((p) => {
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
        progress: p.progress,
        result: p.result as SseCompleteEvent | null,
        error,
        state,
        // This boot just discovered that a formerly-running row lost its
        // stream, so keep its recovery row present and dismissible from now.
        // Genuinely settled history retains its original age.
        settledAt: wasZombie
          ? loadedAt
          : (p.settledAt ?? p.lastProgressAt ?? p.startedAt),
        chain: p.chain,
        lastProgressAt: p.lastProgressAt,
        workStarted: p.workStarted,
        hostId: p.hostId,
        hostLabel: p.hostLabel,
        target: null,
        serverId: p.serverId,
        // Previews are ephemeral SSE payload — never persisted.
        previewUrl: null,
        seedVisual: seedVisualFor(p.request),
      };
    });
    return { jobs: loadedJobs, canvasErrorJobId };
  } catch {
    return { jobs: [], canvasErrorJobId: null };
  }
}

function loadPersistedJobs(raw: string | null): Job[] {
  return loadPersistedState(raw).jobs;
}

function initializePersistedState(raw: string | null): LoadedJobsState {
  const loaded = loadPersistedState(raw);
  // The deep watcher is intentionally not immediate. Write a boot-created
  // dead letter through synchronously so a second refresh cannot rediscover
  // the same formerly-running row as a new current failure.
  if (loaded.canvasErrorJobId !== null) persistJobs(loaded.jobs);
  return loaded;
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
      request: persistenceSafeRequest(j.request),
      startedAt: j.startedAt,
      progress: j.progress,
      result: stripHeavyResult(j.result),
      error: j.error,
      state: j.state,
      settledAt: j.settledAt,
      chain: j.chain,
      lastProgressAt: j.lastProgressAt,
      workStarted: j.workStarted,
      hostId: j.hostId,
      hostLabel: j.hostLabel,
      serverId: j.serverId,
    }));
    const payload: PersistedJobs = {
      version: JOB_STORAGE_VERSION,
      jobs: serializable,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* quota / privacy mode — silently drop */
  }
}

// ── Module-level singleton state ─────────────────────────────────────────────
//
// Pre-singleton: `useGenerateStream()` was invoked inside `CreatePage.vue`'s
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

const initialPersistedState = initializePersistedState(
  typeof localStorage !== "undefined"
    ? localStorage.getItem(STORAGE_KEY)
    : null,
);
const jobs = ref<Job[]>(initialPersistedState.jobs);
// A fresh SPA boot treats rehydrated failures as Activity history instead of
// restoring one as the main canvas. A formerly-running row is different: this
// boot just discovered that its live progress was lost, so its recovery error
// owns the canvas. Live terminal events then update this explicit authority in
// callback order, without wall-clock ambiguity.
const canvasErrorJobId = ref<string | null>(
  initialPersistedState.canvasErrorJobId,
);
const selectedJobId = ref<string | null>(null);
const selectedJob = computed(
  () => jobs.value.find((job) => job.id === selectedJobId.value) ?? null,
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

function recordSuccessfulSettlement(job: Job) {
  job.settledAt = Date.now();
  canvasErrorJobId.value = null;
}

function recordFailedSettlement(job: Job) {
  job.state = "error";
  job.settledAt = Date.now();
  job.previewUrl = null;
  canvasErrorJobId.value = job.id;
}

function failRunningJob(id: string, error: string) {
  const job = jobs.value.find((candidate) => candidate.id === id);
  // Queue reconciliation is asynchronous. A completion or cancellation may
  // land while its GET /api/queue request is in flight; terminal state wins.
  if (!job || job.state !== "running") return;
  job.error = error;
  recordFailedSettlement(job);
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
/// the activity strip flags it as stale. Calibrated for the slowest
/// realistic path: a fresh model swap on a large quantized engine can
/// hold the load lock for ~30 s without an SSE event, and offload-mode
/// transformer-block streaming can be quiet for a similar stretch. 60 s
/// is a comfortable buffer past both — long enough to avoid false
/// positives during legitimate work, short enough that a truly dropped
/// stream surfaces within a minute instead of leaving the user staring
/// at a frozen card indefinitely.
export const STALE_THRESHOLD_MS = 60_000;
const streamSlots = new StreamSlotPool(4);

export const __testing__ = {
  AUTO_REMOVE_DONE_MS,
  STALE_THRESHOLD_MS,
  loadPersistedJobs,
  loadPersistedState,
  initializePersistedState,
  persistJobs,
  STORAGE_KEY,
};

function submitJob(
  req: GenerateRequestWire | ChainRequestWire,
  decision: ChainRoutingDecision = { kind: "single" },
  route: HostRoute | null = null,
): string {
  selectedJobId.value = null;
  canvasErrorJobId.value = null;
  const id = createUuid();
  const controller = new AbortController();
  const isChain = decision.kind === "chain";
  // Wrap in reactive() so that property mutations during SSE streaming
  // (stage, step, state, result) trigger activity-strip re-renders. The
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
    settledAt: null,
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
    target: route?.target ?? null,
    serverId: null,
    previewUrl: null,
    seedVisual: seedVisualFor(req),
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
      const message = serverErrorMessage(err.body);
      job.error =
        err.status === 503
          ? `Queue full (retry after ${err.retryAfter ?? "?"}s)`
          : err.status === 0
            ? (message ?? "generation failed")
            : `HTTP ${err.status}${message ? `: ${message}` : ""}`;
    } else {
      job.error = err.message ?? "network error";
    }
    recordFailedSettlement(job);
  };

  const startStream = async () => {
    if (decision.kind === "chain") {
      const chainReq = resolveChainRequest(req, decision);
      await generateChainStream(
        chainReq,
        {
          onProgress: (evt) => applyChainProgress(job, evt),
          onComplete: (evt) => {
            job.result = chainCompleteToSingle(req, evt);
            job.state = "done";
            recordSuccessfulSettlement(job);
            job.previewUrl = null;
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
      recordFailedSettlement(job);
    } else {
      let lease: ReferenceUploadLease<GenerateRequestWire> | null = null;
      try {
        let transportRequest = req;
        if (requestNeedsReferenceUpload(req)) {
          if (!route) {
            throw new Error(
              "MiniMax H3 reference uploads require a frozen authenticated host route.",
            );
          }
          const prepared = await prepareReferenceUploads({
            target: {
              baseUrl: route.target.baseUrl,
              apiKey: route.target.apiKey ?? null,
            },
            expectedInstanceId: route.instanceId ?? "",
            capabilities: route.referenceUploads,
            request: req,
            signal: controller.signal,
          });
          lease = prepared;
          transportRequest = prepared.request;
        }
        await generateStream(
          transportRequest,
          {
            onProgress: (evt) => applyProgress(job, evt),
            onComplete: (evt) => {
              job.result = evt;
              job.state = "done";
              recordSuccessfulSettlement(job);
              job.previewUrl = null;
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
      } catch (error) {
        if (!controller.signal.aborted && job.state === "running") {
          onErrorCommon({
            kind: "network",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      } finally {
        if (lease) void lease.cancel().catch(() => undefined);
      }
    }
  };

  // Four held-open render streams leave browser connection headroom for queue
  // reconciliation, gallery refreshes, and model downloads. Waiting jobs keep
  // their visible Starting state and can be canceled before they acquire a slot.
  streamSlots.schedule(controller.signal, (release) => {
    void startStream().finally(release);
  });

  return id;
}

async function cancelJob(id: string): Promise<void> {
  const job = jobs.value.find((j) => j.id === id);
  if (!job) return;
  job.controller.abort();
  job.state = "canceled";
  job.settledAt = Date.now();
  job.previewUrl = null;
  if (selectedJobId.value === id) selectedJobId.value = null;
  if (!job.serverId) return;
  try {
    await cancelQueueJob(job.serverId, job.target ?? undefined);
  } catch (error) {
    job.error = error instanceof Error ? error.message : String(error);
    recordFailedSettlement(job);
  }
}

function clearDoneJobs() {
  jobs.value = jobs.value.filter((j) => j.state === "running");
  canvasErrorJobId.value = null;
}

function removeJob(id: string) {
  jobs.value = jobs.value.filter((j) => j.id !== id);
  if (selectedJobId.value === id) selectedJobId.value = null;
  if (canvasErrorJobId.value === id) canvasErrorJobId.value = null;
}

function selectJob(id: string | null) {
  selectedJobId.value =
    id !== null && jobs.value.some((job) => job.id === id) ? id : null;
}

export function useGenerateStream(
  onComplete?: (job: Job) => void,
): UseGenerateStream {
  // Per-call: register the optional `onComplete` listener and tear it
  // down when the calling component unmounts so navigating away from
  // CreatePage doesn't leak callbacks into module-level state.
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
    selectedJob,
    canvasErrorJobId,
    submit: submitJob,
    cancel: cancelJob,
    failRunning: failRunningJob,
    clearDone: clearDoneJobs,
    remove: removeJob,
    select: selectJob,
  };
}

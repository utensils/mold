import { computed, onUnmounted, reactive, ref, watch, type Ref } from "vue";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import {
  cancelQueueJob,
  fetchQueue,
  generateChainStream,
  generateStream,
  listGalleryFrom,
  type StreamTarget,
} from "../api";
import type {
  ChainProgressEvent,
  ChainRequestWire,
  GalleryImage,
  GenerateRequestWire,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
} from "../types";
import type { ChainRoutingDecision } from "../lib/chainRouting";
import type { HostRoute } from "../lib/hostRouting";
import { TargetStreamSlots } from "@studio/lib/targetStreamSlots";
import { createUuid } from "@studio/lib/id";
import {
  prepareReferenceUploads,
  requestNeedsReferenceUpload,
  type ReferenceUploadLease,
} from "@studio/api/referenceUploads";
import { redactGenerationMediaForPersistence } from "@studio/lib/generationMedia";
import { getHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { toast } from "../lib/toasts";
import {
  fetchGalleryBlob,
  fetchGalleryThumbnailBlob,
} from "../lib/galleryMedia";
import { blobToBase64 } from "../lib/base64";
import { inferFormatFromName, type OutputFormat } from "../types";
import { ApiError, apiHeaders, type ApiTarget } from "@studio/api/client";
import { mergeQueueEntries } from "@studio/api/queuePlan";
import {
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  reconcileGenerationBatches,
  supportsDurableRequest,
  supportsDurableGenerationLifecycle,
  type GenerationBatchStatus,
} from "@studio/api/generationAdmission";
import {
  buildGenerationBatchStatusRequest,
  createGenerationBatchTracker,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
  type GenerationBatchTracker,
  type GenerationLifecycleJob,
} from "@studio/lib/generationLifecycle";
import { isMinimaxH3Identity } from "@studio/lib/minimaxH3Identity";

function surfaceRequestWarnings(warnings: string[]): void {
  for (const warning of warnings) toast("warning", warning);
}

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
  /** Immediate UI acknowledgement while DELETE revokes server authority. */
  cancelling?: boolean;
  /** Durable cancellation is an intent until the host confirms DELETE or a
   * terminal outcome wins. This survives ambiguous admission, where the
   * client UUID is known before the server job UUID is. */
  cancelRequested?: boolean;
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
  /** True once the generation endpoint has been opened. Before this flips the
   * job is only waiting for a local browser slot and can be safely aborted;
   * afterward cancellation requires a server queue id and confirmed DELETE. */
  streamStarted?: boolean;
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
  /** True when the HOST owns this job's fate rather than this page: a row
   * rehydrated from a previous session whose SSE stream is gone but whose
   * `serverId` is known (the reconciler, not this boot, rules on it), or a
   * row settled by `settleDetachedJob` — the job was retained across a server
   * restart, or ran while the page was away. A settled detached row is
   * advisory, never a failure, so the activity strip retires it instead of
   * labelling it "Failed". Derived on load for the running case and persisted
   * for the settled one, which a reload would otherwise demote to a plain
   * error. */
  detached?: boolean;
  /** The host journalled THIS job at admission (`/api/queue` row `durable`),
   * captured while the row was still listable. Evidence that a vanished row
   * means "kept and running" rather than "lost". Absent until the answer
   * arrives, and on hosts without a durable queue. */
  durable?: boolean;
  /** Streamless, instance-fenced lifecycle authority. Persisted without the
   * route, key, request media, or any other secret. */
  durableBatch?: {
    clientBatchId: string;
    expectedInstanceId: string;
    serverBatchId: string | null;
    childIndex: number;
  };
  /** A complete durable outcome whose exact media read failed transiently.
   * Authority remains `done`; a later lifecycle reconciliation retries. */
  mediaHydrationError?: string | null;
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

/**
 * Whether a terminal SSE error frame says the host KEPT this generation.
 * A durable-queue host ends the stream of a job it retained across a restart
 * with `{ retained: true, code: "server_restarting" }`; the work is still
 * queued there and will land in that host's gallery. Older servers never send
 * the field, so an absent or malformed body stays a hard failure.
 */
function terminalErrorRetained(body: string | undefined): boolean {
  if (!body) return false;
  try {
    const parsed = JSON.parse(body) as { retained?: unknown };
    return parsed.retained === true;
  } catch {
    return false;
  }
}

/** How long the post-close durability lookup may take before the job settles
 *  as an ordinary failure. The socket just died, so the host may well be gone;
 *  the row must not sit in `running` waiting on a request that never answers. */
const RETENTION_LOOKUP_TIMEOUT_MS = 4_000;

/**
 * Per-job durability, captured when the `queued` event arrives — while the row
 * demonstrably exists — and read later if the stream dies.
 *
 * Asking AFTER the close is a race the job can lose by succeeding: if the
 * transport drops once rendering has finished, the server has already removed
 * the row, and the absent row reads as "not durable" for output that is
 * sitting in the Library. Keyed by client job id, outside Pinia state (it is
 * plumbing, and the value is a promise).
 */
const durabilityByJob = new Map<string, Promise<boolean>>();

/** Take the host's answer while the row is still there to be asked about. */
function captureJobDurability(
  job: Job,
  target: StreamTarget | undefined,
): void {
  if (!job.serverId || durabilityByJob.has(job.id)) return;
  const answer = hostJournalledJob(job.serverId, target);
  durabilityByJob.set(job.id, answer);
  // Mirror it onto the job so consumers that never see this map — queue
  // reconciliation above all — can tell "the host kept it" from "it vanished".
  void answer.then((durable) => {
    if (durable) job.durable = true;
  });
}

/**
 * Whether the host's queue still holds `serverId` AND journalled it.
 *
 * Deliberately per-job rather than `capabilities.queue.durable_queue`: a host
 * that can promise durability still reports `durable: false` for a job it
 * excluded at admission — no gallery target, reference-upload media, or a
 * request over the journal's payload ceiling, which a large inline base64
 * source image can reach. Any failure resolves `false`, so an unreachable
 * host, a vanished row, or an older server all keep the hard failure.
 */
async function hostJournalledJob(
  serverId: string,
  target: StreamTarget | undefined,
): Promise<boolean> {
  try {
    const signal = AbortSignal.timeout(RETENTION_LOOKUP_TIMEOUT_MS);
    let listing = await fetchQueue(target, signal);
    const seenCursors = new Set<string>();
    for (;;) {
      const entry = mergeQueueEntries(
        listing.entries,
        listing.live_only_entries ?? [],
      ).find((candidate) => candidate.id === serverId);
      if (entry) return entry.durable === true;
      const cursor = listing.page?.next_cursor;
      const limit = listing.page?.limit;
      if (!cursor || !limit) return false;
      if (seenCursors.has(cursor)) return false;
      seenCursors.add(cursor);
      listing = await fetchQueue(target, signal, { limit, cursor });
    }
  } catch {
    return false;
  }
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
    output_mode: "one-shot",
    model: req.model,
    // A request long enough to auto-chain still produces ONE print, so its
    // title and its creation-time filing ride along to the stitched output.
    title: req.title ?? undefined,
    tags: req.tags,
    collection: req.collection,
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
  /** Admit sibling requests in one durable parent when the frozen host
   * supports it; otherwise preserve the legacy per-job stream behavior. */
  submitBatch: (
    requests: readonly GenerateRequestWire[],
    decision?: ChainRoutingDecision,
    route?: HostRoute | null,
  ) => string[];
  cancel: (id: string) => Promise<void>;
  /** Settle a still-running job as failed. Used by external liveness
   * authorities such as queue reconciliation so every failure updates the
   * canvas owner and terminal metadata through the same path. */
  failRunning: (id: string, error: string) => void;
  /** Settle a detached (rehydrated after reload) job whose server record is
   * gone. The outcome is genuinely unknown — it may well have finished — so
   * this records a dismissible note WITHOUT seizing the canvas with a
   * "Generation failed" takeover. */
  settleDetached: (id: string, note: string) => void;
  clearDone: () => void;
  /** Remove a specific job from the list (used to dismiss persisted cards). */
  remove: (id: string) => void;
  select: (id: string | null) => void;
}

const STORAGE_KEY = "mold.generate.jobs";

/// Maximum number of *settled* (done/error/canceled) jobs we keep in the
/// ordinary localStorage rail. Actionable durable batches use independently
/// projected recovery records; legacy running jobs remain in this rail.
/// Past this cap, oldest settled jobs are forgotten on the next persist
/// cycle. Completed media itself lives in the gallery DB.
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
  /** A settled row whose fate the host owns. Persisted so a reload cannot
   * resurrect it as an ordinary failure. */
  detached?: boolean;
  durableBatch?: Job["durableBatch"];
  cancelRequested?: boolean;
}

const JOB_STORAGE_VERSION = 1;
const DURABLE_RECOVERY_PREFIX = `${STORAGE_KEY}.recovery.`;

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

/** Job recovery keeps ordinary presentation settings only. Media bytes,
 * authorities, paths, reference provenance, and identity metadata remain
 * session memory and never enter localStorage. */
function persistenceSafeRequest(
  request: GenerateRequestWire | ChainRequestWire,
): GenerateRequestWire | ChainRequestWire {
  return redactGenerationMediaForPersistence(request);
}

/** Durable recovery uses the same privacy projection. Durable-eligible work
 * is media-free already; applying the fence again keeps recovery fail-closed. */
function durablePersistenceSafeRequest(
  request: GenerateRequestWire | ChainRequestWire,
): GenerateRequestWire | ChainRequestWire {
  return persistenceSafeRequest(request);
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
        // A row whose server id is known is NOT dead-lettered on boot: the
        // server may still be rendering it, and the queue reconciler can
        // prove that either way. Only a row that never received its queued
        // frame has nothing to reconcile against.
        !persisted.serverId &&
        !persisted.durableBatch &&
        (!newestZombie || persisted.startedAt > newestZombie.startedAt)
      ) {
        newestZombie = persisted;
      }
    }
    const canvasErrorJobId = newestZombie?.id ?? null;
    const loadedAt = Date.now();
    const loadedJobs = parsed.jobs.map((p) => {
      // A rehydrated RUNNING row with a known server id is detached because
      // the reconciler owns it; a settled row is detached only if it was
      // settled that way before the reload.
      const detached =
        (p.state === "running" &&
          (Boolean(p.serverId) || Boolean(p.durableBatch))) ||
        p.detached === true;
      const wasZombie = p.state === "running" && !detached;
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
        cancelling: p.cancelRequested === true && state === "running",
        cancelRequested: p.cancelRequested === true,
        // This boot just discovered that a formerly-running row lost its
        // stream, so keep its recovery row present and dismissible from now.
        // Genuinely settled history retains its original age.
        settledAt: wasZombie
          ? loadedAt
          : detached
            ? null
            : (p.settledAt ?? p.lastProgressAt ?? p.startedAt),
        chain: p.chain,
        // A detached job has no stream to be stale about until the
        // reconciler has had a chance to speak.
        lastProgressAt: detached ? loadedAt : p.lastProgressAt,
        detached,
        workStarted: p.workStarted,
        hostId: p.hostId,
        hostLabel: p.hostLabel,
        target: null,
        serverId: p.serverId,
        durable: p.durableBatch ? true : undefined,
        durableBatch: p.durableBatch,
        mediaHydrationError: null,
        streamStarted: false,
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

/** Read the per-batch recovery journal. Each key is independent, so admitting
 * the next job never serializes all older outstanding work. Malformed or
 * future records are ignored without poisoning the ordinary activity rail. */
function loadDurableRecoveryJobs(storage: Storage): Job[] {
  const recovered: Job[] = [];
  try {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (!key?.startsWith(DURABLE_RECOVERY_PREFIX)) continue;
      const loaded = loadPersistedJobs(storage.getItem(key));
      recovered.push(...loaded.filter((job) => Boolean(job.durableBatch)));
    }
  } catch {
    // Storage can be disabled wholesale in privacy mode. Admission remains a
    // server concern and must not be coupled to this recovery convenience.
  }
  return recovered;
}

function initializePersistedState(raw: string | null): LoadedJobsState {
  const loaded = loadPersistedState(raw);
  // The deep watcher is intentionally not immediate. Write a boot-created
  // dead letter through synchronously so a second refresh cannot rediscover
  // the same formerly-running row as a new current failure.
  if (loaded.canvasErrorJobId !== null) persistJobs(loaded.jobs);
  return loaded;
}

function persistedJobsJson(jobs: Job[]): string {
  // Durable actionable rows live in independent per-batch recovery records.
  // Keeping them out of this shared rail is the critical O(1) admission rule:
  // adding job N does not stringify jobs 1..N-1 again. Settled durable rows
  // may remain here briefly as ordinary presentation history.
  const settledCount = { n: 0 };
  const filtered = jobs.filter((j) => {
    if (
      j.durableBatch &&
      (j.state === "running" || (j.state === "done" && !j.result))
    ) {
      return false;
    }
    if (j.state === "running") return true;
    settledCount.n += 1;
    return settledCount.n <= SETTLED_HISTORY_CAP;
  });
  const serializable: PersistedJob[] = filtered.map((j) => ({
    id: j.id,
    request: j.durableBatch
      ? durablePersistenceSafeRequest(j.request)
      : persistenceSafeRequest(j.request),
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
    detached: j.detached === true,
    durableBatch: j.durableBatch,
    cancelRequested: j.cancelRequested === true,
  }));
  const payload: PersistedJobs = {
    version: JOB_STORAGE_VERSION,
    jobs: serializable,
  };
  return JSON.stringify(payload);
}

function recoveryKey(clientBatchId: string): string {
  return `${DURABLE_RECOVERY_PREFIX}${clientBatchId}`;
}

/** Persist only one actionable durable batch. The host is the admission and
 * terminal authority; browser quota/privacy failures never veto its work.
 * Records are deleted only after that batch has no unresolved outcome. */
function persistDurableRecoveryBatch(clientBatchId: string): boolean {
  const owned = jobsForClientBatch(clientBatchId);
  const actionable = owned.filter(
    (job) => job.state === "running" || (job.state === "done" && !job.result),
  );
  try {
    if (actionable.length === 0) {
      localStorage.removeItem(recoveryKey(clientBatchId));
      return true;
    }
    localStorage.setItem(
      recoveryKey(clientBatchId),
      persistedJobsJsonFor(actionable),
    );
    return true;
  } catch {
    return false;
  }
}

function persistedJobsJsonFor(source: readonly Job[]): string {
  const serializable: PersistedJob[] = source.map((j) => ({
    id: j.id,
    request: durablePersistenceSafeRequest(j.request),
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
    detached: j.detached === true,
    durableBatch: j.durableBatch,
    cancelRequested: j.cancelRequested === true,
  }));
  return JSON.stringify({ version: JOB_STORAGE_VERSION, jobs: serializable });
}

function persistJobs(jobs: Job[]) {
  try {
    localStorage.setItem(STORAGE_KEY, persistedJobsJson(jobs));
  } catch {
    /* Ordinary history is best-effort in quota / privacy mode. */
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
if (typeof localStorage !== "undefined") {
  const seen = new Set(initialPersistedState.jobs.map(({ id }) => id));
  for (const recovered of loadDurableRecoveryJobs(localStorage)) {
    if (!seen.has(recovered.id)) {
      seen.add(recovered.id);
      initialPersistedState.jobs.push(recovered);
    }
  }
}
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
  durabilityByJob.delete(job.id);
  job.settledAt = Date.now();
  canvasErrorJobId.value = null;
}

function recordFailedSettlement(job: Job) {
  durabilityByJob.delete(job.id);
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

function settleDetachedJob(id: string, note: string) {
  const job = jobs.value.find((candidate) => candidate.id === id);
  if (!job || job.state !== "running") return;
  durabilityByJob.delete(id);
  job.error = note;
  job.state = "error";
  job.settledAt = Date.now();
  job.previewUrl = null;
  // `error` is the only settled state the rail models, but this is NOT a
  // failure: the host owns the job's fate and it may already have landed in
  // the Library. The flag is what keeps the activity strip from labelling it
  // "Failed" for five minutes once the fleet stops listing it as active.
  job.detached = true;
  // Deliberately no canvasErrorJobId takeover: the job may have completed
  // successfully while the page was away — the note is advisory history.
}

/// Grace period before a successfully-completed job's running-strip card
/// is auto-dismissed. Keeps the thumbnail on screen long enough for the
/// user to register that "yes, the thing I asked for finished" before it
/// quietly drops away into the gallery feed below. Errored / canceled
/// jobs never auto-dismiss — they have nothing in the gallery to fall
/// back to and the user may want to re-read the error.
const AUTO_REMOVE_DONE_MS = 1500;

/** Schedule auto-removal of a successfully-completed job. The timer
 * re-checks `state` at fire time so any later terminal reconciliation
 * remains authoritative. Safe to call for jobs that have already been
 * manually dismissed: `removeJob` filters by id, so a missing id is a
 * no-op. */
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
const streamSlots = new TargetStreamSlots(4);

function streamTargetKey(route: HostRoute | null): string {
  if (route?.target.baseUrl) return route.target.baseUrl;
  return typeof window === "undefined" ? "__origin__" : window.location.origin;
}

export const __testing__ = {
  AUTO_REMOVE_DONE_MS,
  STALE_THRESHOLD_MS,
  loadPersistedJobs,
  loadPersistedState,
  loadDurableRecoveryJobs,
  initializePersistedState,
  persistJobs,
  STORAGE_KEY,
  durableRequestIneligibility,
  durablePersistenceSafeRequest,
  reconcileDurableHost,
  handleDurableEvent,
  resetDurableLifecycleForTests,
};

function createJobRecord(
  req: GenerateRequestWire | ChainRequestWire,
  decision: ChainRoutingDecision,
  route: HostRoute | null,
  durableBatch?: Job["durableBatch"],
): Job {
  const now = Date.now();
  return reactive<Job>({
    id: createUuid(),
    request: req,
    startedAt: now,
    controller: new AbortController(),
    progress: emptyProgress(),
    result: null,
    error: null,
    state: "running",
    cancelling: false,
    cancelRequested: false,
    settledAt: null,
    chain:
      decision.kind === "chain"
        ? {
            stageCount: decision.stageCount,
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
    streamStarted: false,
    previewUrl: null,
    seedVisual: seedVisualFor(req),
    ...(durableBatch ? { durable: true, durableBatch } : {}),
    mediaHydrationError: null,
  }) as Job;
}

const durableTrackers = new Map<string, GenerationBatchTracker>();
const durableJobsByBatch = new Map<string, Job[]>();
const durableRoutes = new Map<string, HostRoute>();
const durableEffectKeys = new Set<string>();
const durableEventSessions = new Map<
  string,
  { signature: string; controller: AbortController }
>();
const durableReconciliations = new Map<string, Promise<void>>();
const durableReconciliationPending = new Map<string, Set<string> | null>();
/** Server instance signatures that proved they emit post-commit lifecycle hints. */
const durablePostCommitSignatures = new Map<string, string>();
const durableHydrations = new Map<string, Promise<void>>();
const durableCancellations = new Map<string, Promise<void>>();
const durableGallerySnapshots = new Map<string, Promise<GalleryImage[]>>();
const durableGalleryRows = new Map<string, GalleryImage>();
let durableRecoveryStarted = false;
let durableWakeListenersInstalled = false;

function routeSignature(route: HostRoute): string {
  return JSON.stringify([
    route.hostId,
    route.target.baseUrl,
    route.target.apiKey ?? null,
    route.instanceId ?? null,
  ]);
}

function durableRequestIneligibility(
  request: GenerateRequestWire | ChainRequestWire,
  decision: ChainRoutingDecision,
  route: HostRoute | null,
): string | null {
  if (decision.kind === "chain" || isPrebuiltChainRequest(request)) {
    return "sequences use their dedicated durable lifecycle";
  }
  if (!route || !supportsDurableGenerationLifecycle(route.durableGeneration)) {
    return "the host does not advertise durable generation outcomes";
  }
  if (!route.instanceId) return "the host instance is unknown";
  const generation = request as GenerateRequestWire;
  if (isMinimaxH3Identity(route.modelFamily, generation.model)) {
    return "MiniMax H3 replay authority is intentionally private";
  }
  if (
    !supportsDurableRequest(
      route.durableGeneration,
      route.durableMedia,
      generation,
    )
  ) {
    return "the host does not advertise encrypted durable support for this request";
  }
  return null;
}

function routeApiTarget(route: HostRoute): ApiTarget {
  return {
    baseUrl: route.target.baseUrl,
    apiKey: route.target.apiKey ?? null,
  };
}

function jobsForClientBatch(clientBatchId: string): Job[] {
  const indexed = durableJobsByBatch.get(clientBatchId);
  if (indexed) return indexed;
  const recovered = jobs.value.filter(
    (job) => job.durableBatch?.clientBatchId === clientBatchId,
  );
  if (recovered.length > 0) durableJobsByBatch.set(clientBatchId, recovered);
  return recovered;
}

function errorText(value: unknown): string {
  if (typeof value === "string" && value.trim()) return value;
  if (value instanceof Error && value.message) return value.message;
  if (value == null) return "generation failed";
  try {
    return JSON.stringify(value) || "generation failed";
  } catch {
    return String(value);
  }
}

function durableHostKey(job: Job): string {
  return job.hostId ?? ORIGIN_HOST_ID;
}

function durableGalleryRowKey(hostId: string, filename: string): string {
  return `${hostId.length}:${hostId}|${filename.length}:${filename}`;
}

function galleryRowFromUnknown(
  value: unknown,
  filename: string,
): GalleryImage | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Partial<GalleryImage>;
  if (
    row.filename !== filename ||
    !row.metadata ||
    typeof row.metadata !== "object"
  ) {
    return null;
  }
  return row as GalleryImage;
}

function sharedGallerySnapshot(job: Job): Promise<GalleryImage[]> {
  const key = durableHostKey(job);
  const existing = durableGallerySnapshots.get(key);
  if (existing) return existing;
  const request = listGalleryFrom(routeForDetachedJob(job));
  durableGallerySnapshots.set(key, request);
  return request;
}

async function galleryRowForCompletion(
  job: Job,
  filename: string,
): Promise<GalleryImage> {
  const hostId = durableHostKey(job);
  const cached = durableGalleryRows.get(durableGalleryRowKey(hostId, filename));
  if (cached) return cached;
  const listing = await sharedGallerySnapshot(job);
  const row = listing.find((candidate) => candidate.filename === filename);
  if (!row) {
    throw new Error(
      `completed output '${filename}' is not in the host gallery`,
    );
  }
  durableGalleryRows.set(durableGalleryRowKey(hostId, filename), row);
  return row;
}

interface WavFacts {
  sampleRate: number;
  channels: number;
  durationMs: number;
}

/** Read only the RIFF chunk table needed by the typed audio completion. */
async function wavFacts(blob: Blob): Promise<WavFacts | null> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const ascii = (offset: number, length: number) =>
    String.fromCharCode(...bytes.subarray(offset, offset + length));
  if (bytes.length < 12 || ascii(0, 4) !== "RIFF" || ascii(8, 4) !== "WAVE") {
    return null;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 12;
  let sampleRate: number | null = null;
  let channels: number | null = null;
  let byteRate: number | null = null;
  let dataBytes: number | null = null;
  while (offset + 8 <= bytes.length) {
    const name = ascii(offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    if (body + size > bytes.length) break;
    if (name === "fmt " && size >= 16) {
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      byteRate = view.getUint32(body + 8, true);
    } else if (name === "data") {
      dataBytes = size;
    }
    offset = body + size + (size % 2);
  }
  if (!sampleRate || !channels || !byteRate || dataBytes === null) return null;
  return {
    sampleRate,
    channels,
    durationMs: Math.ceil((dataBytes * 1000) / byteRate),
  };
}

async function durableCompletionResult(
  job: Job,
  lifecycle: GenerationLifecycleJob,
): Promise<SseCompleteEvent> {
  const filename = lifecycle.result?.filename;
  if (!filename) {
    throw new Error("the host completed this print without an output filename");
  }
  const target = routeForDetachedJob(job);
  const print = await galleryRowForCompletion(job, filename);
  const host = getHost(job.hostId ?? ORIGIN_HOST_ID) ?? {
    id: job.hostId ?? ORIGIN_HOST_ID,
    name: job.hostLabel ?? job.hostId ?? "this server",
    url: target?.baseUrl ?? "",
    ...(target?.apiKey ? { apiKey: target.apiKey } : {}),
  };
  const artifact = await fetchGalleryBlob(host, filename);
  const image = await blobToBase64(artifact);
  const metadata = print.metadata;
  const format =
    print.format ?? metadata.output_format ?? inferFormatFromName(filename);
  if (!format) throw new Error(`completed output '${filename}' has no format`);
  const kind =
    format === "mp4" ? "video" : format === "wav" ? "audio" : "image";
  let thumbnail: string | null = null;
  if (kind !== "image") {
    thumbnail = await fetchGalleryThumbnailBlob(host, filename)
      .then(blobToBase64)
      .catch(() => null);
  }
  let originalImage: string | null = null;
  if (lifecycle.result?.originalFilename) {
    originalImage = await blobToBase64(
      await fetchGalleryBlob(host, lifecycle.result.originalFilename),
    );
  }
  const request = job.request as GenerateRequestWire;
  const wav = kind === "audio" ? await wavFacts(artifact) : null;
  if (kind === "audio" && !wav) {
    throw new Error(`completed output '${filename}' is not a valid WAV`);
  }
  return {
    image,
    format: format as OutputFormat,
    width: metadata.width,
    height: metadata.height,
    seed_used: metadata.seed,
    generation_time_ms: Math.max(
      0,
      (lifecycle.completedAtMs ?? lifecycle.version.updatedAtMs) -
        lifecycle.createdAtMs,
    ),
    model: metadata.model,
    ...(originalImage ? { original_image: originalImage } : {}),
    ...(kind === "video"
      ? {
          video_frames: metadata.frames ?? request.frames ?? null,
          video_fps: metadata.fps ?? request.fps ?? null,
          ...(thumbnail ? { video_thumbnail: thumbnail } : {}),
        }
      : {}),
    ...(kind === "audio"
      ? {
          // `isAudioCompletion` keys on sample-rate presence, matching the
          // live SSE contract. A valid WAV yields exact header-derived facts.
          audio_sample_rate: wav!.sampleRate,
          audio_channels: wav!.channels,
          audio_duration_ms: wav!.durationMs,
          ...(thumbnail ? { audio_thumbnail: thumbnail } : {}),
        }
      : {}),
  };
}

function settleDurableTerminal(
  job: Job,
  lifecycle: GenerationLifecycleJob,
): void {
  if (durableEffectKeys.has(lifecycle.key)) return;
  const isReloadedCompletion =
    lifecycle.phase === "complete" && job.state === "done" && !job.result;
  if (job.state !== "running" && !isReloadedCompletion) return;
  if (lifecycle.phase === "cancelled") {
    durableEffectKeys.add(lifecycle.key);
    job.state = "canceled";
    job.cancelling = false;
    job.cancelRequested = false;
    job.settledAt = lifecycle.completedAtMs ?? Date.now();
    job.previewUrl = null;
    return;
  }
  if (lifecycle.phase === "failed") {
    durableEffectKeys.add(lifecycle.key);
    job.cancelling = false;
    job.cancelRequested = false;
    job.error = lifecycle.error ?? errorText(lifecycle.terminalError);
    recordFailedSettlement(job);
    return;
  }
  if (lifecycle.phase !== "complete") return;

  // Claim the terminal transition before fetching media. A concurrent cancel
  // sees `done` and cannot replace a success the host already made durable.
  if (job.state === "running") {
    job.state = "done";
    recordSuccessfulSettlement(job);
  }
  job.previewUrl = null;
  job.cancelling = false;
  job.cancelRequested = false;
  if (durableHydrations.has(lifecycle.key)) return;
  const controller = new AbortController();
  const targetKey = routeForDetachedJob(job)?.baseUrl ?? "__origin__";
  // Use the same per-target connection authority as attached generation
  // streams. Artifact hydration cannot create a second independent pool.
  const hydration = streamSlots
    .acquire(targetKey, controller.signal)
    .then(async (release) => {
      if (!release) return;
      try {
        const result = await durableCompletionResult(job, lifecycle);
        if (job.state !== "done") return;
        job.result = result;
        job.mediaHydrationError = null;
        job.error = null;
        if (lifecycle.result?.filename) {
          durableGalleryRows.delete(
            durableGalleryRowKey(
              durableHostKey(job),
              lifecycle.result.filename,
            ),
          );
        }
        durableEffectKeys.add(lifecycle.key);
        fireComplete(job);
        scheduleAutoRemoveOnDone(job.id);
        if (job.durableBatch) {
          pruneDurableTrackerIfSettled(job.durableBatch.clientBatchId);
        }
      } finally {
        release();
      }
    })
    .catch((error) => {
      if (job.state !== "done") return;
      // The durable outcome remains complete. Keep its tracker/actionable
      // recovery record so the next exact lifecycle reconciliation retries.
      job.mediaHydrationError = errorText(error);
      job.progress.stage = "Completed · media temporarily unavailable";
      if (job.durableBatch) {
        persistDurableRecoveryBatch(job.durableBatch.clientBatchId);
      }
    })
    .finally(() => {
      if (durableHydrations.get(lifecycle.key) === hydration) {
        durableHydrations.delete(lifecycle.key);
      }
    });
  durableHydrations.set(lifecycle.key, hydration);
}

function applyDurableTracker(tracker: GenerationBatchTracker): void {
  const ownedJobs = jobsForClientBatch(tracker.clientBatchId);
  if (tracker.reconciliation.reason === "instance_mismatch") {
    for (const job of ownedJobs) {
      settleDetachedJob(
        job.id,
        "This machine was replaced. The previous server instance still owns this print's outcome.",
      );
    }
    return;
  }
  if (
    tracker.reconciliation.reason === "missing" ||
    tracker.reconciliation.reason === "batch_mismatch"
  ) {
    for (const job of ownedJobs) {
      settleDetachedJob(
        job.id,
        "The durable generation record could not be reconciled on its original machine.",
      );
    }
    return;
  }
  for (const lifecycle of Object.values(tracker.jobs)) {
    const job = ownedJobs.find(
      (candidate) =>
        candidate.durableBatch?.childIndex === lifecycle.childIndex,
    );
    if (!job) continue;
    job.serverId = lifecycle.authority.jobId;
    if (job.durableBatch) {
      job.durableBatch.serverBatchId = lifecycle.authority.batchId;
    }
    job.lastProgressAt = Math.max(
      job.lastProgressAt,
      lifecycle.version.updatedAtMs,
    );
    if (lifecycle.phase === "running") {
      markWorkStarted(job);
      job.progress.stage = "Developing";
    } else if (lifecycle.phase === "held") {
      job.progress.stage = "Queued · waiting for an available lane";
      job.workStarted = false;
    } else if (lifecycle.phase === "accepted" || lifecycle.phase === "queued") {
      job.progress.stage = "Queued";
      job.workStarted = false;
    } else {
      settleDurableTerminal(job, lifecycle);
    }
    if (
      job.cancelRequested &&
      (lifecycle.phase === "accepted" ||
        lifecycle.phase === "queued" ||
        lifecycle.phase === "held" ||
        lifecycle.phase === "running")
    ) {
      void confirmDurableCancellation(job).catch(() => undefined);
    }
  }
  persistDurableRecoveryBatch(tracker.clientBatchId);
  pruneDurableTrackerIfSettled(tracker.clientBatchId);
}

function pruneDurableTrackerIfSettled(clientBatchId: string): void {
  const tracker = durableTrackers.get(clientBatchId);
  if (!tracker) return;
  const owned = jobsForClientBatch(clientBatchId);
  if (
    owned.length === 0 ||
    owned.some(
      (job) => job.state === "running" || (job.state === "done" && !job.result),
    )
  )
    return;
  durableTrackers.delete(clientBatchId);
  durableJobsByBatch.delete(clientBatchId);
  for (const lifecycle of Object.values(tracker.jobs)) {
    durableEffectKeys.delete(lifecycle.key);
  }
  try {
    localStorage.removeItem(recoveryKey(clientBatchId));
  } catch {
    // Privacy mode can disable deletion as well as writes. The stale record is
    // harmless: the next authoritative recovery snapshot settles it again.
  }
  if (
    [...durableTrackers.values()].some((row) => row.hostId === tracker.hostId)
  ) {
    return;
  }
  durableRoutes.delete(tracker.hostId);
  durableEventSessions.get(tracker.hostId)?.controller.abort();
  durableEventSessions.delete(tracker.hostId);
}

function applyDurableBatchStatus(
  clientBatchId: string,
  batch: GenerationBatchStatus,
): void {
  const current = durableTrackers.get(clientBatchId);
  if (!current) return;
  const next = reduceGenerationLifecycle(current, {
    type: "batch_snapshot",
    batch,
  });
  durableTrackers.set(clientBatchId, next);
  applyDurableTracker(next);
}

async function recoverAmbiguousAdmission(
  route: HostRoute,
  clientBatchId: string,
): Promise<void> {
  try {
    const lookup = await lookupGenerationBatchByClientId(
      routeApiTarget(route),
      clientBatchId,
    );
    if (lookup.kind === "found") {
      applyDurableBatchStatus(clientBatchId, lookup.batch);
      return;
    }
    const tracker = durableTrackers.get(clientBatchId);
    if (tracker) {
      durableTrackers.set(
        clientBatchId,
        reduceGenerationLifecycle(tracker, { type: "lookup_missing" }),
      );
    }
    for (const job of jobsForClientBatch(clientBatchId)) {
      if (job.state === "running") {
        job.detached = true;
        job.progress.stage = "Confirming durable admission";
      }
    }
  } catch {
    // The redacted client UUID remains persisted; reconnect/wake reconciliation
    // retries against that same authority and never submits a second job.
  }
}

async function admitDurableBatch(
  route: HostRoute,
  clientBatchId: string,
  requests: readonly GenerateRequestWire[],
): Promise<void> {
  try {
    const batch = await admitGenerationBatch(routeApiTarget(route), {
      client_batch_id: clientBatchId,
      requests: requests.map((request) => ({ ...request, batch_size: 1 })),
    });
    applyDurableBatchStatus(clientBatchId, batch);
  } catch (error) {
    const tracker = durableTrackers.get(clientBatchId);
    if (tracker) {
      durableTrackers.set(
        clientBatchId,
        reduceGenerationLifecycle(tracker, {
          type: "admission_uncertain",
          error: errorText(error),
        }),
      );
    }
    const lookup = await lookupGenerationBatchByClientId(
      routeApiTarget(route),
      clientBatchId,
    ).catch(() => null);
    if (lookup?.kind === "found") {
      applyDurableBatchStatus(clientBatchId, lookup.batch);
      return;
    }
    if (error instanceof ApiError && lookup?.kind === "missing") {
      const current = durableTrackers.get(clientBatchId);
      if (current) {
        durableTrackers.set(
          clientBatchId,
          reduceGenerationLifecycle(current, {
            type: "admission_rejected",
            error: error.message,
          }),
        );
      }
      for (const job of jobsForClientBatch(clientBatchId)) {
        if (job.state !== "running") continue;
        job.error = error.message;
        recordFailedSettlement(job);
      }
      return;
    }
    await recoverAmbiguousAdmission(route, clientBatchId);
  }
}

function submitDurableJobs(
  requests: readonly GenerateRequestWire[],
  decision: ChainRoutingDecision,
  route: HostRoute,
): string[] {
  selectedJobId.value = null;
  canvasErrorJobId.value = null;
  const clientBatchId = createUuid();
  const tracker = createGenerationBatchTracker({
    hostId: route.hostId,
    expectedInstanceId: route.instanceId!,
    clientBatchId,
    submittedAtMs: Date.now(),
  });
  const admitted = requests.map((request, offset) =>
    createJobRecord(request, decision, route, {
      clientBatchId,
      expectedInstanceId: route.instanceId!,
      serverBatchId: null,
      childIndex: offset + 1,
    }),
  );
  jobs.value = [...admitted, ...jobs.value];
  durableJobsByBatch.set(clientBatchId, admitted);
  // Journal only THIS batch before dispatch. The ordinary rail's debounced
  // presentation write is deliberately outside admission, so submitting job
  // N never projects or serializes jobs 1..N-1.
  persistDurableRecoveryBatch(clientBatchId);
  durableTrackers.set(clientBatchId, tracker);
  durableRoutes.set(route.hostId, {
    ...route,
    target: { ...route.target },
  });
  ensureDurableEventSession(route);
  void admitDurableBatch(route, clientBatchId, requests);
  return admitted.map((job) => job.id);
}

async function runDurableReconciliation(
  hostId: string,
  clientBatchIds?: ReadonlySet<string>,
): Promise<void> {
  const route = durableRoutes.get(hostId);
  if (!route) return;
  // One authoritative gallery snapshot is shared by every legacy completion
  // in this REST reconciliation wave. The next wave invalidates it.
  durableGallerySnapshots.delete(hostId);
  const current = [...durableTrackers.values()].filter(
    (tracker) =>
      tracker.hostId === hostId &&
      (!clientBatchIds || clientBatchIds.has(tracker.clientBatchId)),
  );
  const request = buildGenerationBatchStatusRequest(current, hostId);
  if (request.client_batch_ids.length === 0 && !request.batch_ids?.length)
    return;
  const response = await reconcileGenerationBatches(
    routeApiTarget(route),
    request,
  );
  const merged = mergeBulkGenerationBatchResponse(current, hostId, response);
  for (const tracker of merged.trackers) {
    durableTrackers.set(tracker.clientBatchId, tracker);
    if (tracker.hostId === hostId) applyDurableTracker(tracker);
  }
}

function startDurableReconciliation(
  hostId: string,
  scope?: ReadonlySet<string>,
): Promise<void> {
  const active = durableReconciliations.get(hostId);
  if (active) {
    const pending = durableReconciliationPending.get(hostId);
    if (scope === undefined) {
      durableReconciliationPending.set(hostId, null);
    } else if (pending !== null) {
      const next = pending ?? new Set<string>();
      for (const clientBatchId of scope) next.add(clientBatchId);
      durableReconciliationPending.set(hostId, next);
    }
    return active;
  }
  const task = runDurableReconciliation(hostId, scope)
    .catch(() => undefined)
    .finally(() => {
      durableReconciliations.delete(hostId);
      const pending = durableReconciliationPending.get(hostId);
      if (pending !== undefined) {
        durableReconciliationPending.delete(hostId);
        if (pending === null) {
          void startDurableReconciliation(hostId);
        } else {
          void startDurableReconciliation(hostId, pending);
        }
      }
    });
  durableReconciliations.set(hostId, task);
  return task;
}

function reconcileDurableHost(
  hostId: string,
  clientBatchId?: string,
): Promise<void> {
  return startDurableReconciliation(
    hostId,
    clientBatchId === undefined ? undefined : new Set([clientBatchId]),
  );
}

function handleDurableEvent(
  hostId: string,
  eventName: string,
  rawData: string,
): void {
  const route = durableRoutes.get(hostId);
  if (!route) return;
  let data: Record<string, unknown>;
  try {
    const parsed = JSON.parse(rawData) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("malformed event");
    }
    data = parsed as Record<string, unknown>;
  } catch {
    // Malformed hints carry no authority and no safe identity to target. The
    // stream's explicit resync frame or wake/open snapshot performs repair.
    return;
  }
  if (eventName === "authority" || eventName === "resync_required") {
    // Event data is only a hint. Even an authority mismatch must be confirmed
    // by the bulk REST authority before it can detach locally tracked work.
    void reconcileDurableHost(hostId);
    return;
  }
  if (eventName !== "event" || typeof data.type !== "string") {
    return;
  }
  let exactJob: Job | undefined;
  if (typeof data.id === "string") {
    exactJob = jobs.value.find(
      (candidate) =>
        candidate.hostId === hostId && candidate.serverId === data.id,
    );
  }
  if (data.type === "job_state_committed") {
    // Normal settlements reconcile only their owning batch. Admission and
    // execution are concurrent, so an event whose id is not mapped yet must
    // still fall back to the coalesced host-wide authority read.
    durablePostCommitSignatures.set(hostId, routeSignature(route));
    void reconcileDurableHost(hostId, exactJob?.durableBatch?.clientBatchId);
    return;
  }
  if (data.type === "generation_states_committed") {
    // Bulk cancellation deliberately emits one host-wide post-commit hint.
    durablePostCommitSignatures.set(hostId, routeSignature(route));
    void reconcileDurableHost(hostId);
    return;
  }
  if (data.type === "job_started" && typeof data.id === "string") {
    if (exactJob?.state === "running") {
      markWorkStarted(exactJob);
      exactJob.progress.stage = "Developing";
      if (typeof data.gpu === "number") exactJob.progress.gpu = data.gpu;
    }
  } else if (data.type === "job_queued" && typeof data.id === "string") {
    if (exactJob?.state === "running") exactJob.progress.stage = "Queued";
  } else if (
    data.type === "gallery_added" &&
    typeof data.filename === "string"
  ) {
    const row = galleryRowFromUnknown(data.image, data.filename);
    if (row) {
      const jobId = row.metadata.job_id;
      if (jobId) {
        exactJob = jobs.value.find(
          (candidate) =>
            candidate.hostId === hostId && candidate.serverId === jobId,
        );
      }
      if (exactJob) {
        durableGalleryRows.set(
          durableGalleryRowKey(hostId, data.filename),
          row,
        );
      }
    }
    // Older servers need gallery invalidation as their last completion hint.
    // Once this exact server instance proves it emits the ordered commit hint,
    // reconciling here would duplicate every completion read.
    if (durablePostCommitSignatures.get(hostId) === routeSignature(route))
      return;
  }
  if (exactJob?.durableBatch) {
    void reconcileDurableHost(hostId, exactJob.durableBatch.clientBatchId);
  }
}

function ensureDurableEventSession(route: HostRoute): void {
  if (route.eventsAvailable === false) return;
  const signature = routeSignature(route);
  const existing = durableEventSessions.get(route.hostId);
  if (
    existing?.signature === signature &&
    !existing.controller.signal.aborted
  ) {
    return;
  }
  existing?.controller.abort();
  const controller = new AbortController();
  durableEventSessions.set(route.hostId, { signature, controller });
  void fetchEventSource(`${route.target.baseUrl}/api/events`, {
    method: "GET",
    headers: Object.fromEntries(apiHeaders(routeApiTarget(route)).entries()),
    signal: controller.signal,
    openWhenHidden: true,
    onopen: async (response) => {
      if (!response.ok) {
        const error = new Error(
          `events stream failed: ${response.status}`,
        ) as Error & { status?: number };
        error.status = response.status;
        throw error;
      }
      void reconcileDurableHost(route.hostId);
    },
    onmessage: (message) =>
      handleDurableEvent(route.hostId, message.event || "event", message.data),
    onclose: () => {
      if (!controller.signal.aborted) {
        void reconcileDurableHost(route.hostId);
        throw new Error("events stream closed");
      }
    },
    onerror: (error) => {
      void reconcileDurableHost(route.hostId);
      const status = (error as { status?: number }).status;
      if (status === 401 || status === 403 || status === 404) throw error;
    },
  }).catch(() => {
    const current = durableEventSessions.get(route.hostId);
    if (current?.controller === controller) {
      durableEventSessions.delete(route.hostId);
    }
  });
}

function recoverDurableLifecycle(): void {
  if (durableRecoveryStarted) return;
  durableRecoveryStarted = true;
  const grouped = new Map<string, Job[]>();
  for (const job of jobs.value) {
    if (
      !job.durableBatch ||
      (job.state === "done" && job.result) ||
      job.state === "canceled"
    ) {
      continue;
    }
    const group = grouped.get(job.durableBatch.clientBatchId) ?? [];
    group.push(job);
    grouped.set(job.durableBatch.clientBatchId, group);
  }
  for (const [clientBatchId, owned] of grouped) {
    durableJobsByBatch.set(clientBatchId, owned);
    const first = owned[0]!;
    const durable = first.durableBatch!;
    const hostId = first.hostId ?? ORIGIN_HOST_ID;
    let tracker = createGenerationBatchTracker({
      hostId,
      expectedInstanceId: durable.expectedInstanceId,
      clientBatchId,
      submittedAtMs: first.startedAt,
    });
    if (durable.serverBatchId) {
      tracker = { ...tracker, serverBatchId: durable.serverBatchId };
    }
    durableTrackers.set(clientBatchId, tracker);
    const target = routeForDetachedJob(first);
    const route: HostRoute = {
      hostId,
      label: first.hostLabel ?? hostId,
      target: target
        ? { ...target }
        : {
            baseUrl:
              typeof window === "undefined" ? "" : window.location.origin,
          },
      instanceId: durable.expectedInstanceId,
      durableGeneration: {
        heterogeneous_batch: true,
        durable_batch_outcomes: true,
      },
      eventsAvailable: true,
    };
    durableRoutes.set(hostId, route);
    ensureDurableEventSession(route);
    void reconcileDurableHost(hostId);
  }
}

function installDurableWakeListeners(): void {
  if (durableWakeListenersInstalled || typeof window === "undefined") return;
  durableWakeListenersInstalled = true;
  const reconcileAll = () => {
    for (const hostId of durableRoutes.keys())
      void reconcileDurableHost(hostId);
  };
  window.addEventListener("pageshow", reconcileAll);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") reconcileAll();
  });
}

function resetDurableLifecycleForTests(): void {
  for (const session of durableEventSessions.values()) {
    session.controller.abort();
  }
  durableEventSessions.clear();
  durableTrackers.clear();
  durableJobsByBatch.clear();
  durableRoutes.clear();
  durableEffectKeys.clear();
  durableReconciliations.clear();
  durableReconciliationPending.clear();
  durablePostCommitSignatures.clear();
  durableHydrations.clear();
  durableCancellations.clear();
  durableGallerySnapshots.clear();
  durableGalleryRows.clear();
  durableRecoveryStarted = false;
}

function submitJobs(
  requests: readonly GenerateRequestWire[],
  decision: ChainRoutingDecision = { kind: "single" },
  route: HostRoute | null = null,
): string[] {
  if (
    requests.length > 0 &&
    requests.every(
      (request) =>
        durableRequestIneligibility(request, decision, route) === null,
    )
  ) {
    return submitDurableJobs(requests, decision, route!);
  }
  return requests.map((request) => submitJob(request, decision, route, false));
}

function submitJob(
  req: GenerateRequestWire | ChainRequestWire,
  decision: ChainRoutingDecision = { kind: "single" },
  route: HostRoute | null = null,
  allowDurable = true,
): string {
  if (
    allowDurable &&
    !isPrebuiltChainRequest(req) &&
    durableRequestIneligibility(req, decision, route) === null
  ) {
    return submitDurableJobs([req], decision, route!)[0]!;
  }
  selectedJobId.value = null;
  canvasErrorJobId.value = null;
  const job = createJobRecord(req, decision, route);
  const { id, controller } = job;
  jobs.value = [job, ...jobs.value];

  const onErrorCommon = (err: {
    kind: "http" | "network";
    status?: number;
    retryAfter?: number;
    body?: string;
    message?: string;
  }) => {
    // Terminal state wins. A confirmed cancellation (or a completion) can land
    // between the socket dying and this handler running, and a settled row must
    // not be re-opened by a late transport event — not even to carry its note.
    if (job.state !== "running") return;
    if (err.kind === "http") {
      const message = serverErrorMessage(err.body);
      job.error =
        err.status === 503
          ? `Queue full (retry after ${err.retryAfter ?? "?"}s)`
          : err.status === 0
            ? (message ?? "generation failed")
            : `HTTP ${err.status}${message ? `: ${message}` : ""}`;
      // A terminal frame is self-describing and per-job by construction: the
      // host sends `retained` only for a job it actually kept.
      if (err.status === 0 && terminalErrorRetained(err.body)) {
        settleDetachedJob(job.id, job.error ?? "");
        return;
      }
      recordFailedSettlement(job);
      return;
    }
    job.error = err.message ?? "network error";
    // The socket died with no terminal frame, so the host never got to say
    // whether it kept this job. Ask it — per job, because a host that can
    // promise durability still excludes some jobs at admission and a false
    // "it will finish" is worse than a failure the user can simply retry.
    void settleFramelessClose(job, job.error, req);
  };

  /** Resolve a frameless close against the answer captured at `queued` time.
   *  Anything short of a `durable: true` row — an unreachable host, a vanished
   *  row, an unadmitted job, a host that never promised durability — keeps the
   *  hard failure that has always been the behaviour here.
   *
   *  Deliberately NOT gated on the route's advertised `durable_queue`: a
   *  single-host page submits with no route at all (`normalizeSubmitRoute`
   *  collapses the origin to `null`), which is the common deployment, and
   *  requiring one would leave the default path permanently unreachable. The
   *  row's own `durable` flag is the stronger per-job answer anyway — only a
   *  host with the durable queue ever reports it — so asking for it directly
   *  is both correct for the origin and correct for a routed host whose
   *  capabilities were never read. */
  const settleFramelessClose = async (
    job: Job,
    note: string,
    request: GenerateRequestWire | ChainRequestWire,
  ) => {
    const durable =
      !!job.serverId &&
      !requestNeedsReferenceUpload(request as GenerateRequestWire) &&
      // Captured at `queued`, never asked now: by this point a job that
      // SUCCEEDED has already had its row removed, and asking would read that
      // absence as "not durable" for a print already in the Library.
      (await (durabilityByJob.get(job.id) ?? Promise.resolve(false)));
    // The lookup is asynchronous: a cancellation can be confirmed, or a late
    // completion can land, while it is in flight. Re-check before EITHER
    // settlement — replacing a confirmed cancellation with an error is as
    // wrong as replacing it with a detached note.
    if (job.state !== "running") return;
    if (durable) {
      // Soft settle: the row stops moving and keeps its note, but the canvas
      // never claims a print failed that the host is still going to render.
      settleDetachedJob(job.id, note);
      return;
    }
    recordFailedSettlement(job);
  };

  const startStream = async () => {
    if (decision.kind === "chain") {
      const chainReq = resolveChainRequest(req, decision);
      job.streamStarted = true;
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
          onRequestWarnings: surfaceRequestWarnings,
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
        job.streamStarted = true;
        await generateStream(
          transportRequest,
          {
            onProgress: (evt) => {
              applyProgress(job, evt);
              // `queued` is the first event the server emits per request, and
              // the one moment the durable row is guaranteed to be listable.
              if (evt.type === "queued")
                captureJobDurability(job, route?.target);
            },
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
            onRequestWarnings: surfaceRequestWarnings,
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
  streamSlots.schedule(streamTargetKey(route), controller.signal, (release) => {
    void startStream().finally(release);
  });

  return id;
}

/** The route for a job whose in-memory target died with its session (API
 * keys never persist): resolve the host back through the registry so cancel
 * reaches the machine that actually holds the job, not the origin. */
function routeForDetachedJob(job: Job): StreamTarget | undefined {
  if (job.target) return job.target;
  if (!job.hostId || job.hostId === ORIGIN_HOST_ID) return undefined;
  const host = getHost(job.hostId);
  if (!host) return undefined;
  const target: StreamTarget = { baseUrl: host.url };
  if (host.apiKey) target.apiKey = host.apiKey;
  return target;
}

function markCancellationConfirmed(job: Job): void {
  if (job.state !== "running") return;
  durabilityByJob.delete(job.id);
  job.controller.abort();
  job.state = "canceled";
  job.cancelling = false;
  job.cancelRequested = false;
  job.settledAt = Date.now();
  job.previewUrl = null;
  if (job.durableBatch) {
    persistDurableRecoveryBatch(job.durableBatch.clientBatchId);
    pruneDurableTrackerIfSettled(job.durableBatch.clientBatchId);
  }
  if (selectedJobId.value === job.id) selectedJobId.value = null;
}

async function confirmDurableCancellation(job: Job): Promise<void> {
  if (job.state !== "running" || !job.serverId) return;
  const active = durableCancellations.get(job.id);
  if (active) return active;
  job.cancelRequested = true;
  job.cancelling = true;
  if (job.durableBatch) {
    persistDurableRecoveryBatch(job.durableBatch.clientBatchId);
  }
  const task = cancelQueueJob(job.serverId, routeForDetachedJob(job))
    .then(() => {
      // Complete/failed/cancelled authority may have arrived during DELETE.
      if (job.state === "running") markCancellationConfirmed(job);
    })
    .catch((error) => {
      if (job.state === "running") {
        // Keep the intent and tracker. A later exact lifecycle snapshot can
        // expose the final outcome or retry the exact DELETE.
        job.cancelling = false;
        if (job.durableBatch) {
          persistDurableRecoveryBatch(job.durableBatch.clientBatchId);
        }
      }
      throw error;
    })
    .finally(() => {
      if (durableCancellations.get(job.id) === task) {
        durableCancellations.delete(job.id);
      }
    });
  durableCancellations.set(job.id, task);
  return task;
}

async function cancelJob(id: string): Promise<void> {
  const job = jobs.value.find((j) => j.id === id);
  if (!job || job.state !== "running") return;
  if (job.cancelling && !job.cancelRequested) return;
  if (job.durableBatch) {
    job.cancelRequested = true;
    job.cancelling = true;
    persistDurableRecoveryBatch(job.durableBatch.clientBatchId);
    if (!job.serverId) {
      job.progress.stage = "Cancelling when admission is confirmed";
      return;
    }
    return confirmDurableCancellation(job);
  }
  if (job.cancelling) return;
  job.cancelling = true;
  if (job.serverId) {
    try {
      await cancelQueueJob(job.serverId, routeForDetachedJob(job));
    } catch (error) {
      // Completion can race the DELETE. If the stream already settled, its
      // terminal frame is authoritative; otherwise cancellation was not
      // confirmed and the server-owned job must remain live locally.
      if (job.state !== "running") return;
      job.cancelling = false;
      throw error;
    }
  } else if (job.streamStarted) {
    job.cancelling = false;
    throw new Error(
      "Remote cancellation was not confirmed before the queue ID arrived.",
    );
  }
  // A completion snapshot can win while DELETE is in flight. Once the host's
  // durable outcome has claimed the job, a late successful cancel response
  // cannot replace it with a local cancellation.
  if (job.state !== "running") return;
  markCancellationConfirmed(job);
}

function clearDoneJobs() {
  jobs.value = jobs.value.filter((j) => j.state === "running");
  canvasErrorJobId.value = null;
}

function removeJob(id: string) {
  durabilityByJob.delete(id);
  const removed = jobs.value.find((job) => job.id === id);
  if (removed?.durableBatch) {
    const clientBatchId = removed.durableBatch.clientBatchId;
    const remaining = jobsForClientBatch(clientBatchId).filter(
      (job) => job.id !== id,
    );
    if (remaining.length > 0) durableJobsByBatch.set(clientBatchId, remaining);
    else durableJobsByBatch.delete(clientBatchId);
  }
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
  recoverDurableLifecycle();
  installDurableWakeListeners();
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
    submitBatch: submitJobs,
    cancel: cancelJob,
    failRunning: failRunningJob,
    settleDetached: settleDetachedJob,
    clearDone: clearDoneJobs,
    remove: removeJob,
    select: selectJob,
  };
}

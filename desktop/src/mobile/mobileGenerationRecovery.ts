import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { describeTransportError, isTransportFailure } from "../lib/api/errors";
import type {
  ChainJobDetail,
  CompleteEvent,
  GalleryImage,
  OutputFormat,
  OutputMetadata,
} from "../lib/api/types";
import { isCancelledError, metadataOnlyResult, type Job } from "../lib/generationJob";

/**
 * Foreground-resume reconciliation for iPhone generations.
 *
 * iOS freezes the WebView and tears down sockets while the app is
 * backgrounded, so every held generation stream dies with a raw WebKit
 * transport error even though the host kept working: a RUNNING job finishes
 * and lands in the host's gallery, a QUEUED job is skipped at dispatch once
 * its stream is gone. This module re-queries the job's frozen submission
 * route and settles each interrupted job with its true outcome — the
 * finished print, a re-attached live job, or a directed human failure —
 * before any raw transport text can reach the UI.
 *
 * Invariants: only the frozen submission target is queried (never a
 * re-resolved host), jobs settled by anything else (cancel, unmount) are
 * never overwritten, and prepared batches are neither resized nor rerouted —
 * reconciliation only reads the jobs it was handed.
 */

const STREAM_CLOSED_EARLY = "The generation stream closed before completion.";
const DEFAULT_POLL_INTERVAL_MS = 4_000;
/** Reconnection attempts after a reconcile query itself dies in transit.
 *  Resume fires the first query while iOS Wi-Fi may still be re-associating —
 *  the exact moment reconciliation exists for — so a dead query is retried
 *  (bounded) instead of settling a possibly-finished print as failed. */
const MAX_TRANSPORT_RETRIES = 3;

/** Minimal `GET /api/queue` row — declared locally so the remote-only phone
 * never imports a desktop store to borrow the shape. */
interface RecoveryQueueEntry {
  id: string;
  model?: string;
  state: "queued" | "running";
  seed_pinned?: boolean | null;
  metadata?: OutputMetadata | null;
}

export interface GenerationRecoveryOptions {
  /** The exact target the batch was submitted to — the frozen route. */
  target: ApiTarget;
  hostLabel: string;
  /** Whether this batch went through the durable long-video chain shim. */
  chain?: boolean;
  /** Reacquire a renderable URL for a job settled as complete. */
  refreshResultUrl: (clientId: number) => void;
  /** Abandon reconciliation when the owning view has unmounted. */
  isActive?: () => boolean;
  pollIntervalMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

/**
 * Whether a settled job error is the connection dying (iOS suspension, lost
 * Wi-Fi) rather than the host reporting a failure. Server-authored SSE error
 * copy and cancellations are never reconciled.
 */
export function isInterruptedGenerationError(error: string | null | undefined): boolean {
  if (!error || isCancelledError(error)) return false;
  return error === STREAM_CLOSED_EARLY || isTransportFailure(error);
}

/**
 * Match a job to the gallery row its completion produced. Every mobile
 * sibling ships an explicit seed (`base + i`), so seed + model is an exact
 * join; the recorded prompt is the tiebreaker for prepared siblings, whose
 * prompts are unique within a batch. Dimensions and steps must also agree
 * when both sides know them — a fixed-seed re-run at new settings whose
 * stream died pre-queue must not resurrect an older print.
 */
export function matchGalleryPrint(
  job: Pick<Job, "visualSeed" | "model" | "prompt" | "width" | "height" | "total">,
  prints: readonly GalleryImage[],
): GalleryImage | null {
  const seed = Number(job.visualSeed);
  if (!Number.isSafeInteger(seed)) return null;
  const differs = (recorded: number | null | undefined, submitted: number): boolean =>
    typeof recorded === "number" && submitted > 0 && recorded !== submitted;
  return (
    prints.find((print) => {
      const meta = print.metadata;
      if (!meta || meta.seed !== seed || meta.model !== job.model) return false;
      if (meta.prompt && job.prompt && meta.prompt !== job.prompt) return false;
      if (differs(meta.width, job.width) || differs(meta.height, job.height)) return false;
      if (differs(meta.steps, job.total)) return false;
      return true;
    }) ?? null
  );
}

function formatFromFilename(filename: string): OutputFormat | null {
  const ext = /\.([a-z0-9]+)$/i.exec(filename)?.[1]?.toLowerCase();
  if (!ext) return null;
  if (ext === "jpg") return "jpeg";
  return (["png", "jpeg", "webp", "gif", "apng", "mp4"] as const).find((f) => f === ext) ?? null;
}

/** Synthesize the metadata-only completion the lost SSE frame would have carried. */
export function galleryCompletion(print: GalleryImage): CompleteEvent {
  const meta = print.metadata;
  return {
    image: "",
    format: print.format ?? meta.output_format ?? formatFromFilename(print.filename) ?? "png",
    width: meta.width,
    height: meta.height,
    original_image: null,
    original_width: null,
    original_height: null,
    seed_used: meta.seed,
    // The true duration was lost with the stream; 0 tells the UI to omit it.
    generation_time_ms: 0,
    model: meta.model,
    video_frames: meta.frames ?? meta.video_frames ?? null,
    video_fps: meta.fps ?? meta.video_fps ?? null,
    video_thumbnail: null,
    video_gif_preview: null,
    video_has_audio: false,
    video_duration_ms: null,
    video_audio_sample_rate: null,
    video_audio_channels: null,
    gpu: null,
    filename: print.filename,
    original_filename: null,
    metadata: meta,
  };
}

function chainFileCompletion(job: Job, filename: string): CompleteEvent {
  const seed = Number(job.visualSeed);
  return {
    image: "",
    format: formatFromFilename(filename) ?? "mp4",
    width: job.width,
    height: job.height,
    original_image: null,
    original_width: null,
    original_height: null,
    seed_used: Number.isSafeInteger(seed) ? seed : 0,
    generation_time_ms: 0,
    model: job.model,
    video_frames: null,
    video_fps: null,
    video_thumbnail: null,
    video_gif_preview: null,
    video_has_audio: false,
    video_duration_ms: null,
    video_audio_sample_rate: null,
    video_audio_channels: null,
    gpu: null,
    filename,
    original_filename: null,
    metadata: null,
  };
}

function settledExternally(job: Job): boolean {
  return job.status === "complete" || job.status === "error";
}

function settleFailure(job: Job, message: string): void {
  job.stage = null;
  job.error = message;
  job.status = "error";
}

function settleCompleted(job: Job, complete: CompleteEvent, opts: GenerationRecoveryOptions): void {
  job.result = metadataOnlyResult(complete);
  job.visualSeed = String(complete.seed_used);
  job.stage = null;
  job.error = null;
  job.status = "complete";
  opts.refreshResultUrl(job.clientId);
}

function findQueueEntry(
  entries: readonly RecoveryQueueEntry[],
  job: Job,
): RecoveryQueueEntry | null {
  if (job.id) return entries.find((entry) => entry.id === job.id) ?? null;
  // Pre-ID jobs (the queued frame never arrived) join on the pinned seed the
  // request carried — mirrors the cancel path's pre-ID snapshot semantics.
  const seed = Number(job.visualSeed);
  if (!Number.isSafeInteger(seed)) return null;
  return (
    entries.find(
      (entry) =>
        entry.seed_pinned !== false &&
        entry.metadata?.seed === seed &&
        (entry.metadata.model === job.model || entry.model === job.model) &&
        (!entry.metadata.prompt || !job.prompt || entry.metadata.prompt === job.prompt),
    ) ?? null
  );
}

async function reconcileJob(job: Job, opts: GenerationRecoveryOptions): Promise<void> {
  const isActive = opts.isActive ?? (() => true);
  const sleep =
    opts.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const interval = opts.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const interruptedCopy = `The connection to ${opts.hostLabel} was interrupted and this print didn’t finish.`;
  let transportRetries = 0;
  try {
    while (isActive() && !settledExternally(job)) {
      try {
        await reconcileJobOnce();
        return;
      } catch (cause) {
        if (settledExternally(job) || !isActive()) return;
        // The reconcile query died the same way the stream did — the network
        // is likely still coming back from suspension. Back off and retry
        // (bounded) before declaring an outcome; any successful round-trip
        // inside reconcileJobOnce resets the budget.
        if (isTransportFailure(cause) && transportRetries < MAX_TRANSPORT_RETRIES) {
          transportRetries += 1;
          await sleep(interval);
          continue;
        }
        settleFailure(job, describeTransportError(cause, opts.hostLabel));
        return;
      }
    }
  } catch (cause) {
    if (!settledExternally(job)) {
      settleFailure(job, describeTransportError(cause, opts.hostLabel));
    }
  }

  // One full reconcile pass; throws on query failure, returns once the job is
  // settled or abandoned (external settle / unmount).
  async function reconcileJobOnce(): Promise<void> {
    while (isActive() && !settledExternally(job)) {
      if (opts.chain && job.id) {
        const detail = await apiJsonTo<ChainJobDetail>(
          opts.target,
          `/api/chain-jobs/${encodeURIComponent(job.id)}`,
        ).catch((cause: unknown) => {
          if (cause instanceof ApiError && cause.status === 404) return null;
          throw cause;
        });
        transportRetries = 0;
        if (settledExternally(job) || !isActive()) return;
        if (detail && (detail.state === "queued" || detail.state === "running")) {
          job.stage = `Developing on ${opts.hostLabel}`;
          await sleep(interval);
          continue;
        }
        if (detail?.state === "failed") {
          settleFailure(
            job,
            detail.error?.trim() || `The video chain failed on ${opts.hostLabel}.`,
          );
          return;
        }
        if (detail?.state === "cancelled") {
          settleFailure(job, "Cancelled");
          return;
        }
        if (detail?.state === "interrupted") {
          settleFailure(
            job,
            `${opts.hostLabel} interrupted this video chain before it finished. Develop again to restart it.`,
          );
          return;
        }
        const chainOutput = detail?.finalizes.at(-1)?.output ?? null;
        const prints = await apiJsonTo<GalleryImage[]>(opts.target, "/api/gallery");
        transportRetries = 0;
        if (settledExternally(job) || !isActive()) return;
        const match =
          (chainOutput ? prints.find((print) => print.filename === chainOutput) : null) ??
          matchGalleryPrint(job, prints);
        if (match) {
          settleCompleted(job, galleryCompletion(match), opts);
        } else if (chainOutput) {
          settleCompleted(job, chainFileCompletion(job, chainOutput), opts);
        } else {
          settleFailure(job, interruptedCopy);
        }
        return;
      }

      const listing = await apiJsonTo<{ entries?: RecoveryQueueEntry[] }>(
        opts.target,
        "/api/queue",
      );
      transportRetries = 0;
      if (settledExternally(job) || !isActive()) return;
      const entry = findQueueEntry(listing.entries ?? [], job);
      if (entry?.state === "running") {
        // Still developing server-side — re-attach by polling until it
        // leaves the queue, then collect the print from the gallery.
        job.stage = `Developing on ${opts.hostLabel}`;
        await sleep(interval);
        continue;
      }
      if (entry?.state === "queued") {
        // The host skips queued jobs whose client stream died with the
        // suspension — clear the zombie row instead of letting it linger.
        await apiFetchTo(opts.target, `/api/queue/${encodeURIComponent(entry.id)}`, {
          method: "DELETE",
        }).catch(() => undefined);
        if (settledExternally(job)) return;
        settleFailure(
          job,
          `The connection dropped while this print waited in ${opts.hostLabel}’s queue. Develop again to requeue it.`,
        );
        return;
      }
      const prints = await apiJsonTo<GalleryImage[]>(opts.target, "/api/gallery");
      transportRetries = 0;
      if (settledExternally(job) || !isActive()) return;
      const match = matchGalleryPrint(job, prints);
      if (match) {
        settleCompleted(job, galleryCompletion(match), opts);
      } else {
        settleFailure(job, interruptedCopy);
      }
      return;
    }
  }
}

/**
 * Settle every job in `jobs` whose error is a dead-socket interruption by
 * querying the frozen submission route. Jobs with server-authored failures,
 * cancellations, or completions are left untouched. Resolves when every
 * interrupted job has a true outcome, so the caller's settled handling
 * (result promotion, prepared one-based failure naming, pruning) runs on
 * reconciled state.
 */
export function reconcileInterruptedGenerationJobs(
  jobs: readonly Job[],
  opts: GenerationRecoveryOptions,
): Promise<void> {
  const interrupted = jobs.filter(
    (job) => job.status === "error" && isInterruptedGenerationError(job.error),
  );
  if (interrupted.length === 0) return Promise.resolve();
  for (const job of interrupted) {
    // Reclaim the job as live before any awaits: the raw transport error must
    // never render, and a re-attached job belongs back in the queue rail.
    job.error = null;
    job.status = "loading";
    job.stage = `Reconnecting to ${opts.hostLabel}`;
  }
  return Promise.all(interrupted.map((job) => reconcileJob(job, opts))).then(() => undefined);
}

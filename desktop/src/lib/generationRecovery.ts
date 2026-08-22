import { ApiError, apiJsonTo, type ApiTarget } from "./api/client";
import { describeTransportError, isTransportFailure } from "./api/errors";
import type {
  ChainJobDetail,
  ChainJobListing,
  CompleteEvent,
  GalleryImage,
  OutputFormat,
  OutputMetadata,
} from "./api/types";
import { isCancelledError, metadataOnlyResult, type Job } from "./generationJob";
import { sha256HexOfBase64 } from "./sourceRestore";
import {
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
  type MinimaxH3Task,
} from "@studio/lib/minimaxH3Authoring";

/**
 * Reconciliation for generations whose stream died while the host kept going.
 *
 * Two shells need this. iOS freezes the WebView and tears down sockets while
 * the app is backgrounded, so every held generation stream dies with a raw
 * WebKit transport error even though the host kept working; and on any surface
 * a durable-queue host ends a retained job's stream while keeping the job. In
 * both cases the host is still the authority: an accepted RUNNING or QUEUED
 * job finishes and lands in the host's gallery independently of the client
 * stream. This module re-queries the job's frozen submission
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
/** Retry budget for a job the host explicitly RETAINED. The host told us it
 *  kept the work and is restarting, so an unreachable socket is expected for
 *  as long as that takes — a cold model load alone outlasts the suspension
 *  budget above, and settling such a job as failed after ~12 s would announce
 *  a failure for work that is about to run. Bounded so the reconcile always
 *  ends: ~5 minutes at the default poll interval. */
const RETAINED_TRANSPORT_RETRIES = 75;
/** How many times a RETAINED job may be absent from a successfully-read queue
 *  before reconciliation gives up on it. This is the restart handoff window —
 *  the worker has exited and its registry entry is gone, but the durable row
 *  is invisible until replay resubmits it — so it is sized like the retry
 *  budget above rather than like a normal poll. */
const RETAINED_HANDOFF_POLLS = 75;
/** How many times a visible `queued` row may be re-read before reconciliation
 *  stops waiting on it. Every accepted row belongs to the host and runs without
 *  an attached client stream. `durable` only says whether a server restart can
 *  replay the request; it never authorizes the client to cancel live work.
 *  Waiting is patient by design, but a host booted with no dispatch owner
 *  (`MOLD_GPUS=none`) may keep a row listed forever. Unbounded there does not merely
 *  spin — desktop awaits this inside `settled`, so the batch's completion toast
 *  and its pending-consumer entry are stranded with it. ~30 minutes at the
 *  default poll interval, which outlasts an ordinary queue wait; giving up is
 *  purely a client-side decision, so it settles as unreconciled and the host's
 *  own row is released to surface the work if it ever does run. */
const QUEUED_JOB_POLLS = 450;
/** How long a finished reconciliation pass suppresses another one for the same
 *  job. The shared store runs recovery as part of a batch settling and the
 *  iPhone shell calls the same entry point immediately afterwards in its own
 *  `settled.then`; a pass that reached no verdict deliberately leaves the job
 *  eligible, so without this the second caller spends the ENTIRE retry budget
 *  again before the phone can show a summary. Short enough that a genuine
 *  later resume still reconciles. */
const RECONCILE_COOLDOWN_MS = 5_000;
const PRE_ID_CLOCK_SKEW_MS = 1_000;
const PRE_ID_JOIN_WINDOW_MS = 5_000;

/** Minimal `GET /api/queue` row — declared locally so the remote-only phone
 * never imports a desktop store to borrow the shape. */
interface RecoveryQueueEntry {
  id: string;
  model?: string;
  /** `held` is additive: a journalled job the host parked and will not
   * auto-run. Absent on servers without the durable queue. */
  state: "queued" | "running" | "held";
  seed_pinned?: boolean | null;
  started_at_unix_ms?: number;
  metadata?: OutputMetadata | null;
  /** Whether the host journalled THIS job — additive and deliberately
   * per-job: a host that can promise durability still reports `false` for a
   * job it excluded at admission. Absent on older servers. */
  durable?: boolean | null;
  /** Why a held job is parked. Present only for `state: "held"`. */
  held_reason?: string | null;
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

type GalleryMatchJob = Pick<
  Job,
  | "id"
  | "recoveredJobId"
  | "visualSeed"
  | "model"
  | "prompt"
  | "width"
  | "height"
  | "total"
  | "request"
  | "submittedAtUnixMs"
>;

interface H3BoundaryIdentity {
  task: "fl2va";
  firstFrameSha256: string | null;
  keyframes: Array<{ frame: number; sha256: string }>;
}

interface H3ReferenceIdentity {
  task: "ref2va";
  references: Array<{ index: number; kind: string; sha256: string }>;
}

type H3ConditioningIdentity = H3BoundaryIdentity | H3ReferenceIdentity;

interface H3RecoveryIdentity {
  conditioning: H3ConditioningIdentity;
  width: number;
  height: number;
  steps: number;
  frames: number;
  fps: number;
}

function normalizedSha256(value: string | null | undefined): string | null {
  return value?.trim().toLowerCase() || null;
}

async function inlineSha256(value: string | null | undefined): Promise<string | null> {
  if (!value) return null;
  return sha256HexOfBase64(value).catch(() => null);
}

function requiredPositiveInteger(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : null;
}

async function submittedH3RecoveryIdentity(
  job: GalleryMatchJob,
): Promise<H3RecoveryIdentity | null | undefined> {
  if (!isMinimaxH3Identity(null, job.model)) return undefined;
  const task = minimaxH3TaskForModel(job.model);
  const request = job.request;
  if (!task || !request || request.model !== job.model) return null;
  // H3 recovery is intentionally stricter than the legacy gallery join. The
  // frozen request is the only authority for its exact render shape, and a
  // host row that predates any of these fields cannot safely be claimed.
  const width = requiredPositiveInteger(request.width);
  const height = requiredPositiveInteger(request.height);
  const steps = requiredPositiveInteger(request.steps);
  const frames = requiredPositiveInteger(request.frames);
  const fps = requiredPositiveInteger(request.fps);
  if (!width || !height || !steps || !frames || !fps) return null;

  if (task === "fl2va") {
    // Snapshot every mutable field before the first digest await. This one
    // concrete submission identity is then reused for every recovery poll.
    const sourceImage = request.source_image;
    const submittedKeyframes = (request.keyframes ?? []).map(({ frame, image }) => ({
      frame,
      image,
    }));
    const firstFrameSha256 = await inlineSha256(sourceImage);
    if (sourceImage && !firstFrameSha256) return null;
    const keyframes: H3BoundaryIdentity["keyframes"] = [];
    for (const keyframe of submittedKeyframes) {
      const sha256 = await inlineSha256(keyframe.image);
      if (!sha256) return null;
      keyframes.push({ frame: keyframe.frame, sha256 });
    }
    return {
      conditioning: { task, firstFrameSha256, keyframes },
      width,
      height,
      steps,
      frames,
      fps,
    };
  }

  const submittedReferences = (request.references ?? []).map((reference) => ({
    kind: reference.kind,
    media: reference.media,
    provenanceSha256: normalizedSha256(reference.provenance?.sha256),
  }));
  if (submittedReferences.length === 0) return null;
  const references: H3ReferenceIdentity["references"] = [];
  for (const [offset, reference] of submittedReferences.entries()) {
    let sha256 = reference.provenanceSha256;
    if (reference.media.authority === "inline") {
      const inline = await inlineSha256(reference.media.data);
      if (!inline || (sha256 && inline !== sha256)) return null;
      sha256 = inline;
    }
    if (!sha256) return null;
    references.push({ index: offset + 1, kind: reference.kind, sha256 });
  }
  return {
    conditioning: { task, references },
    width,
    height,
    steps,
    frames,
    fps,
  };
}

function recordedH3Conditioning(
  task: MinimaxH3Task,
  metadata: OutputMetadata | null | undefined,
): H3ConditioningIdentity | null {
  if (!metadata) return null;
  if (task === "fl2va") {
    const firstFrameSha256 = normalizedSha256(metadata.source_image_sha256);
    if (metadata.source_image_name && !firstFrameSha256) return null;
    const keyframes: H3BoundaryIdentity["keyframes"] = [];
    for (const keyframe of metadata.keyframes ?? []) {
      const sha256 = normalizedSha256(keyframe.sha256);
      if (!sha256) return null;
      keyframes.push({ frame: keyframe.frame, sha256 });
    }
    return { task, firstFrameSha256, keyframes };
  }

  const references: H3ReferenceIdentity["references"] = [];
  for (const reference of metadata.references ?? []) {
    const sha256 = normalizedSha256(reference.sha256);
    if (!sha256) return null;
    references.push({ index: reference.index, kind: reference.kind, sha256 });
  }
  return { task, references };
}

function h3RecoveryIdentityMatches(
  submitted: H3RecoveryIdentity,
  metadata: OutputMetadata | null | undefined,
): boolean {
  if (
    !metadata ||
    metadata.width !== submitted.width ||
    metadata.height !== submitted.height ||
    metadata.steps !== submitted.steps ||
    metadata.frames !== submitted.frames ||
    metadata.fps !== submitted.fps
  ) {
    return false;
  }
  const recorded = recordedH3Conditioning(submitted.conditioning.task, metadata);
  return recorded !== null && JSON.stringify(recorded) === JSON.stringify(submitted.conditioning);
}

function matchGalleryPrintWithIdentity(
  job: GalleryMatchJob,
  prints: readonly GalleryImage[],
  h3Identity: H3RecoveryIdentity | null | undefined,
): GalleryImage | null {
  // The server stamps the producing job's queue id into the print it saves.
  // Read that before inferring anything: it is exact, it survives clock skew
  // between host and client (the age fence below compares a CLIENT submit time
  // against a HOST print stamp), and it separates same-second fixed-seed twins
  // that every other field in this join reproduces identically. The heuristic
  // stays as the fallback for hosts that predate the stamp.
  const stampedId = serverIdentity(job);
  if (stampedId) {
    // One job can legitimately own two prints: a replay that crashed after
    // publishing leaves the original and the re-render, both stamped. Take the
    // NEWEST rather than the first listed — `/api/gallery` happens to sort
    // newest-first today, but recovery must not depend on an ordering it never
    // states, or an unrelated change to that endpoint silently hands back the
    // older copy and the stale-print bug returns wearing someone else's diff.
    const stamped = prints
      .filter((print) => print.metadata?.job_id === stampedId)
      .reduce<GalleryImage | null>(
        (newest, print) =>
          !newest || (print.timestamp ?? 0) > (newest.timestamp ?? 0) ? print : newest,
        null,
      );
    if (stamped) return stamped;
  }
  const seed = Number(job.visualSeed);
  if (!Number.isSafeInteger(seed) || h3Identity === null) return null;
  const differs = (recorded: number | null | undefined, submitted: number): boolean =>
    typeof recorded === "number" && submitted > 0 && recorded !== submitted;
  // A print that already existed cannot be the output of this submission.
  // Every other field in this join — seed, model, prompt, dimensions, steps —
  // is reproduced exactly by a fixed-seed re-run, so age is the only thing
  // separating "the render I just asked for" from "one I made yesterday".
  const earliestSeconds = (job.submittedAtUnixMs - PRE_ID_CLOCK_SKEW_MS) / 1000;
  for (const print of prints) {
    const meta = print.metadata;
    if (typeof print.timestamp === "number" && print.timestamp < earliestSeconds) continue;
    if (!meta || meta.seed !== seed || meta.model !== job.model) continue;
    if (meta.prompt && job.prompt && meta.prompt !== job.prompt) continue;
    if (h3Identity) {
      if (!h3RecoveryIdentityMatches(h3Identity, meta)) continue;
    } else {
      if (differs(meta.width, job.width) || differs(meta.height, job.height)) continue;
      if (differs(meta.steps, job.total)) continue;
    }
    return print;
  }
  return null;
}

/**
 * Match a job to the gallery row its completion produced. Every mobile
 * sibling ships an explicit seed (`base + i`), so seed + model is an exact
 * join; the recorded prompt is the tiebreaker for prepared siblings, whose
 * prompts are unique within a batch. Dimensions and steps must also agree
 * when both sides know them — a fixed-seed re-run at new settings whose
 * stream died pre-queue must not resurrect an older print. Age is the final
 * bound and the only one a fixed-seed re-run cannot reproduce: a print that
 * already existed when this job was submitted can never be its output. H3
 * additionally joins on exact endpoint and ordered-reference content hashes.
 */
export async function matchGalleryPrint(
  job: GalleryMatchJob,
  prints: readonly GalleryImage[],
): Promise<GalleryImage | null> {
  const h3Identity = await submittedH3RecoveryIdentity(job);
  return matchGalleryPrintWithIdentity(job, prints, h3Identity);
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

/**
 * The server id this job is known by, whether it is still on the row or was
 * released by an earlier pass.
 *
 * `settleUnreconciled` moves the id to `recoveredJobId` so the host's own fleet
 * row stops being suppressed — but the job is no more anonymous for it. Every
 * question of the form "do we have a server identity?" must ask THIS, or a
 * later pass re-reads a known job as one that never received a queued frame
 * and falls back to guesswork that is deliberately narrower than the truth.
 */
function serverIdentity(job: Pick<Job, "id" | "recoveredJobId">): string {
  return job.id || job.recoveredJobId || "";
}

function settleFailure(job: Job, message: string): void {
  job.stage = null;
  job.error = message;
  job.status = "error";
  job.interrupted = false;
}

/**
 * Settle a job the host never answered about. Reconciliation reached no
 * verdict, so the row stays marked as an interruption: a later resume may
 * still learn the truth, and callers that treat `interrupted` as "do not
 * announce this as a failure" keep that protection. A definitive answer —
 * the host has no such job and no such print — uses `settleFailure` instead.
 */
function settleUnreconciled(job: Job, message: string): void {
  job.stage = null;
  job.error = message;
  job.status = "error";
  job.interrupted = true;
  // Release the server id. Desktop suppresses every fleet row whose id matches
  // a local job — that is how one job renders as one row — so a row this
  // client has stopped tracking would otherwise HIDE the host's own row when
  // it comes back and finishes the work, and the print would land unseen. The
  // id is kept aside: it is still the exact key for the print the host saves.
  if (job.id) {
    job.recoveredJobId = job.id;
    job.id = "";
  }
}

function settleCompleted(job: Job, complete: CompleteEvent, opts: GenerationRecoveryOptions): void {
  job.result = metadataOnlyResult(complete);
  job.visualSeed = String(complete.seed_used);
  job.stage = null;
  job.error = null;
  job.status = "complete";
  job.interrupted = false;
  opts.refreshResultUrl(job.clientId);
}

function findQueueEntry(
  entries: readonly RecoveryQueueEntry[],
  job: Job,
  h3Identity: H3RecoveryIdentity | null | undefined,
): RecoveryQueueEntry | null {
  const known = serverIdentity(job);
  if (known) return entries.find((entry) => entry.id === known) ?? null;
  // Pre-ID jobs (the queued frame never arrived) join on the pinned seed the
  // request carried, plus H3 conditioning identity when applicable — mirrors
  // the cancel path's pre-ID snapshot semantics without crossing submissions.
  const seed = Number(job.visualSeed);
  if (!Number.isSafeInteger(seed) || h3Identity === null) return null;
  const earliest = job.submittedAtUnixMs - PRE_ID_CLOCK_SKEW_MS;
  const latest = job.submittedAtUnixMs + PRE_ID_JOIN_WINDOW_MS;
  const candidates = entries
    .filter(
      (entry) =>
        typeof entry.started_at_unix_ms === "number" &&
        entry.started_at_unix_ms >= earliest &&
        entry.started_at_unix_ms <= latest &&
        entry.seed_pinned !== false &&
        entry.metadata?.seed === seed &&
        (entry.metadata.model === job.model || entry.model === job.model) &&
        (!entry.metadata.prompt || !job.prompt || entry.metadata.prompt === job.prompt),
    )
    .sort((a, b) => a.started_at_unix_ms! - b.started_at_unix_ms!);
  return (
    candidates.find(
      (entry) => !h3Identity || h3RecoveryIdentityMatches(h3Identity, entry.metadata),
    ) ?? null
  );
}

async function findPreIdChain(job: Job, opts: GenerationRecoveryOptions): Promise<string | null> {
  // Public sequence lists intentionally hide the ephemeral chain-runner shim
  // behind a one-shot long video. Recovery is the one consumer that needs to
  // find that internal record when iOS suspended before receiving its id.
  const listing = await apiJsonTo<ChainJobListing>(
    opts.target,
    "/api/chain-jobs?include_ephemeral=true",
  );
  const earliest = job.submittedAtUnixMs - PRE_ID_CLOCK_SKEW_MS;
  const latest = job.submittedAtUnixMs + PRE_ID_JOIN_WINDOW_MS;
  return (
    listing.jobs
      .filter(
        (candidate) =>
          candidate.ephemeral === true &&
          candidate.model === job.model &&
          (candidate.state === "queued" || candidate.state === "running") &&
          candidate.created_at_unix_ms >= earliest &&
          candidate.created_at_unix_ms <= latest,
      )
      .sort((a, b) => a.created_at_unix_ms - b.created_at_unix_ms)[0]?.id ?? null
  );
}

async function reconcileJob(job: Job, opts: GenerationRecoveryOptions): Promise<void> {
  const isActive = opts.isActive ?? (() => true);
  const sleep =
    opts.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const interval = opts.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const interruptedCopy = `The connection to ${opts.hostLabel} was interrupted and this print didn’t finish.`;
  // A retained job is the host's to finish; never claim it failed just because
  // this client stopped being able to see it.
  const retained = job.retainedByHost === true;
  const maxTransportRetries = retained ? RETAINED_TRANSPORT_RETRIES : MAX_TRANSPORT_RETRIES;
  const retainedUnreachableCopy =
    `${opts.hostLabel} kept this print in its queue but hasn’t come back yet. ` +
    `It will finish there — check the Library.`;
  let transportRetries = 0;
  // Successful polls that found no row for a retained job — the restart
  // handoff window, bounded so reconciliation always ends.
  let handoffPolls = 0;
  // Re-reads of a visible `queued` row, bounded so a row the host never
  // dispatches cannot hold the reconcile open forever.
  let queuedJobPolls = 0;
  const queuedStalledCopy =
    `${opts.hostLabel} still has this print queued but hasn’t started it. ` +
    `It will finish there if the host picks it up — check the Library.`;
  let h3Identity: H3RecoveryIdentity | null | undefined;
  try {
    // Digest the frozen H3 request once. Queue polls and the eventual gallery
    // lookup reuse this value, so large frame/reference payloads are never
    // re-read every four seconds.
    h3Identity = await submittedH3RecoveryIdentity(job);
    if (settledExternally(job) || !isActive()) return;
    if (h3Identity === null) {
      settleFailure(job, interruptedCopy);
      return;
    }
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
        if (isTransportFailure(cause) && transportRetries < maxTransportRetries) {
          transportRetries += 1;
          await sleep(interval);
          continue;
        }
        if (isTransportFailure(cause)) {
          // The host never answered — no verdict, so no announcement.
          settleUnreconciled(
            job,
            retained ? retainedUnreachableCopy : describeTransportError(cause, opts.hostLabel),
          );
          return;
        }
        settleFailure(job, describeTransportError(cause, opts.hostLabel));
        return;
      }
    }
  } catch (cause) {
    if (!settledExternally(job) && isActive()) {
      if (isTransportFailure(cause)) {
        settleUnreconciled(job, describeTransportError(cause, opts.hostLabel));
      } else {
        settleFailure(job, describeTransportError(cause, opts.hostLabel));
      }
    }
  }

  // One full reconcile pass; throws on query failure, returns once the job is
  // settled or abandoned (external settle / unmount).
  async function reconcileJobOnce(): Promise<void> {
    while (isActive() && !settledExternally(job)) {
      if (opts.chain && !serverIdentity(job)) {
        const chainId = await findPreIdChain(job, opts);
        transportRetries = 0;
        if (settledExternally(job) || !isActive()) return;
        job.id = chainId ?? "";
        if (!job.id) {
          // The host has no chain record for this submission, so nothing
          // proves it was ever admitted. Guessing from the gallery here would
          // hand back whichever older clip happens to match.
          settleFailure(job, interruptedCopy);
          return;
        }
      }
      if (opts.chain && serverIdentity(job)) {
        const detail = await apiJsonTo<ChainJobDetail>(
          opts.target,
          `/api/chain-jobs/${encodeURIComponent(serverIdentity(job))}`,
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
          matchGalleryPrintWithIdentity(job, prints, h3Identity);
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
      // `Array.prototype.entries` exists, so a body that is itself an array
      // must never be read as a listing — that yields a FUNCTION here and
      // throws deep inside the join, which the retry loop then mistakes for a
      // transport failure and waits out.
      const rows = Array.isArray(listing?.entries) ? listing.entries : [];
      const entry = findQueueEntry(rows, job, h3Identity);
      // Adopt the recovered id immediately. The pre-ID join identifies the row
      // by seed, timing, model and conditioning when suspension beat the queued
      // frame — and every id-keyed action downstream, `cancel()` above all,
      // reads `job.id`. Without this the durable branch can wait on a job the
      // user is then told cannot be cancelled ("Remote cancellation was not
      // confirmed"): identified by this client, and uncancellable by it.
      if (entry && !job.id) job.id = entry.id;
      if (entry?.state === "running") {
        // Still developing server-side — re-attach by polling until it
        // leaves the queue, then collect the print from the gallery.
        job.stage = `Developing on ${opts.hostLabel}`;
        await sleep(interval);
        continue;
      }
      if (entry?.state === "held") {
        // Listed but parked: the host exhausted this job's dispatch or replay
        // budget and will never start it on its own. Waiting would never end.
        settleFailure(
          job,
          `${opts.hostLabel} is holding this print and will not run it automatically${
            entry.held_reason ? ` (${entry.held_reason})` : ""
          }. Develop again to requeue it.`,
        );
        return;
      }
      if (entry?.state === "queued") {
        // The server detached accepted work from its submitting stream. A
        // reference-upload job reports `durable: false` because its media is
        // intentionally not journalled for restart replay, but it still runs
        // normally while this server process remains alive. Only an explicit
        // user cancellation may DELETE it.
        if (queuedJobPolls >= QUEUED_JOB_POLLS) {
          // Listed and never dispatched. The host still owns the work, so this
          // is never announced as a failure — but the wait ends, because a
          // caller is blocked on it.
          settleUnreconciled(job, queuedStalledCopy);
          return;
        }
        queuedJobPolls += 1;
        job.stage = `Waiting in ${opts.hostLabel}’s queue`;
        await sleep(interval);
        continue;
      }
      const prints = await apiJsonTo<GalleryImage[]>(opts.target, "/api/gallery");
      transportRetries = 0;
      if (settledExternally(job) || !isActive()) return;
      // Only claim a print for work the host is known to have accepted. A job
      // id is that proof: it arrives on the server's own `queued` frame, and
      // the pre-ID join adopts it from a matching queue row. Without either,
      // this submission may never have reached the host at all — and a
      // fixed-seed re-run of the same prompt makes an unrelated print look
      // exactly like its output.
      const match = serverIdentity(job)
        ? matchGalleryPrintWithIdentity(job, prints, h3Identity)
        : null;
      if (match) {
        settleCompleted(job, galleryCompletion(match), opts);
        return;
      }
      if (retained && handoffPolls < RETAINED_HANDOFF_POLLS) {
        // Absent from the queue AND absent from the gallery, on a job the host
        // told us it kept. That combination is the handoff window: the worker
        // has exited and dropped its registry entry, but the durable row stays
        // invisible to `/api/queue` until restart replay resubmits it. A
        // successful empty listing there is server bookkeeping, not evidence.
        handoffPolls += 1;
        job.stage = `Waiting for ${opts.hostLabel} to come back`;
        await sleep(interval);
        continue;
      }
      if (retained) {
        // Bounded, so reconciliation always ends — but the host promised to
        // run this, so it stays reconcilable rather than declared failed.
        settleUnreconciled(job, retainedUnreachableCopy);
        return;
      }
      settleFailure(job, interruptedCopy);
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
  const now = Date.now();
  const interrupted = jobs.filter(
    (job) =>
      job.status === "error" &&
      (job.interrupted || isInterruptedGenerationError(job.error)) &&
      // A cancel is terminal even on a row already flagged as interrupted —
      // the store guards this in its own pre-filter, and the entry point both
      // callers share must not depend on that.
      !isCancelledError(job.error) &&
      // Not one another pass just finished: its outcome is this outcome.
      !(
        typeof job.reconciledAtUnixMs === "number" &&
        now - job.reconciledAtUnixMs < RECONCILE_COOLDOWN_MS
      ),
  );
  if (interrupted.length === 0) return Promise.resolve();
  for (const job of interrupted) {
    // Reclaim the job as live before any awaits: the raw transport error must
    // never render, and a re-attached job belongs back in the queue rail.
    job.error = null;
    job.interrupted = false;
    job.status = "loading";
    job.stage = `Reconnecting to ${opts.hostLabel}`;
  }
  return Promise.all(
    interrupted.map((job) =>
      reconcileJob(job, opts).finally(() => {
        job.reconciledAtUnixMs = Date.now();
      }),
    ),
  ).then(() => undefined);
}

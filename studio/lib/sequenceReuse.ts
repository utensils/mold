/**
 * Sequence provenance helpers shared by every shell.
 *
 * A finished sequence is a print: a `/api/gallery` row carrying
 * `metadata.chain_job_id` and `metadata.chain`. These helpers are how a
 * surface gets from that print back to something editable — the Create
 * canvas's settled-sequence result, History ▸ Sequences' "Show print", and
 * the Library's **Reuse settings** / **Edit sequence** pair.
 *
 * The distinction the two Library actions carry:
 *
 * - **Reuse settings** makes a NEW sequence from the recorded clips. It is
 *   lossy by construction (a print records one negative and one source sha,
 *   never the whole rail) and says so.
 * - **Edit sequence** continues the OLD job through `GET /api/chain-jobs/:id`
 *   → `chainScriptToClips`, which is lossless. Nothing here duplicates that
 *   path; a second clip loader would drift from it.
 *
 * `OutputMetadataLike` is a structural minimum on purpose: desktop, web, and
 * iPhone each keep their own `OutputMetadata` interface, and unifying them is
 * a real but separate cleanup.
 */

import type { ChainOutputMetadata } from "./api/chainTypes";
import {
  DEFAULT_FADE_FRAMES,
  newSequenceClip,
  type SequenceClipForm,
} from "./sequenceForm";

export interface OutputMetadataLike {
  chain_job_id?: string | null;
  chain?: ChainOutputMetadata | null;
  /** Clip 1's negative prompt — `synthetic_generate_request` records the
   * first stage's, never the rest. */
  negative_prompt?: string | null;
  source_image_sha256?: string | null;
  source_image_name?: string | null;
}

/**
 * Recorded stages → editable clips.
 *
 * Deliberately NOT mapped:
 * - per-stage `seed`: the rail carries no per-clip seed (the wire carries a
 *   `seed_offset`), and deriving an offset from an effective u64 is guesswork
 *   that would silently produce a different sequence claiming to be a reuse.
 * - `sourceImage`: a print records at most clip 1's opening-frame sha, and the
 *   bytes live in a surface-local stash. Restoring it is desktop's job.
 */
export function chainMetadataToClips(
  chain: ChainOutputMetadata,
  opts: { negativePromptForFirstClip?: string | null } = {},
): SequenceClipForm[] {
  const negative = opts.negativePromptForFirstClip?.trim() ?? "";
  return chain.stages.map((stage, idx) => {
    const clip = newSequenceClip(stage.frames);
    clip.prompt = stage.prompt;
    // Clip 0 has nothing to transition FROM; the form coerces it anyway, and
    // a recorded fade there would render a seam control that does nothing.
    clip.transition = idx === 0 ? "smooth" : (stage.transition ?? "smooth");
    clip.fadeFrames = stage.fade_frames ?? DEFAULT_FADE_FRAMES;
    clip.negativePrompt = idx === 0 ? negative : "";
    return clip;
  });
}

/** What a print could not give back. Each flag is true only when the row
 * actually had that thing — an unconditional disclaimer is noise. */
export interface SequenceReuseLossiness {
  /** A negative was recorded, but only clip 1's; clips 2..n are unknowable. */
  negatives: boolean;
  /** A source image was recorded; this plan carries none. */
  clipSources: boolean;
  /** Effective per-stage seeds were recorded but are not reusable. */
  perClipSeeds: boolean;
}

export interface SequenceReusePlan {
  clips: SequenceClipForm[];
  /** The tail the sequence was rendered with — VALIDATION only. The live tail
   * is a property of the currently selected model. */
  recordedMotionTailFrames: number;
  lossy: SequenceReuseLossiness;
}

/**
 * Plan a fresh sequence draft from a print's recorded provenance, or `null`
 * when the row carries none (legacy pre-#564 rows, synthetic-metadata rows,
 * and single prints). Never synthesizes a one-clip sequence: missing metadata
 * stays omitted rather than guessed.
 */
export function planSequenceReuse(
  metadata: OutputMetadataLike,
): SequenceReusePlan | null {
  const chain = metadata.chain;
  if (!chain?.stages?.length) return null;
  const negative = metadata.negative_prompt?.trim() ?? "";
  return {
    clips: chainMetadataToClips(chain, {
      negativePromptForFirstClip: negative,
    }),
    recordedMotionTailFrames: chain.motion_tail_frames,
    lossy: {
      negatives: negative.length > 0 && chain.stages.length > 1,
      clipSources: Boolean(
        metadata.source_image_sha256 ?? metadata.source_image_name,
      ),
      perClipSeeds: chain.stages.some((stage) => Boolean(stage.seed)),
    },
  };
}

/**
 * The one quiet line a surface shows beneath the rail after a reuse (§11:
 * say what happened, don't apologize). Per-clip seeds are deliberately not
 * named — the shared seed IS applied, and the per-stage values were
 * server-derived offsets that never were reusable, so listing them every
 * time would drown the parts a user can actually act on.
 */
export function sequenceReuseNote(
  clipCount: number,
  lossy: SequenceReuseLossiness,
): string {
  const head = `reused ${clipCount} clip${clipCount === 1 ? "" : "s"}`;
  const parts: string[] = [];
  if (lossy.negatives) parts.push("negatives");
  if (lossy.clipSources) parts.push("clip sources");
  if (parts.length === 0) return head;
  return `${head} · ${parts.join(" and ")} aren't recorded in prints`;
}

/** Said once when a reuse had to raise durations (§11 voice). */
export function sequenceReuseClampNote(modelLabel: string): string {
  return `Clip durations raised to fit ${modelLabel} — the model's motion tail changed.`;
}

/** The click-time 404 fallback's copy — always names the host. */
export function sequenceGoneMessage(hostLabel: string): string {
  return `That sequence job is gone from ${hostLabel}. Duplicated its saved settings as a new sequence instead.`;
}

/**
 * The click-time network/5xx copy (§11 server error). Deliberately NOT paired
 * with a silent reuse fallback: the job probably still exists, and quietly
 * starting a fresh sequence would throw away its cached clips.
 */
export function sequenceHostUnreachableMessage(hostLabel: string): string {
  return `Can't reach ${hostLabel}. Check the host in Machines.`;
}

/**
 * Clip duration must stay STRICTLY greater than the active motion tail (an
 * existing repo invariant). A reused draft can violate it when the model's
 * tail differs from the one the print was rendered with, so raise the short
 * clips to `minFrames` and report how many moved — the surface must say so
 * rather than silently resizing.
 *
 * `minFrames` is the caller's model-derived minimum; if it is itself at or
 * below the tail it walks up the 8n+1 grid, because an invalid draft must
 * never be submittable no matter what the caller passed.
 */
export function clampClipsToMotionTail(
  clips: readonly SequenceClipForm[],
  motionTailFrames: number,
  minFrames: number,
): { clips: SequenceClipForm[]; raised: number } {
  let floor = Math.max(minFrames, 9);
  floor -= (floor - 1) % 8;
  while (floor <= motionTailFrames) floor += 8;

  let raised = 0;
  const next = clips.map((clip) => {
    if (clip.frames > motionTailFrames) return clip;
    raised += 1;
    return { ...clip, frames: floor };
  });
  return { clips: next, raised };
}

export type SequenceEditAvailability = "available" | "absent" | "unknown-host";

/**
 * Whether a print's **Edit sequence** action should render, WITHOUT probing.
 *
 * Positive knowledge only: a loaded host listing that lacks the id proves the
 * job is gone, but an unloaded listing proves nothing — absence of evidence
 * rendered as evidence of absence would hide a job that still exists. The
 * real check happens once, on click.
 *
 * `hostId` must come from the entry's OWN origin bucket. A merged print can
 * appear on three hosts; the other two hold auto-saved copies, not the
 * producer, and a hit there would edit an unrelated job.
 */
export function sequenceEditAvailability(args: {
  chainJobId: string | null | undefined;
  hostId: string | null;
  /** The origin host's chain listing, or `null` when it isn't loaded. */
  knownJobIds: readonly string[] | null;
}): SequenceEditAvailability {
  const jobId = args.chainJobId?.trim();
  if (!jobId) return "absent";
  if (!args.hostId) return "unknown-host";
  if (args.knownJobIds === null) return "available";
  return args.knownJobIds.includes(jobId) ? "available" : "absent";
}

/**
 * True when this gallery row is the stitched output of `jobId`.
 *
 * Filename or seed lookalikes are deliberately not accepted: ephemeral chain
 * outputs and legacy rows carry no job id at all, and guessing would hand the
 * canvas an unrelated video.
 */
export function isPrintOfChainJob(
  metadata: OutputMetadataLike,
  jobId: string,
): boolean {
  if (!jobId) return false;
  return metadata.chain_job_id === jobId;
}

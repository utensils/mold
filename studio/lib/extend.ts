/**
 * Browser-safe mirror of the server's `extend_video` admission rules
 * (`validate_extend` in `crates/mold-core/src/validation.rs`).
 *
 * Extend continues an existing clip in one request: the model renders `frames`
 * frames whose leading `overlap` reproduce the source tail, and the delivered
 * output is the original followed by everything past that overlap. It is the
 * chain motion-tail handoff with the carryover coming from a file, which is
 * why it inherits the same two hard constraints — the overlap must sit on the
 * family's own VAE temporal grid, and it must be strictly shorter than the
 * rendered clip or nothing new is produced.
 *
 * The server stays authoritative; this exists so the Create surfaces can
 * explain a bad combination before a round-trip.
 */

import { WAN_HANDOFF_DUPLICATED_FRAMES } from "./chainRouting";

/** Matches `DEFAULT_EXTEND_OVERLAP_FRAMES`; the shared chain motion tail. */
export const DEFAULT_EXTEND_OVERLAP_FRAMES = 17;

/**
 * Families whose engines implement continuation at all.
 *
 * This is the floor, not the answer — the browser mirror of
 * `require_extend_capable_family` in `crates/mold-core/src/validation.rs`.
 * Wan joined it in #783 because its continuation is the chain seam's
 * last-frame image conditioning, which the family's image-conditioned
 * checkpoints do accept; which of them do is `canOfferExtend`'s business.
 */
const EXTEND_CAPABLE_FAMILIES: ReadonlySet<string> = new Set([
  "ltx2",
  "ltx-2",
  "wan",
]);

/**
 * The frame-count grid a continuation's carried frames must sit on.
 *
 * Wan's VAE compresses time by 4 where the LTX families compress by 8, so its
 * valid counts are `4k+1`, not `8k+1`. Offering an off-grid option sends the
 * request straight into a 422 from the validator that owns the real rule.
 *
 * This lived in the shared sequence module until scene authoring was retired;
 * continuation is its only consumer now, so it lives beside it.
 */
function familyFrameStep(family: string | null | undefined): number {
  return family?.trim().toLowerCase() === "wan" ? 4 : 8;
}

export function familySupportsExtend(
  family: string | null | undefined,
): boolean {
  return EXTEND_CAPABLE_FAMILIES.has((family ?? "").trim().toLowerCase());
}

function isWanFamily(family: string | null | undefined): boolean {
  return (family ?? "").trim().toLowerCase() === "wan";
}

/** The `/api/models` row fields continuation depends on. */
export type ExtendCapableModel = {
  family?: string | null;
  supports_extend?: boolean | null;
  extend_default_overlap_frames?: number | null;
};

/**
 * Whether the extend controls should be offered for this model.
 *
 * A server that predates continuation omits `supports_extend` entirely, so
 * absence must read as "no" — offering the control would only produce a
 * rejected request. Unlike prompt expansion there is no useful optimistic
 * fallback.
 *
 * Within a capable family the advertised per-model field is the whole answer
 * (#783). It was once equivalent to the family, because LTX-2 extends on
 * every checkpoint; wan does not — its continuation is seeded with the
 * source's final frame, a channel only an image-conditioned checkpoint has,
 * which the server decides per model in `catalog::extend_capable_model` from
 * the same `source_image` contract the chain seam reads. The family set
 * remains only as the floor that keeps a stray advertisement off an engine
 * with no continuation path at all.
 */
export function canOfferExtend(
  model: ExtendCapableModel | null | undefined,
): boolean {
  if (!model) return false;
  if (model.supports_extend !== true) return false;
  return familySupportsExtend(model.family);
}

export function serverExtendOverlapDefault(
  model: ExtendCapableModel | null | undefined,
): number {
  // `extend_default_overlap_frames` is advertised per family, so a wan row
  // still carries LTX-2's 17 — a value wan's engine refuses outright. Pin it
  // to the one frame the handoff actually carries so an untouched
  // continuation is submittable.
  if (isWanFamily(model?.family)) return WAN_HANDOFF_DUPLICATED_FRAMES;
  const advertised = model?.extend_default_overlap_frames;
  return typeof advertised === "number" && advertised > 0
    ? advertised
    : DEFAULT_EXTEND_OVERLAP_FRAMES;
}

/**
 * The overlap a continuation actually submits.
 *
 * The control always renders one concrete number, so the request carries that
 * same number rather than leaving `extend_overlap_frames` absent: an absent
 * field takes whatever the host resolves, and for wan that is the family-wide
 * advertised 17 — which `wan/pipeline.rs`'s `extend_inner` refuses outright,
 * failing every continuation the user never had a second choice for. Wan's
 * select offers exactly one option, so `@change` never fires and "untouched"
 * is the normal case, not an edge one.
 *
 * Sending it explicitly also makes the client self-contained: the clamp holds
 * against a host whose own default is still family-blind. Every surface's
 * display computed and both request builders read this one function, so what
 * is on screen is what goes on the wire.
 */
export function resolveExtendOverlapFrames(
  explicit: number | null | undefined,
  model: ExtendCapableModel | null | undefined,
): number {
  return typeof explicit === "number" && explicit > 0
    ? explicit
    : serverExtendOverlapDefault(model);
}

/**
 * Valid overlap choices for a clip of `frames` frames, strictly below it.
 *
 * The carried frames re-encode through the family's own video VAE, so the
 * grid is that family's frame step — `8k+1` for the LTX families, `4k+1` for
 * wan (`familyFrameStep`). Wan then collapses to a single choice: it has no
 * latent motion tail, so the continuation carries exactly the one frame it
 * was seeded with and `wan/pipeline.rs`'s `extend_inner` refuses anything
 * larger rather than silently trimming good frames.
 */
export function extendOverlapOptions(
  frames: number | null | undefined,
  family?: string | null,
): number[] {
  const limit = frames ?? 0;
  if (isWanFamily(family)) {
    return WAN_HANDOFF_DUPLICATED_FRAMES < limit
      ? [WAN_HANDOFF_DUPLICATED_FRAMES]
      : [];
  }
  const step = familyFrameStep(family);
  const options: number[] = [];
  for (let overlap = 1; overlap < limit; overlap += step) options.push(overlap);
  return options;
}

/**
 * Whether the composer will actually submit a continuation.
 *
 * The browser mirror of `GenerateRequest::is_extend()` — either continuation
 * field present — ANDed with the extend-capable family gate both request
 * builders already apply before spreading those fields onto the wire. Both
 * halves matter: `is_extend()` is what
 * `mold_core::validation::request_carries_source_frames` reads, and a staged
 * clip on a family that cannot continue is dropped by the builder, so it must
 * not be allowed to satisfy the source-image contract either.
 *
 * One derivation, three surfaces: web reads `extendVideo` + `extendVideoPath`,
 * desktop and iPhone hold only the staged clip.
 */
export function submitsExtend(input: {
  family?: string | null;
  extendVideo?: unknown;
  extendVideoPath?: string | null;
}): boolean {
  if (!familySupportsExtend(input.family)) return false;
  return (
    Boolean(input.extendVideo) || Boolean(input.extendVideoPath?.trim().length)
  );
}

export type ExtendValidationInput = {
  overlapFrames: number | null | undefined;
  frames: number | null | undefined;
  hasSourceImage?: boolean;
  hasSourceVideo?: boolean;
  hasKeyframes?: boolean;
  /** Selected model's family — the overlap grid is per family. Absent keeps
   * the LTX-2 rule, which is what every pre-wan caller meant. */
  family?: string | null;
};

/**
 * Explain why a continuation would be rejected, or `null` when it is valid.
 * One message at a time, in the same order the server checks them, so the two
 * never disagree about which problem to name first.
 */
export function extendValidationError(
  input: ExtendValidationInput,
): string | null {
  if (input.hasSourceVideo) {
    return "Continuing a video cannot be combined with a source video — extend continues an existing clip, while a source video is reference conditioning for a fresh render.";
  }
  if (input.hasSourceImage) {
    return "Continuing a video cannot be combined with a source image — the first frames are pinned by the source video's tail.";
  }
  if (input.hasKeyframes) {
    return "Continuing a video cannot be combined with keyframes.";
  }

  const overlap = input.overlapFrames ?? DEFAULT_EXTEND_OVERLAP_FRAMES;
  if (overlap < 1) {
    return "Overlap must be at least 1 frame so the continuation has motion context.";
  }
  // Wan carries no latent motion tail — the continuation is seeded with the
  // source's final frame and re-renders exactly that frame — so its engine
  // accepts one value, not a grid (`wan/pipeline.rs`'s `extend_inner`).
  if (isWanFamily(input.family)) {
    if (overlap !== WAN_HANDOFF_DUPLICATED_FRAMES) {
      return `Overlap (${overlap}) must be exactly ${WAN_HANDOFF_DUPLICATED_FRAMES} frame — a Wan continuation is seeded with the source clip's final frame and carries no longer motion tail.`;
    }
  } else {
    // One grid, one derivation: `extendOverlapOptions` enumerates from
    // `familyFrameStep` and the rejection has to come from the same place,
    // or a family whose VAE compresses time differently gets an option list
    // and an error message that disagree.
    const step = familyFrameStep(input.family);
    if (overlap % step !== 1) {
      const ladder = [1, step + 1, 2 * step + 1, 3 * step + 1].join(", ");
      return `Overlap (${overlap}) must be ${step}k+1 (${ladder}, …) so the carried frames re-encode cleanly through the video VAE.`;
    }
  }
  if (input.frames != null && overlap >= input.frames) {
    return `Overlap (${overlap}) must be less than the clip length (${input.frames}) so the continuation adds at least one new frame.`;
  }
  return null;
}

/** Net-new frames a continuation appends, for the "adds N frames" cue. */
export function extendNewFrames(
  frames: number | null | undefined,
  overlapFrames: number | null | undefined,
): number | null {
  if (frames == null) return null;
  const overlap = overlapFrames ?? DEFAULT_EXTEND_OVERLAP_FRAMES;
  return Math.max(0, frames - overlap);
}

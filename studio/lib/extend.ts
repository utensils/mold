/**
 * Browser-safe mirror of the server's `extend_video` admission rules
 * (`validate_extend` in `crates/mold-core/src/validation.rs`).
 *
 * Extend continues an existing clip in one request: the model renders `frames`
 * frames whose leading `overlap` reproduce the source tail, and the delivered
 * output is the original followed by everything past that overlap. It is the
 * chain motion-tail handoff with the carryover coming from a file, which is
 * why it inherits the same two hard constraints — the overlap must sit on the
 * LTX-2 VAE's `8k+1` causal temporal grid, and it must be strictly shorter
 * than the rendered clip or nothing new is produced.
 *
 * The server stays authoritative; this exists so the Create surfaces can
 * explain a bad combination before a round-trip.
 */

/** Matches `DEFAULT_EXTEND_OVERLAP_FRAMES`; the shared chain motion tail. */
export const DEFAULT_EXTEND_OVERLAP_FRAMES = 17;

/** Families whose engines implement continuation. */
const EXTEND_CAPABLE_FAMILIES: ReadonlySet<string> = new Set(["ltx2", "ltx-2"]);

export function familySupportsExtend(
  family: string | null | undefined,
): boolean {
  return EXTEND_CAPABLE_FAMILIES.has((family ?? "").trim().toLowerCase());
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
 * fallback, and unlike audio the capability does not depend on optional
 * checkpoint assets, so the family answer is the whole answer.
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
  const advertised = model?.extend_default_overlap_frames;
  return typeof advertised === "number" && advertised > 0
    ? advertised
    : DEFAULT_EXTEND_OVERLAP_FRAMES;
}

/** Valid overlap choices for a clip of `frames` frames: `8k+1`, below frames. */
export function extendOverlapOptions(
  frames: number | null | undefined,
): number[] {
  const limit = frames ?? 0;
  const options: number[] = [];
  for (let overlap = 1; overlap < limit; overlap += 8) options.push(overlap);
  return options;
}

export type ExtendValidationInput = {
  overlapFrames: number | null | undefined;
  frames: number | null | undefined;
  hasSourceImage?: boolean;
  hasSourceVideo?: boolean;
  hasKeyframes?: boolean;
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
  if (overlap % 8 !== 1) {
    return `Overlap (${overlap}) must be 8k+1 (1, 9, 17, 25, …) so the carried frames re-encode cleanly through the LTX-2 video VAE.`;
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

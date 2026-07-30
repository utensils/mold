/**
 * Seam (transition) vocabulary — spec §11. The seam control names
 * transitions in words: Smooth / Cut / Fade, with the zero-motion-tail
 * fallback staying honest as "Join" (LTX-Video carries no motion across
 * the seam, so "Smooth" would over-promise; the editor row's description
 * still teaches "Clips join end to end"). Lives in ui/ so the
 * SeamPill/SeamEditor primitives can render labels without reaching up
 * into studio domain logic; studio re-exports these for its consumers.
 */

export type SequenceTransition = "smooth" | "cut" | "fade";

/** LTX-2's context handoff length; LTX-Video uses 0 ("Join"). */
export const DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES = 17;

export function transitionLabel(
  transition: SequenceTransition,
  motionTailFrames = DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
): string {
  if (transition === "smooth") {
    return motionTailFrames > 0 ? "Smooth" : "Join";
  }
  if (transition === "fade") return "Fade";
  return "Cut";
}

/** One-line teaching description per transition for the seam editor rows. */
export function transitionDescription(
  transition: SequenceTransition,
  motionTailFrames = DEFAULT_SEQUENCE_MOTION_TAIL_FRAMES,
): string {
  if (transition === "smooth") {
    return motionTailFrames > 0
      ? "Motion carries through"
      : "Clips join end to end";
  }
  if (transition === "fade") return "Blend across the seam";
  return "Hard change of shot";
}

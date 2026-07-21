/*
 * Advanced "N on" count — the badge on the Advanced button and drawer header.
 * A pure sum of the currently-active advanced fields so the badge, the drawer
 * header, and any tests all agree. Extends the prototype's advCount (negative +
 * source + loras + upscale + non-default scheduler) with custom size and video
 * non-defaults, per the Create spec. Capability gating is the caller's job:
 * pass a flag only for a field the current family actually exposes.
 */
import type { Scheduler } from "../../types";

export interface AdvancedCountParams {
  /** Negative-prompt text (counts when non-empty after trim). */
  negativePrompt: string;
  /** A source image is attached. */
  hasSource: boolean;
  /** Number of active LoRA adapters (each counts). */
  loraCount: number;
  /** Upscale-after-generate is enabled. */
  upscaleOn: boolean;
  /** Selected scheduler (counts when set and not "default"). */
  scheduler: Scheduler | null;
  /** The width/height fall outside the canonical shape × resolution grid. */
  customSize: boolean;
  /** A video family has non-default video controls set. */
  videoNonDefault: boolean;
}

/** Count of active advanced fields for the "N on" / "N active" badge. */
export function advancedActiveCount(p: AdvancedCountParams): number {
  return (
    (p.negativePrompt.trim() ? 1 : 0) +
    (p.hasSource ? 1 : 0) +
    Math.max(0, p.loraCount) +
    (p.upscaleOn ? 1 : 0) +
    (p.scheduler && p.scheduler !== "default" ? 1 : 0) +
    (p.customSize ? 1 : 0) +
    (p.videoNonDefault ? 1 : 0)
  );
}

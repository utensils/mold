/*
 * Advanced "N on" count — the badge on the Advanced entry and section header.
 * A pure sum of the currently-active advanced fields so the badge, the section
 * header, and any tests all agree. Extends the prototype's advCount (negative +
 * source + loras + upscale + non-default scheduler) with custom size, video
 * non-defaults, ControlNet, and the LTX-2 video suite, per the Create spec.
 * Capability gating is the caller's job: pass a flag only for a field the
 * current family actually exposes.
 */
import { negativePromptWireValue } from "@studio/lib/negativePrompt";
import type { Scheduler } from "../../types";

export interface AdvancedCountParams {
  /** Negative-prompt text (counts when it differs from the advertised
   * default — an untouched wan default is not user-active, an explicit
   * clear is; see `@studio/lib/negativePrompt`). */
  negativePrompt: string;
  /** The model's advertised default negative ("" when none). */
  negativePromptDefault?: string;
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
  /** A ControlNet guidance image + model is active (Advanced Source section).
   * Optional so callers that never surface ControlNet can omit it. */
  controlNet?: boolean;
  /** Any LTX-2 / video advanced control beyond frames/fps is set — pipeline,
   * audio, source video, keyframes, retake, spatial/temporal upscale, or the
   * GIF preview toggle. Counts once. Optional for the same reason. */
  videoSuite?: boolean;
  /** How many wan recipe controls (flow shift, distill strengths) are set.
   * Each counts, matching how LoRA rows do. Optional for the same reason. */
  wanRecipe?: number;
}

/** Count of active advanced fields for the "N on" / "N active" badge. */
export function advancedActiveCount(p: AdvancedCountParams): number {
  return (
    (negativePromptWireValue(
      p.negativePrompt,
      p.negativePromptDefault ?? "",
    ) !== undefined
      ? 1
      : 0) +
    (p.hasSource ? 1 : 0) +
    Math.max(0, p.loraCount) +
    (p.upscaleOn ? 1 : 0) +
    (p.scheduler && p.scheduler !== "default" ? 1 : 0) +
    (p.customSize ? 1 : 0) +
    (p.videoNonDefault ? 1 : 0) +
    (p.controlNet ? 1 : 0) +
    (p.videoSuite ? 1 : 0) +
    Math.max(0, p.wanRecipe ?? 0)
  );
}

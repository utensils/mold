/*
 * Advanced "N on" count — the badge on the Advanced entry and section header.
 * A pure sum of the currently-active advanced fields so the badge, the section
 * header, and any tests all agree. Source images and ControlNet moved to the
 * primary form (SourceMediaPanel) and no longer count here.
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
    Math.max(0, p.loraCount) +
    (p.upscaleOn ? 1 : 0) +
    (p.scheduler && p.scheduler !== "default" ? 1 : 0) +
    (p.customSize ? 1 : 0) +
    (p.videoNonDefault ? 1 : 0) +
    (p.videoSuite ? 1 : 0) +
    Math.max(0, p.wanRecipe ?? 0)
  );
}

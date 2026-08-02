/**
 * Projections between web's generate form and the shared sequence kit.
 *
 * `sequenceSharedParams` is THE stale-inspector fix: the sequence composer
 * used to keep private copies of width/steps/guidance that silently ignored
 * the inspector. The new submit path projects the LIVE form at submit time,
 * so what the controls say is what the chain request carries.
 */

import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import { DEFAULT_VIDEO_FPS } from "@studio/lib/sequence";
import type { GenerateFormState } from "../types";

/** Project the live web generate form onto the shared chain params. */
export function sequenceSharedParams(
  state: GenerateFormState,
  family: string,
): SequenceSharedParams {
  return {
    model: state.model,
    family,
    width: state.width,
    height: state.height,
    // The form carries the selected model's own `default_fps`; this fallback
    // only covers a form that never saw a video model.
    fps: state.fps ?? DEFAULT_VIDEO_FPS,
    steps: state.steps,
    guidance: state.guidance,
    strength: state.strength,
    sourceFitPolicy: state.sourceFitPolicy,
    upscalerModel: state.upscaleModel,
    seed:
      state.seedMode === "random" || state.seed === null
        ? ""
        : String(state.seed),
  };
}

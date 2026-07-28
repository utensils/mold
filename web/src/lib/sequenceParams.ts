/**
 * Projections between web's generate form and the shared sequence kit.
 *
 * `sequenceSharedParams` is THE stale-inspector fix: the sequence composer
 * used to keep private copies of width/steps/guidance that silently ignored
 * the inspector. The new submit path projects the LIVE form at submit time,
 * so what the controls say is what the chain request carries.
 */

import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
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
    fps: state.fps ?? 24,
    steps: state.steps,
    guidance: state.guidance,
    strength: state.strength,
    seed:
      state.seedMode === "random" || state.seed === null
        ? ""
        : String(state.seed),
  };
}

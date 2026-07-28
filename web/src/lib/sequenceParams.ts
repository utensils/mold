/**
 * Projections between web's generate form and the shared sequence kit.
 *
 * `sequenceSharedParams` is THE stale-inspector fix: the sequence composer
 * used to keep private copies of width/steps/guidance that silently ignored
 * the inspector. The new submit path projects the LIVE form at submit time,
 * so what the controls say is what the chain request carries.
 */

import type {
  ChainScript,
  ChainScriptStage,
} from "@studio/lib/api/chainTypes";
import type { SequenceSharedParams } from "@studio/lib/sequenceForm";
import type { ChainScriptWire, GenerateFormState } from "../types";

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

/**
 * Normalize the server's `ChainJobDetail.script` wire shape into the studio
 * `ChainScript`: Rust serde keys stages as `stage` and serializes seeds as
 * integers, while the studio projection uses `stages` and decimal-string
 * seeds (u64 exceeds `Number`'s safe range).
 */
export function chainScriptFromWire(
  wire: ChainScriptWire | null | undefined,
): ChainScript | null {
  if (!wire) return null;
  const chain = { ...wire.chain } as unknown as ChainScript["chain"];
  if (wire.chain.seed != null) chain.seed = String(wire.chain.seed);
  const stages: ChainScriptStage[] = wire.stage.map((stage) => ({
    prompt: stage.prompt,
    frames: stage.frames,
    ...(stage.transition ? { transition: stage.transition } : {}),
    ...(stage.fade_frames != null ? { fade_frames: stage.fade_frames } : {}),
    ...(stage.negative_prompt ? { negative_prompt: stage.negative_prompt } : {}),
    ...(stage.source_image ? { source_image_b64: stage.source_image } : {}),
    ...(stage.seed_offset != null
      ? { seed_offset: String(stage.seed_offset) }
      : {}),
  }));
  return { schema: wire.schema ?? "mold.chain.v1", chain, stages };
}

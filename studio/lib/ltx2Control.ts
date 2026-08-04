/**
 * Which LTX-2 pipeline a first-party IC-LoRA control adapter implies.
 *
 * Most adapters are just weights bolted onto the generic in-context pipeline.
 * Lip dub is not: its adapter stays loaded for both denoise stages, the
 * reference clip's speech is appended as negatively positioned audio tokens,
 * and stage 2's audio is frozen. Selecting it on `ic-lora` would load the
 * right weights and run the wrong graph, and the server rejects that pairing.
 *
 * `mold_core::ltx2_control::pipeline_for_control_id` is the authority;
 * `ltx2_control_pipeline_ts_mirror_matches_the_rust_registry` pins this to it.
 */
/**
 * The pipelines an IC-LoRA control adapter can select. A subset of each
 * shell's `Ltx2PipelineMode`, declared here so `studio/` stays independent of
 * both application shells.
 */
export type ControlAdapterPipeline = "ic-lora" | "lip-dub";

/** The one adapter id whose pipeline is not `ic-lora`. */
export const LIP_DUB_CONTROL_ID = "lipdub";

/** Normalise a control id the way the server does before matching. */
export function normalizeControlId(value: string): string {
  return value.trim().toLowerCase().replace(/_/g, "-");
}

export function pipelineForControlId(id: string): ControlAdapterPipeline {
  return normalizeControlId(id) === LIP_DUB_CONTROL_ID ? "lip-dub" : "ic-lora";
}

/** Whether a pipeline is one that an IC-LoRA control adapter drives. */
export function isControlAdapterPipeline(
  pipeline: string | null | undefined,
): pipeline is ControlAdapterPipeline {
  return pipeline === "ic-lora" || pipeline === "lip-dub";
}

/**
 * Lip dub takes its length and frame rate from the reference clip, so the
 * frames/fps a client sends are advisory at best. Surfaces say so instead of
 * letting a user set a duration that the server will replace.
 */
export const LIP_DUB_TIMING_HINT =
  "Lip dub matches the reference video: its frame count and frame rate come from the clip, not these fields.";

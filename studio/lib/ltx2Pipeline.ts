/**
 * The one LTX-2 pipeline that renders no frames.
 *
 * `t2a` produces a WAV and nothing else: no video container is valid, no
 * conditioning input has anywhere to land, and `enable_audio` has nothing
 * left to say. Every surface has to agree on that, and the surfaces that
 * spelled it `pipeline === "t2a"` inline drifted apart — desktop kept sending
 * a fresh form's `enable_audio: false` alongside `pipeline: t2a`, which the
 * server rejects outright.
 *
 * `mold_core::Ltx2PipelineMode::is_audio_only` is the authority.
 */
export const AUDIO_ONLY_PIPELINE = "t2a";

/** Whether a pipeline renders audio only — no frames, no video container. */
export function isAudioOnlyPipeline(
  pipeline: string | null | undefined,
): boolean {
  return pipeline === AUDIO_ONLY_PIPELINE;
}

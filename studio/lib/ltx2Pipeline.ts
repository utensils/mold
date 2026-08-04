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

/**
 * A completion event carrying an audio-only artifact.
 *
 * Structural on purpose: web and desktop declare their own completion types,
 * and both need the same probe. `audio_sample_rate` is the discriminator the
 * server sets for exactly this case.
 */
export interface AudioCompletionProbe {
  audio_sample_rate?: number | null;
}

/**
 * Whether a completed generation produced audio rather than a picture.
 *
 * Must be probed BEFORE any video or image branch. An audio print carries the
 * WAV in the event's `image` field and has no frames, so a video-shaped probe
 * falls through to the still branch and renders the WAV as an image.
 */
export function isAudioCompletion(
  result: AudioCompletionProbe | null | undefined,
): boolean {
  return result?.audio_sample_rate != null;
}

/**
 * Wire fields an audio-only render cannot carry.
 *
 * Mirrors `mold_core::validation`'s `Ltx2PipelineMode::T2a` arm, which refuses
 * every conditioning input and upscaler outright rather than ignoring media
 * the caller paid to upload. Three of these are not on that list but must
 * travel with it anyway: `mask_image` is rejected without `source_image`,
 * and `source_image_name` / `strength` describe an image that is no longer
 * there. The ControlNet trio is refused a layer earlier, by the family gate —
 * dropping it here means the user sees the error that actually matters.
 */
const AUDIO_ONLY_INCOMPATIBLE_FIELDS = [
  "source_image",
  "source_image_name",
  "strength",
  "mask_image",
  "source_video",
  "source_video_path",
  "audio_file",
  "audio_file_path",
  "extend_video",
  "extend_video_path",
  "extend_overlap_frames",
  "keyframes",
  "retake_range",
  "spatial_upscale",
  "temporal_upscale",
  "upscale_model",
  "control_image",
  "control_model",
  "control_scale",
] as const;

/**
 * Drop what an audio-only render has no place for, leaving the form itself
 * untouched.
 *
 * A user who sets a source video, switches to `t2a`, and hits Generate would
 * otherwise send hidden state the server refuses — the controls stay on
 * screen with a hint, but nothing was clearing the values. Stripping at
 * serialization rather than on the pipeline transition means toggling `t2a`
 * on and back does not silently destroy the clip they picked.
 *
 * `enable_audio: false` goes too: audio is what the pipeline renders, so the
 * flag can only contradict it. `true` and absent are both fine.
 */
export function stripAudioOnlyIncompatibleFields<
  T extends { pipeline?: string | null; enable_audio?: boolean | null },
>(request: T): T {
  if (!isAudioOnlyPipeline(request.pipeline)) return request;
  const next: Record<string, unknown> = { ...request };
  for (const field of AUDIO_ONLY_INCOMPATIBLE_FIELDS) delete next[field];
  if (next.enable_audio === false) delete next.enable_audio;
  return next as T;
}

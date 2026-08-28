/**
 * LTX-2 video-only (#1037) — the one browser-side policy for when the opt-in
 * may ride a request, mirrored on web and desktop so neither surface invents
 * its own conflict table. The server (`mold_core::validation`) stays the
 * authority; this exists so a blocked toggle explains itself inline instead
 * of surfacing as a 422.
 *
 * Skipping the audio branch is output-changing for the VIDEO (the branch
 * feeds the video stream through the a2v cross-attention), which is why the
 * control is labelled that way and is never a default.
 */
export interface VideoOnlyConflictInputs {
  /** The form's explicit audio-output opt-in (`enable_audio === true`). */
  audioEnabled: boolean;
  /** `pipeline === "t2a"` — there is no video to keep. */
  audioOnlyPipeline: boolean;
  /** An attached or server-path conditioning audio file. */
  hasConditioningAudio: boolean;
  /** A staged continuation (`extend_video` / `extend_video_path`). */
  isExtend: boolean;
}

/** Why `video_only` cannot ride the current form, or `null` when it can. */
export function videoOnlyBlockedReason(
  inputs: VideoOnlyConflictInputs,
): string | null {
  if (inputs.audioOnlyPipeline)
    return "Text-to-audio renders sound only; video-only does not apply.";
  if (inputs.audioEnabled)
    return "Turn off Generate audio first — video-only skips the branch that renders it.";
  if (inputs.hasConditioningAudio)
    return "Remove the conditioning audio first — video-only skips the branch it drives.";
  if (inputs.isExtend)
    return "A continuation keeps its source clip's rendering path.";
  return null;
}

/**
 * The wire value: `true` only for an enabled, conflict-free opt-in;
 * `undefined` otherwise so the field stays absent and the server's default
 * multimodal path remains authoritative.
 */
export function requestVideoOnly(
  enabled: boolean,
  inputs: VideoOnlyConflictInputs,
): true | undefined {
  return enabled && videoOnlyBlockedReason(inputs) === null ? true : undefined;
}

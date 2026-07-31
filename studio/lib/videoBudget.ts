/**
 * Client-side mirror of the LTX-2 temporal budget in
 * `crates/mold-core/src/validation.rs`.
 *
 * The LTX-2 checkpoints ship `pos_embed_max_pos = 20`, and both upstream
 * `ltx_core` and mold's RoPE path convert the temporal axis to *seconds*
 * before normalizing by it. So the ceiling is twenty seconds of runtime, not
 * twenty latent frames — which is why it moves with fps and cannot be written
 * down as a single frame count.
 *
 * Servers advertise the authoritative values (`max_runtime_seconds` /
 * `max_frames_absolute` on `/api/models`, `frames_per_clip_runtime_seconds` on
 * `/api/capabilities/chain-limits`). These constants are the fallback for
 * older servers that do not send them, and the server validator stays
 * authoritative either way.
 */

export const LTX2_MAX_RUNTIME_SECONDS = 20;

/** fps assumed when a caller has no request-level fps. Matches the manifest. */
export const LTX2_DEFAULT_FPS = 24;

/**
 * Resource guard applied on top of the duration budget, mirroring
 * `LTX2_MAX_FRAMES_ABSOLUTE`. 20s at 30 fps.
 */
export const LTX2_MAX_FRAMES_ABSOLUTE = LTX2_MAX_RUNTIME_SECONDS * 30 + 4;

/** Flat single-request frame ceiling for families without a duration budget. */
export const MAX_FRAMES_GLOBAL = 257;

/**
 * Largest pixel-frame count whose final RoPE token still lands inside the
 * LTX-2 temporal budget at `fps`.
 *
 * Latent frame `k` spans pixel bounds `[8k - 7, 8k + 1]` after the causal
 * first-frame fix, so `F` frames put the last midpoint at `(F - 4) / fps`
 * seconds: `F <= seconds * fps + 4`.
 */
export function ltx2MaxFramesAtFps(
  fps: number | null | undefined,
  runtimeSeconds: number = LTX2_MAX_RUNTIME_SECONDS,
  absolute: number = LTX2_MAX_FRAMES_ABSOLUTE,
): number {
  const effectiveFps = Math.max(1, Math.floor(fps ?? LTX2_DEFAULT_FPS));
  return Math.min(runtimeSeconds * effectiveFps + 4, absolute);
}

/**
 * Single-request frame ceiling for `family` at `fps`. `null` when the family
 * is not a video family and therefore publishes no frame ceiling.
 */
export function maxFramesForFamilyAtFps(
  family: string | null | undefined,
  fps: number | null | undefined,
): number | null {
  const normalized = (family ?? "").trim().toLowerCase();
  if (normalized === "ltx2" || normalized === "ltx-2") {
    return ltx2MaxFramesAtFps(fps);
  }
  if (normalized === "ltx-video") return MAX_FRAMES_GLOBAL;
  return null;
}

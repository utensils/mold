import { MINIMAX_H3_MAX_FRAMES } from "./minimaxH3Authoring";

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
 * `ltx2MaxFramesAtFps` snapped down onto the `8n+1` grid the server enforces.
 *
 * The raw ceiling is not requestable — 20 x 24 + 4 = 484 and 483 % 8 == 3 — so
 * a control clamped to it submits a frame count the server rejects. The
 * server advertises this value on `/api/models.max_frames`; this mirror is the
 * fallback for a host that predates it.
 */
export function ltx2MaxFramesOnGridAtFps(
  fps: number | null | undefined,
  runtimeSeconds: number = LTX2_MAX_RUNTIME_SECONDS,
  absolute: number = LTX2_MAX_FRAMES_ABSOLUTE,
): number {
  const cap = ltx2MaxFramesAtFps(fps, runtimeSeconds, absolute);
  return cap <= 1 ? 1 : cap - ((cap - 1) % 8);
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
    return ltx2MaxFramesOnGridAtFps(fps);
  }
  if (normalized === "ltx-video") return MAX_FRAMES_GLOBAL;
  // Wan's ceiling is a frame count, not a duration, so it does not move with
  // fps — the mirror of `"wan" => Some(MAX_FRAMES_GLOBAL)` in
  // `max_frames_for_family_at_fps`. Its temporal RoPE indexes *latent* frames
  // against a 1024-entry table, which 257 pixel frames comes nowhere near, so
  // memory is the binding constraint and the flat global guard is the real
  // limit. 257 also sits on Wan's `4k+1` grid, so the advertised maximum is
  // itself submittable.
  if (normalized === "wan") return MAX_FRAMES_GLOBAL;
  if (["minimax-h3", "minimax_h3", "minimaxh3"].includes(normalized)) {
    return MINIMAX_H3_MAX_FRAMES;
  }
  return null;
}

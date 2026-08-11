/**
 * LTX-2 frame-count validation shared by the desktop Advanced settings, the
 * single-video Generate path, and the iPhone parameters sheet. The math
 * mirrors `mold-core/src/validation.rs` (`frame_step_for_family`) at its LTX-2
 * arm; the Rust `is_ltx2_frame_count` spelling is now test-only. The old chain
 * composer's stitch math and TOML emitter moved to `@studio/lib/sequence`
 * and `@studio/lib/chainToml` with the unified Create switchover.
 */

/** LTX-2 pixel-frame constraint: counts of the form 8k+1 (1, 9, 17, …, 97). */
export function isLtx2FrameCount(n: number): boolean {
  return Number.isInteger(n) && n > 0 && n % 8 === 1;
}

/** Validation message for an invalid frame count, or null when it's fine. */
export function frames8n1Error(n: number): string | null {
  return isLtx2FrameCount(n) ? null : "Frames must be 8n+1 — try 97.";
}

/** Nearest valid 8k+1 count ≥ 1 — used to snap the frames stepper. */
export function snapFrames(n: number): number {
  if (n <= 1) return 1;
  const k = Math.round((n - 1) / 8);
  return Math.max(1, k * 8 + 1);
}

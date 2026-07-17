/**
 * Pure helpers for the chain composer: LTX-2 frame-count validation, stitched
 * frame / duration math, transition cycling, and a `mold.chain.v1` TOML
 * emitter. The frame math mirrors `mold-core/src/chain.rs`
 * (`estimated_total_frames`, `is_ltx2_frame_count`) so the footer forecast
 * matches what the server will actually stitch.
 */
import type { ChainScript, ChainStage, TransitionMode } from "./api/types";

export const DEFAULT_MOTION_TAIL_FRAMES = 17;
export const DEFAULT_FADE_FRAMES = 8;
export const DEFAULT_FPS = 24;
export const MAX_CHAIN_STAGES = 16;

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

/**
 * Stitched frame total before any top-level trim. Per-boundary rule (matches
 * chain.rs): smooth drops the incoming clip's leading `motionTail`; cut keeps
 * the whole clip; fade nets `-fadeFrames`.
 */
export function estimatedTotalFrames(stages: ChainStage[], motionTail: number): number {
  let total = 0;
  stages.forEach((stage, idx) => {
    if (idx === 0) {
      total += stage.frames;
      return;
    }
    const transition = stage.transition ?? "smooth";
    if (transition === "smooth") {
      total += Math.max(0, stage.frames - motionTail);
    } else if (transition === "cut") {
      total += stage.frames;
    } else {
      total += Math.max(0, stage.frames - (stage.fade_frames ?? DEFAULT_FADE_FRAMES));
    }
  });
  return total;
}

export function durationSeconds(totalFrames: number, fps: number): number {
  return fps > 0 ? totalFrames / fps : 0;
}

/** Click-cycle order for the splice glyph: smooth → cut → fade → smooth. */
export function cycleTransition(mode: TransitionMode): TransitionMode {
  return mode === "smooth" ? "cut" : mode === "cut" ? "fade" : "smooth";
}

/** Escape a string for a TOML basic (double-quoted) string. */
function tomlString(s: string): string {
  const escaped = s
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t");
  return `"${escaped}"`;
}

/**
 * Serialize a `ChainScript` to canonical `mold.chain.v1` TOML. Pure and
 * round-trip-shaped: emits the `[chain]` table and one `[[stage]]` table per
 * stage, omitting optional fields that are unset. Stage `source_image` bytes
 * are emitted as `source_image_b64` (the schema's inline authoring field) so
 * exported scripts round-trip through Open and `mold run --script` intact.
 */
export function chainToToml(script: ChainScript): string {
  const c = script.chain;
  const lines: string[] = ['schema = "mold.chain.v1"', "", "[chain]"];
  lines.push(`model = ${tomlString(c.model)}`);
  lines.push(`width = ${c.width}`);
  lines.push(`height = ${c.height}`);
  lines.push(`fps = ${c.fps}`);
  if (c.seed != null) lines.push(`seed = ${c.seed}`);
  lines.push(`steps = ${c.steps}`);
  lines.push(`guidance = ${c.guidance}`);
  lines.push(`strength = ${c.strength}`);
  lines.push(`motion_tail_frames = ${c.motion_tail_frames}`);
  lines.push(`output_format = ${tomlString(c.output_format)}`);
  if (c.enable_audio != null) lines.push(`enable_audio = ${c.enable_audio}`);

  for (const stage of script.stage) {
    lines.push("", "[[stage]]");
    lines.push(`prompt = ${tomlString(stage.prompt)}`);
    lines.push(`frames = ${stage.frames}`);
    lines.push(`transition = ${tomlString(stage.transition ?? "smooth")}`);
    if ((stage.transition ?? "smooth") === "fade" && stage.fade_frames != null) {
      lines.push(`fade_frames = ${stage.fade_frames}`);
    }
    if (stage.negative_prompt) lines.push(`negative_prompt = ${tomlString(stage.negative_prompt)}`);
    // `source_image_b64` is the mold.chain.v1 authoring form of the canonical
    // `source_image` wire field (see mold-core chain_toml.rs) — the CLI's
    // `read_script_resolving_paths` folds it back into `source_image`.
    if (stage.source_image) lines.push(`source_image_b64 = ${tomlString(stage.source_image)}`);
  }

  return lines.join("\n") + "\n";
}

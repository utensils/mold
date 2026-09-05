/*
 * The New-image workbench is a fixed-height flex column with
 * `overflow-hidden`, and in clip mode it stacks four incompressible pieces:
 * canvas, resizer, sequence bench, composer. Whatever the clamp does not
 * reserve, `overflow-hidden` eats off the BOTTOM — and the bottom is the
 * composer, which in clip mode carries the scene's words and the only
 * Generate button. So the reservation has to name every piece, not just the
 * canvas: reserving only a canvas floor is what let the default bench open
 * 216px taller than the window could show.
 *
 * The canvas floor here is the same number the canvas's own `min-height`
 * uses (the view binds it from `canvasFloor`), because a clamp that reserves
 * less than the CSS floor is a clamp that guarantees the overflow.
 */

/** The resizer strip between the canvas and the bench (`h-3`). */
export const BENCH_RESIZER_HEIGHT = 12;

/**
 * How little canvas a clip sequence may be squeezed to. Small on purpose: at
 * the 700px minimum window (`desktop/src-tauri/tauri.conf.json`) the bench's
 * own floor, the composer and the shell chrome leave exactly this much, and a
 * larger number would make the composer unreachable at every bench height the
 * user can drag to.
 */
export const MIN_SEQUENCE_CANVAS_HEIGHT = 144;

/** A one-shot has no bench, so its canvas keeps the roomier floor. */
export const MIN_STILL_CANVAS_HEIGHT = 320;

/** The timeline's own hard chrome floor (transport, ruler, lane, readout). */
export const MIN_SEQUENCE_BENCH_HEIGHT = 320;

/** A one-shot never renders the bench; this is the stored-height floor. */
export const MIN_BENCH_HEIGHT = 280;

/**
 * Opening height for a fresh clip sequence. Chosen so the default 860px
 * window (`tauri.conf.json`) shows canvas, timeline and composer at once
 * without the clamp having to trim it on mount.
 */
export const DEFAULT_SEQUENCE_BENCH_HEIGHT = 380;

/**
 * Height the composer occupies before it has been measured: one prompt row
 * plus one control row plus the card's padding. The view replaces this with
 * the real measurement as soon as the element exists.
 */
export const COMPOSER_FALLBACK_HEIGHT = 114;

export interface BenchClampInput {
  /** The height the user (or storage) asked for. */
  requested: number;
  /** The workbench's own client height — it contains all four pieces. */
  available: number;
  /** Floor for the bench itself. */
  minBench: number;
  /** Floor for the canvas above it. */
  canvasFloor: number;
  /** Height of the drag strip, normally `BENCH_RESIZER_HEIGHT`. */
  resizerHeight: number;
  /** Measured composer height, or `COMPOSER_FALLBACK_HEIGHT`. */
  composerHeight: number;
}

/**
 * The largest bench height that still leaves the canvas its floor, the
 * resizer its strip, and the composer every pixel it measured. Never below
 * `minBench`: at a window too small for all four the bench keeps its floor
 * and the canvas is the piece that gives, because a timeline squeezed under
 * its chrome floor shows nothing at all.
 */
export function benchHeightCeiling(input: Omit<BenchClampInput, "requested">): number {
  const reserved = input.canvasFloor + input.resizerHeight + input.composerHeight;
  return Math.max(input.minBench, input.available - reserved);
}

/** Clamp a requested bench height into `[minBench, benchHeightCeiling]`. */
export function clampBenchHeight(input: BenchClampInput): number {
  return Math.round(Math.min(benchHeightCeiling(input), Math.max(input.minBench, input.requested)));
}

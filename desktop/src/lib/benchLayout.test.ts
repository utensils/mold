import { describe, expect, it } from "vitest";
import {
  BENCH_RESIZER_HEIGHT,
  COMPOSER_FALLBACK_HEIGHT,
  DEFAULT_SEQUENCE_BENCH_HEIGHT,
  MIN_SEQUENCE_BENCH_HEIGHT,
  MIN_SEQUENCE_CANVAS_HEIGHT,
  benchHeightCeiling,
  clampBenchHeight,
} from "./benchLayout";

/*
 * The workbench is `overflow-hidden`, so whatever the clamp fails to reserve
 * is cut off the bottom — and the bottom is the composer, which in clip mode
 * is the only Generate button. These pin the reservation itself rather than
 * any one number.
 */

// TitleBar + the per-view toolbar + the status bar (ui/mold-desktop.css).
const SHELL_CHROME = 44 + 40 + 26;
/** Window heights from desktop/src-tauri/tauri.conf.json. */
const DEFAULT_WINDOW = 860;
const MIN_WINDOW = 700;

function sequenceClamp(windowHeight: number, requested: number): number {
  return clampBenchHeight({
    requested,
    available: windowHeight - SHELL_CHROME,
    minBench: MIN_SEQUENCE_BENCH_HEIGHT,
    canvasFloor: MIN_SEQUENCE_CANVAS_HEIGHT,
    resizerHeight: BENCH_RESIZER_HEIGHT,
    composerHeight: COMPOSER_FALLBACK_HEIGHT,
  });
}

/** What the four stacked pieces actually need at a given bench height. */
function stackHeight(bench: number, composer = COMPOSER_FALLBACK_HEIGHT): number {
  return MIN_SEQUENCE_CANVAS_HEIGHT + BENCH_RESIZER_HEIGHT + bench + composer;
}

describe("bench height clamp", () => {
  it("reserves the canvas floor, the resizer AND the composer", () => {
    const available = 900;
    expect(
      benchHeightCeiling({
        available,
        minBench: MIN_SEQUENCE_BENCH_HEIGHT,
        canvasFloor: MIN_SEQUENCE_CANVAS_HEIGHT,
        resizerHeight: BENCH_RESIZER_HEIGHT,
        composerHeight: 130,
      }),
    ).toBe(available - (MIN_SEQUENCE_CANVAS_HEIGHT + BENCH_RESIZER_HEIGHT + 130));
  });

  it("gives a taller composer back its pixels", () => {
    const short = benchHeightCeiling({
      available: 900,
      minBench: MIN_SEQUENCE_BENCH_HEIGHT,
      canvasFloor: MIN_SEQUENCE_CANVAS_HEIGHT,
      resizerHeight: BENCH_RESIZER_HEIGHT,
      composerHeight: 114,
    });
    const tall = benchHeightCeiling({
      available: 900,
      minBench: MIN_SEQUENCE_BENCH_HEIGHT,
      canvasFloor: MIN_SEQUENCE_CANVAS_HEIGHT,
      resizerHeight: BENCH_RESIZER_HEIGHT,
      composerHeight: 174,
    });
    expect(short - tall).toBe(60);
  });

  it("opens the default sequence bench unclipped at the default window", () => {
    const bench = sequenceClamp(DEFAULT_WINDOW, DEFAULT_SEQUENCE_BENCH_HEIGHT);
    expect(bench).toBe(DEFAULT_SEQUENCE_BENCH_HEIGHT);
    expect(stackHeight(bench)).toBeLessThanOrEqual(DEFAULT_WINDOW - SHELL_CHROME);
  });

  it("still fits every piece at the minimum window", () => {
    const bench = sequenceClamp(MIN_WINDOW, DEFAULT_SEQUENCE_BENCH_HEIGHT);
    expect(stackHeight(bench)).toBeLessThanOrEqual(MIN_WINDOW - SHELL_CHROME);
  });

  it("trims a stored height that no longer fits", () => {
    // 520 is the pre-fix stored default; it must not survive a small window.
    expect(sequenceClamp(MIN_WINDOW, 520)).toBeLessThan(520);
    expect(stackHeight(sequenceClamp(MIN_WINDOW, 520))).toBeLessThanOrEqual(
      MIN_WINDOW - SHELL_CHROME,
    );
  });

  it("keeps the timeline's own chrome floor when the window is too small for all four", () => {
    // Below this the canvas is the piece that gives: a bench under its floor
    // shows neither transport nor readout.
    expect(sequenceClamp(560, DEFAULT_SEQUENCE_BENCH_HEIGHT)).toBe(MIN_SEQUENCE_BENCH_HEIGHT);
  });

  it("never returns a fractional height", () => {
    expect(
      clampBenchHeight({
        requested: 401.6,
        available: 4000,
        minBench: MIN_SEQUENCE_BENCH_HEIGHT,
        canvasFloor: MIN_SEQUENCE_CANVAS_HEIGHT,
        resizerHeight: BENCH_RESIZER_HEIGHT,
        composerHeight: COMPOSER_FALLBACK_HEIGHT,
      }),
    ).toBe(402);
  });
});

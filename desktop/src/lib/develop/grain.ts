/**
 * The Develop — mold's signature element. Every generation is visualized as
 * a print developing: a deterministic grain field, seeded from the job's
 * actual seed, that organizes and warms in lockstep with real DenoiseStep
 * events, then fixes into the final image.
 *
 * Pure math lives here so it is testable; DevelopCanvas.vue owns the rAF.
 */

export type DevelopPhase = "latent" | "developing" | "fixed" | "stopped";

/** Safelight tokens the shader tints between (cold → warm → stop). */
export const GRAIN_COLD = { r: 147, g: 167, b: 176 }; // Halide
export const GRAIN_WARM = { r: 240, g: 138, b: 36 }; // Safelight
export const GRAIN_STOP = { r: 201, g: 79, b: 61 }; // Stop

/** Deterministic 32-bit PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fold any seed (bigint-scale numbers, strings) into a 32-bit PRNG seed. */
export function foldSeed(seed: number | string): number {
  const s = String(seed);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export interface GrainParams {
  /** Noise opacity 0–1: falls as the print develops. */
  amplitude: number;
  /** Correlation length in px: grain organizes from fine to coarse blotches. */
  cell: number;
  /** 0 = cold Halide, 1 = warm Safelight. */
  warmth: number;
}

/**
 * Map denoise progress (0–1) to grain parameters. Monotonic on purpose:
 * amplitude only falls, cell only grows, warmth only rises.
 */
export function grainParams(progress: number, phase: DevelopPhase): GrainParams {
  const p = Math.min(1, Math.max(0, progress));
  if (phase === "latent") return { amplitude: 0.9, cell: 2, warmth: 0 };
  if (phase === "stopped") return { amplitude: 0.55, cell: 2 + 22 * p, warmth: 0 };
  if (phase === "fixed") return { amplitude: 0, cell: 24, warmth: 1 };
  return {
    amplitude: 0.9 - 0.75 * p,
    cell: 2 + 22 * p,
    warmth: p,
  };
}

/** Per-seed monochrome noise tile; the grain structure of this negative. */
export function makeGrainTile(seed: number | string, size = 96): ImageData {
  const rand = mulberry32(foldSeed(seed));
  const data = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const v = Math.floor(rand() * 256);
    data[i * 4] = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  return new ImageData(data, size, size);
}

export function tint(warmth: number, stopped = false): { r: number; g: number; b: number } {
  if (stopped) return GRAIN_STOP;
  const w = Math.min(1, Math.max(0, warmth));
  return {
    r: Math.round(GRAIN_COLD.r + (GRAIN_WARM.r - GRAIN_COLD.r) * w),
    g: Math.round(GRAIN_COLD.g + (GRAIN_WARM.g - GRAIN_COLD.g) * w),
    b: Math.round(GRAIN_COLD.b + (GRAIN_WARM.b - GRAIN_COLD.b) * w),
  };
}

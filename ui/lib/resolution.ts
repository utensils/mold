/*
 * Megapixel-based resolution math (spec §03 — Resolution selector).
 * Dimensions derive from a target megapixel budget and an aspect ratio,
 * rounded to the legacy /64 grid, with a /16 under-ceiling correction for
 * the 1.8 MP tier used by newer model contracts.
 */

export interface AspectOption {
  id: string;
  /** Human label, e.g. "1:1". */
  label: string;
  /** width / height. */
  ratio: number;
}

/** The five canonical shapes (spec §03 — Shape picker). */
export const ASPECTS: readonly AspectOption[] = [
  { id: "square", label: "1:1", ratio: 1 },
  { id: "portrait", label: "3:4", ratio: 3 / 4 },
  { id: "landscape", label: "4:3", ratio: 4 / 3 },
  { id: "wide", label: "16:9", ratio: 16 / 9 },
  { id: "tall", label: "9:16", ratio: 9 / 16 },
];

/** The megapixel steps with their human labels (spec §03). */
export const RESOLUTIONS = [
  { mp: 0.5, label: "0.5 MP", sub: "Draft" },
  { mp: 1, label: "1 MP", sub: "Standard" },
  { mp: 1.8, label: "1.8 MP", sub: "Large" },
] as const;

function snap64(value: number): number {
  return Math.max(64, Math.round(value / 64) * 64);
}

/** Resolved pixel dimensions for a megapixel budget at an aspect ratio. */
export function dimsForMp(
  mp: number,
  ratio: number,
): { width: number; height: number } {
  const width = Math.sqrt(mp * 1e6 * ratio);
  let snappedWidth = snap64(width);
  let snappedHeight = snap64(width / ratio);
  const maxPixels = mp * 1e6;
  if (snappedWidth * snappedHeight > maxPixels && mp >= 1.8) {
    const scale = Math.sqrt(maxPixels / (snappedWidth * snappedHeight));
    snappedWidth = Math.max(16, Math.floor((snappedWidth * scale) / 16) * 16);
    snappedHeight = Math.max(16, Math.floor((snappedHeight * scale) / 16) * 16);
  }
  return { width: snappedWidth, height: snappedHeight };
}

/** "1024×1024" display form. */
export function dimsLabel(mp: number, ratio: number): string {
  const { width, height } = dimsForMp(mp, ratio);
  return `${width}×${height}`;
}

/**
 * Closest megapixel step for explicit dimensions — used when reusing prints
 * or restoring persisted forms so the selector reflects reality.
 */
export function nearestMp(width: number, height: number): number {
  const mp = (width * height) / 1e6;
  let best: number = RESOLUTIONS[0].mp;
  for (const step of RESOLUTIONS) {
    if (Math.abs(step.mp - mp) < Math.abs(best - mp)) best = step.mp;
  }
  return best;
}

/**
 * Proportional swatch box for a shape (spec: "the swatch is the actual
 * ratio"). The five canonical shapes use the prototype's exact pixel boxes;
 * anything else falls back to fitting the ratio in a 27×27 box.
 */
const SWATCH_BOXES: Record<string, { width: number; height: number }> = {
  "1": { width: 24, height: 24 },
  [String(3 / 4)]: { width: 18, height: 24 },
  [String(4 / 3)]: { width: 24, height: 18 },
  [String(16 / 9)]: { width: 27, height: 15 },
  [String(9 / 16)]: { width: 15, height: 27 },
};

export function swatchDims(ratio: number): { width: number; height: number } {
  const exact = SWATCH_BOXES[String(ratio)];
  if (exact) return exact;
  const max = 27;
  return ratio >= 1
    ? { width: max, height: Math.max(6, Math.round(max / ratio)) }
    : { width: Math.max(6, Math.round(max * ratio)), height: max };
}

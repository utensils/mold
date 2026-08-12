import { orientationLabel, type ResolutionPreset } from "../lib/resolutions";

export type ResolutionOrientation = ReturnType<typeof orientationLabel>;

/** Phone-first order: the neutral square first, then the two directional choices. */
export const MOBILE_RESOLUTION_ORIENTATIONS: readonly ResolutionOrientation[] = [
  "Square",
  "Portrait",
  "Landscape",
];

export function presetsForOrientation(
  presets: readonly ResolutionPreset[],
  orientation: ResolutionOrientation,
): ResolutionPreset[] {
  return presets.filter((preset) => orientationLabel(preset.width, preset.height) === orientation);
}

/** Preserve desktop ordering while collapsing repeated ratios into one chip. */
export function aspectsForOrientation(
  presets: readonly ResolutionPreset[],
  orientation: ResolutionOrientation,
): string[] {
  return [...new Set(presetsForOrientation(presets, orientation).map((preset) => preset.aspect))];
}

/** Pick the available bucket whose pixel area is closest to the current size. */
export function closestResolutionPreset(
  presets: readonly ResolutionPreset[],
  width: number,
  height: number,
): ResolutionPreset | null {
  if (presets.length === 0) return null;
  const targetArea = Math.max(1, width * height);
  return presets.reduce((closest, candidate) => {
    const closestDelta = Math.abs(closest.width * closest.height - targetArea);
    const candidateDelta = Math.abs(candidate.width * candidate.height - targetArea);
    return candidateDelta < closestDelta ? candidate : closest;
  });
}

/** Desktop generation also snaps manual dimensions to the engine's latent stride. */
export function snapMobileDimension(value: number, minimum = 64, alignment = 16): number {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return minimum;
  const grid = Math.max(1, Math.round(alignment));
  return Math.max(minimum, Math.round(numeric / grid) * grid);
}

export function resolutionTierLabel(index: number, total: number): string {
  if (total <= 1) return "Recommended";
  if (total === 2) return index === 0 ? "Standard" : "High";
  if (total === 3) return ["Compact", "Standard", "High"][index] ?? "High";
  if (total === 4) return ["Compact", "Standard", "High", "Max"][index] ?? "Max";
  if (index === 0) return "Compact";
  if (index === total - 1) return "Max";
  return `Tier ${index + 1}`;
}

export function sortedResolutionTiers(
  presets: readonly ResolutionPreset[],
  aspect: string,
): ResolutionPreset[] {
  return presets
    .filter((preset) => preset.aspect === aspect)
    .sort((left, right) => left.width * left.height - right.width * right.height);
}

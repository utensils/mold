/*
 * Resolution projection — the Create controls keep width/height as the
 * persisted source of truth (see `useGenerateForm`); the Shape picker and
 * Resolution selector are PROJECTIONS of those pixels. This derives the
 * canonical aspect id + megapixel step for an explicit width/height, and flags
 * when the pixels fall outside the canonical grid ("custom size").
 *
 * Matching is by EXACT canonical dims, not ratio tolerance: the /64 snap in
 * `dimsForMp` shifts a shape's ratio by up to ~10% at small sizes (e.g. 16:9 @
 * 0.5 MP → 960×512), so a ratio comparison can't reliably recover the shape a
 * canonical projection came from. We instead match the pixels against every
 * shape × MP projection.
 */
import { ASPECTS, RESOLUTIONS, dimsForMp, nearestMp } from "@ui/lib/resolution";

export interface ResolutionProjection {
  /** Canonical aspect id, or `null` for a custom (off-grid) size. */
  aspectId: string | null;
  /** Nearest megapixel step for the current pixels. */
  mp: number;
  /** True when the pixels don't land on the canonical shape × MP grid. */
  isCustom: boolean;
}

/** Project explicit pixels onto the Shape + Resolution controls. */
export function projectResolution(
  width: number,
  height: number,
): ResolutionProjection {
  for (const a of ASPECTS) {
    for (const r of RESOLUTIONS) {
      const d = dimsForMp(r.mp, a.ratio);
      if (d.width === width && d.height === height) {
        return { aspectId: a.id, mp: r.mp, isCustom: false };
      }
    }
  }
  return { aspectId: null, mp: nearestMp(width, height), isCustom: true };
}

/** The canonical aspect id for a width/height, or `null` when it's custom.
 * Drives the Shape picker's highlighted chip. */
export function aspectIdForDims(width: number, height: number): string | null {
  return projectResolution(width, height).aspectId;
}

/**
 * The default canvas a `canvas: last-reference` recipe takes from its
 * references (`ReferenceImagesProfile.canvas`, Qwen Image 2.1).
 *
 * `GenerateRequest.width`/`height` are required, so the server cannot tell a
 * chosen size from a default one: the rule is a CLIENT rule and admission
 * stays exact. While the canvas intent is `model-default` — the user has not
 * picked a size — every surface sizes the canvas to the LAST reference's
 * aspect at upstream's fixed 1024x1024 area (`output_resolution` defaults to
 * 1024, `pipeline_qwenimage21.py:527`, `:621-623`), on the recipe's grid,
 * clamped into the recipe's advertised bounds. Never the recipe's default
 * size: that is a host's configuration, and the server's advisory derives
 * the canvas from the fixed area. A manual canvas never moves.
 *
 * This is `mold_core::validation::last_reference_canvas`, exactly:
 * `fit_to_target_area_ties_even` mirrors diffusers' `calculate_dimensions`
 * (`pipeline_qwenimage21.py:149-156` at `e0abab83b`): `width = sqrt(area *
 * ratio)`, `height = width / ratio`, each rounded with Python's `round()` —
 * halves to EVEN — onto the grid, so a client and the engine never land on
 * different sides of a tie (a 4225x4096 reference is 1024x1024 upstream,
 * 1056x1024 under half-away-from-zero). `clampCanvasToLimits` is mold's
 * deliberate divergence: upstream caps nothing, so a panorama wider than
 * about 7.3:1 would derive a width past the 2752 px axis ceiling and be
 * refused at admission for a size the user never chose.
 */

import type {
  ReferenceCanvasRule,
  ResolutionProfile,
} from "./generated/generationProfileV1";
import {
  imageDimensionsFromBase64,
  type ImageDimensions,
} from "./imageDimensions";
import type { CanvasIntent } from "./outputShape";

/** Python's `round()` for a finite float: halves go to the even neighbour. */
export function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) return floor + 1;
  if (fraction < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** `fit_to_target_area_ties_even(src_w, src_h, area, align)`, exactly. */
export function fitToTargetAreaTiesEven(
  sourceWidth: number,
  sourceHeight: number,
  targetArea: number,
  alignment: number,
): ImageDimensions {
  const align = Math.max(1, alignment);
  const ratio = Math.max(1, sourceWidth) / Math.max(1, sourceHeight);
  const width = Math.sqrt(targetArea * ratio);
  const height = width / ratio;
  // A degenerate aspect that rounds an axis to zero is lifted to one cell.
  const snap = (value: number) =>
    Math.max(1, roundHalfToEven(value / align)) * align;
  return { width: snap(width), height: snap(height) };
}

/** `mold_core::validation::LAST_REFERENCE_CANVAS_AREA`: upstream's 1024². */
export const LAST_REFERENCE_CANVAS_AREA = 1024 * 1024;

/** The advertised bounds a derived canvas must land inside. */
export type CanvasLimits = Pick<
  ResolutionProfile,
  "alignment" | "min_width" | "min_height" | "max_pixels" | "max_axis_pixels"
>;

/**
 * `mold_core::validation::clamp_canvas_to_limits`, exactly: integer cells
 * only. The long side walks down one grid cell at a time from the axis
 * ceiling, the short side follows by FLOORED proportion and is lifted to its
 * minimum, until both the axis and the pixel ceilings hold. A canvas already
 * inside is returned unchanged.
 */
export function clampCanvasToLimits(
  width: number,
  height: number,
  limits: CanvasLimits,
): ImageDimensions {
  const align = Math.max(1, limits.alignment);
  const axis = limits.max_axis_pixels ?? null;
  const fits = (w: number, h: number) =>
    w * h <= limits.max_pixels && (axis === null || (w <= axis && h <= axis));
  if (fits(width, height)) return { width, height };
  const landscape = width >= height;
  const long = landscape ? width : height;
  const short = landscape ? height : width;
  const minCells = (pixels: number) => Math.max(1, Math.ceil(pixels / align));
  const longMin = minCells(landscape ? limits.min_width : limits.min_height);
  const shortMin = minCells(landscape ? limits.min_height : limits.min_width);
  let longCells = Math.max(1, Math.floor(long / align));
  if (axis !== null) {
    longCells = Math.min(longCells, Math.max(1, Math.floor(axis / align)));
  }
  for (;;) {
    const shortCells = Math.max(
      Math.floor((longCells * short) / long),
      shortMin,
    );
    const lw = longCells * align;
    const sh = shortCells * align;
    if (fits(lw, sh) || longCells <= longMin) {
      return landscape ? { width: lw, height: sh } : { width: sh, height: lw };
    }
    longCells -= 1;
  }
}

/** `mold_core::validation::last_reference_canvas`, exactly. */
export function lastReferenceCanvas(
  referenceWidth: number,
  referenceHeight: number,
  limits: CanvasLimits,
): ImageDimensions {
  const fitted = fitToTargetAreaTiesEven(
    referenceWidth,
    referenceHeight,
    LAST_REFERENCE_CANVAS_AREA,
    limits.alignment,
  );
  return clampCanvasToLimits(fitted.width, fitted.height, limits);
}

export interface ReferenceCanvasInput {
  /** The recipe's advertised rule; `null` is no rule, or an older server. */
  canvas: ReferenceCanvasRule | null | undefined;
  /**
   * The ordered references' decoded sizes; `null` where a reference's header
   * could not be read.
   */
  references: readonly (ImageDimensions | null)[];
  /** The recipe's advertised `resolution`: its grid and its bounds. */
  resolution: CanvasLimits;
  intent: CanvasIntent;
}

/**
 * The canvas the rule asks for, or `null` when the canvas should be left
 * alone: no rule, a canvas the user chose, an empty strip, or a last
 * reference whose size is not known yet (never guess — the next read settles
 * it). An empty strip answers `null` rather than the recipe default because
 * surfaces also run this when a restored draft hydrates, and a restored
 * canvas must not be silently reset; emptying the strip keeps the last
 * reference's shape, which Reset returns to the default.
 */
export function referenceCanvasSize(
  input: ReferenceCanvasInput,
): ImageDimensions | null {
  if (input.canvas !== "last-reference") return null;
  if (input.intent !== "model-default") return null;
  if (input.references.length === 0) return null;
  const last = input.references[input.references.length - 1];
  if (!last) return null;
  return lastReferenceCanvas(last.width, last.height, input.resolution);
}

/** The strip's staged images, in any surface's own shape. */
export interface StagedReferenceImage {
  /** Raw base64 or a data URL; empty/null is a bytes-less reattach entry. */
  base64?: string | null;
  data?: string | null;
  width?: number | null;
  height?: number | null;
}

/**
 * Each staged reference's size: the dimensions the picker already recorded,
 * else read from the header (PNG, JPEG or WebP). `null` where neither is
 * known — a bytes-less reattach entry, or an unreadable header.
 */
export function stagedReferenceDimensions(
  images: readonly StagedReferenceImage[],
): (ImageDimensions | null)[] {
  return images.map((image) => {
    if (image.width && image.height) {
      return { width: image.width, height: image.height };
    }
    const bytes = image.base64 || image.data;
    // Every container a reference strip may hold; admission, not this read,
    // decides which the recipe accepts.
    return bytes
      ? imageDimensionsFromBase64(bytes, ["png", "jpeg", "webp"])
      : null;
  });
}

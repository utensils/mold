/**
 * The default canvas a `canvas: last-reference` recipe takes from its
 * references (`ReferenceImagesProfile.canvas`, Qwen Image 2.1).
 *
 * `GenerateRequest.width`/`height` are required, so the server cannot tell a
 * chosen size from a default one: the rule is a CLIENT rule and admission
 * stays exact. While the canvas intent is `model-default` — the user has not
 * picked a size — every surface sizes the canvas to the LAST reference's
 * aspect at the recipe's default pixel area, on its alignment grid. A manual
 * canvas never moves.
 *
 * The arithmetic is `mold_core::validation::fit_to_target_area_ties_even`,
 * which mirrors diffusers' `calculate_dimensions`
 * (`pipeline_qwenimage21.py:149-156` at `e0abab83b`): `width = sqrt(area *
 * ratio)`, `height = width / ratio`, each rounded with Python's `round()` —
 * halves to EVEN — onto the grid. The CLI and the engine use the same rule, so
 * a client and the engine never land on different sides of a tie (a 4225x4096
 * reference is 1024x1024 upstream, 1056x1024 under half-away-from-zero).
 */

import type { ReferenceCanvasRule } from "./generated/generationProfileV1";
import {
  orientedImageDimensionsFromBase64,
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

export interface ReferenceCanvasInput {
  /** The recipe's advertised rule; `null` is no rule, or an older server. */
  canvas: ReferenceCanvasRule | null | undefined;
  /**
   * The ordered references' decoded sizes; `null` where a reference's header
   * could not be read.
   */
  references: readonly (ImageDimensions | null)[];
  /** The recipe default canvas; its AREA is the target. */
  defaults: ImageDimensions;
  /** The recipe's resolution alignment (32 on Qwen Image 2.1). */
  alignment: number;
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
  return fitToTargetAreaTiesEven(
    last.width,
    last.height,
    input.defaults.width * input.defaults.height,
    input.alignment,
  );
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
 * Each staged reference's UPRIGHT size: read from the header with its EXIF
 * orientation applied (PNG, JPEG or WebP), which is how the engine decodes a
 * reference and how the CLI and the server size the same canvas; else the
 * dimensions the picker recorded. `null` where neither is known — a
 * bytes-less reattach entry, or an unreadable header.
 *
 * The bytes win over a recorded size because a picker records the stored
 * header (a portrait phone photo is landscape pixels plus `Orientation = 6`),
 * and a canvas sized from that would come out sideways.
 */
export function stagedReferenceDimensions(
  images: readonly StagedReferenceImage[],
): (ImageDimensions | null)[] {
  return images.map((image) => {
    const bytes = image.base64 || image.data;
    // Every container a reference strip may hold; admission, not this read,
    // decides which the recipe accepts.
    const read = bytes
      ? orientedImageDimensionsFromBase64(bytes, ["png", "jpeg", "webp"])
      : null;
    if (read) return read;
    return image.width && image.height
      ? { width: image.width, height: image.height }
      : null;
  });
}

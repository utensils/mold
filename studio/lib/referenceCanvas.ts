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
import type { ImageDimensions } from "./imageDimensions";
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
 * alone: no rule, a canvas the user chose, or a last reference whose size is
 * not known yet (never guess — the next read settles it). With no references
 * the answer is the recipe default, so emptying the strip gives the default
 * canvas back.
 */
export function referenceCanvasSize(
  input: ReferenceCanvasInput,
): ImageDimensions | null {
  if (input.canvas !== "last-reference") return null;
  if (input.intent !== "model-default") return null;
  if (input.references.length === 0) {
    return { width: input.defaults.width, height: input.defaults.height };
  }
  const last = input.references[input.references.length - 1];
  if (!last) return null;
  return fitToTargetAreaTiesEven(
    last.width,
    last.height,
    input.defaults.width * input.defaults.height,
    input.alignment,
  );
}

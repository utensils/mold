/**
 * User crop rectangle for a MiniMax H3 Ref2VA IMAGE reference — one policy
 * for web, desktop, and iPhone.
 *
 * This is deliberately NOT a fit-to-canvas policy. The server normalizes
 * every oversized image reference DOWN onto its own 2048-short-edge canvas
 * and keeps a smaller one at native geometry — never upscaling
 * (`mold_core::minimax_h3::reference_image_dimensions`), so the `SourceFitPolicy`
 * modes are architecturally wrong for references; the only client-side
 * decision that exists is "which part of this photograph is the reference".
 * The crop is applied at the ORIGINAL resolution before digest/upload — no
 * resampling here, the server resamples — and it never touches the print's
 * `width`/`height` or its `CanvasIntent`.
 */
import { FAMILY_LOG_TOLERANCE } from "./outputShape";
import type { Rect, Size, SourceFitTransform } from "./sourceFit";
import type { GenerationReferenceCrop } from "./generationReferences";

/** Integer rectangle in source pixels. */
export type ReferenceCrop = Rect;

/** Below this the 32 px vision-pad grid has nothing left to describe. */
export const REFERENCE_CROP_MIN_AXIS = 64;

export type ReferenceCropAspectId = "free" | "1:1" | "4:3" | "3:2" | "16:9";

export interface ReferenceCropAspect {
  id: ReferenceCropAspectId;
  label: string;
  /** Landscape ratio; `null` is Free. Portrait sources take the reciprocal. */
  ratio: number | null;
}

/**
 * The quick presets: Free plus the canonical output-shape families that have
 * a distinct landscape/portrait pair (5:4 and 21:9 are deliberately absent —
 * a reference is a subject, not a canvas).
 */
export const REFERENCE_CROP_ASPECTS: readonly ReferenceCropAspect[] = [
  { id: "free", label: "Free", ratio: null },
  { id: "1:1", label: "1:1", ratio: 1 },
  { id: "4:3", label: "4:3", ratio: 4 / 3 },
  { id: "3:2", label: "3:2", ratio: 3 / 2 },
  { id: "16:9", label: "16:9", ratio: 16 / 9 },
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function minAxis(extent: number): number {
  return Math.min(REFERENCE_CROP_MIN_AXIS, extent);
}

/** Clamp a rect into `size` as integers, at least 64 px on each axis. */
export function normalizeReferenceCrop(crop: Rect, size: Size): ReferenceCrop {
  const minWidth = minAxis(size.width);
  const minHeight = minAxis(size.height);
  // The origin wins over the extent: a rect restored from provenance keeps
  // where it started and loses only what no longer fits.
  const x = clamp(Math.round(crop.x), 0, size.width - minWidth);
  const y = clamp(Math.round(crop.y), 0, size.height - minHeight);
  const width = clamp(Math.round(crop.width), minWidth, size.width - x);
  const height = clamp(Math.round(crop.height), minHeight, size.height - y);
  return { x, y, width, height };
}

export function fullReferenceCrop(size: Size): ReferenceCrop {
  return { x: 0, y: 0, width: size.width, height: size.height };
}

/** True when the crop is absent or covers the whole source. */
export function referenceCropIsIdentity(
  crop: ReferenceCrop | null | undefined,
  size: Size,
): boolean {
  return (
    !crop ||
    (crop.x === 0 &&
      crop.y === 0 &&
      crop.width === size.width &&
      crop.height === size.height)
  );
}

/** The preset's ratio in the source's own orientation. */
function orientedRatio(aspect: ReferenceCropAspect, size: Size): number | null {
  if (aspect.ratio === null) return null;
  return size.height > size.width ? 1 / aspect.ratio : aspect.ratio;
}

/** The largest centered rect of that aspect (Free = the whole source). */
export function referenceCropForAspect(
  size: Size,
  aspectId: ReferenceCropAspectId,
): ReferenceCrop {
  const aspect = REFERENCE_CROP_ASPECTS.find((entry) => entry.id === aspectId);
  const ratio = aspect ? orientedRatio(aspect, size) : null;
  if (ratio === null) return fullReferenceCrop(size);
  let width = size.width;
  let height = width / ratio;
  if (height > size.height) {
    height = size.height;
    width = height * ratio;
  }
  return normalizeReferenceCrop(
    {
      x: (size.width - width) / 2,
      y: (size.height - height) / 2,
      width,
      height,
    },
    size,
  );
}

/** Which preset a rect already matches (within the output-shape tolerance). */
export function referenceCropAspectId(
  crop: ReferenceCrop,
  size: Size,
): ReferenceCropAspectId {
  const ratio = crop.width / crop.height;
  let best: ReferenceCropAspectId = "free";
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const aspect of REFERENCE_CROP_ASPECTS) {
    const oriented = orientedRatio(aspect, size);
    if (oriented === null) continue;
    const distance = Math.abs(Math.log(ratio / oriented));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = aspect.id;
    }
  }
  return bestDistance <= FAMILY_LOG_TOLERANCE ? best : "free";
}

export type ReferenceCropCorner = "nw" | "ne" | "sw" | "se";

/**
 * Resize by dragging one corner; the opposite corner stays anchored. With a
 * locked `ratio` the width follows the pointer and the height derives from
 * it, shrinking both when the derived height would leave the source.
 */
export function resizeReferenceCropFromCorner(
  crop: ReferenceCrop,
  corner: ReferenceCropCorner,
  pointer: { x: number; y: number },
  size: Size,
  ratio: number | null,
): ReferenceCrop {
  const anchorX =
    corner === "nw" || corner === "sw" ? crop.x + crop.width : crop.x;
  const anchorY =
    corner === "nw" || corner === "ne" ? crop.y + crop.height : crop.y;
  const towardLeft = corner === "nw" || corner === "sw";
  const towardTop = corner === "nw" || corner === "ne";
  const maxWidth = towardLeft ? anchorX : size.width - anchorX;
  const maxHeight = towardTop ? anchorY : size.height - anchorY;
  let width = clamp(
    towardLeft ? anchorX - pointer.x : pointer.x - anchorX,
    minAxis(size.width),
    maxWidth,
  );
  let height = clamp(
    towardTop ? anchorY - pointer.y : pointer.y - anchorY,
    minAxis(size.height),
    maxHeight,
  );
  if (ratio !== null) {
    height = width / ratio;
    if (height > maxHeight) {
      height = maxHeight;
      width = height * ratio;
    }
    if (height < minAxis(size.height)) {
      height = minAxis(size.height);
      width = height * ratio;
    }
  }
  return normalizeReferenceCrop(
    {
      x: towardLeft ? anchorX - width : anchorX,
      y: towardTop ? anchorY - height : anchorY,
      width,
      height,
    },
    size,
  );
}

/** Translate the rect, stopping at the source edges. */
export function moveReferenceCrop(
  crop: ReferenceCrop,
  dx: number,
  dy: number,
  size: Size,
): ReferenceCrop {
  // A move keeps the extent and only slides the origin; `normalize` would
  // let the origin win and shrink the rect against the far edge.
  const width = Math.min(crop.width, size.width);
  const height = Math.min(crop.height, size.height);
  return normalizeReferenceCrop(
    {
      x: clamp(Math.round(crop.x + dx), 0, size.width - width),
      y: clamp(Math.round(crop.y + dy), 0, size.height - height),
      width,
      height,
    },
    size,
  );
}

/**
 * The crop as the same transform shape `resolveSourceFitTransform` returns,
 * so `SourceFitCanvasOps.fitImage` executes it unchanged: a crop-sized canvas
 * with the whole source drawn at full size, offset by the crop origin.
 */
export function applyReferenceCropTransform(
  crop: ReferenceCrop,
  source: Size,
): SourceFitTransform {
  return {
    outputWidth: crop.width,
    outputHeight: crop.height,
    drawWidth: source.width,
    drawHeight: source.height,
    // `0 - 0` is `-0`, which a deep-equality check treats as a different value.
    offsetX: crop.x === 0 ? 0 : -crop.x,
    offsetY: crop.y === 0 ? 0 : -crop.y,
    maskPaddedPixels: false,
  };
}

const REFERENCE_SHORT_EDGE = 2048;
const REFERENCE_ALIGNMENT = 32;

/** `f64::round_ties_even`, which `aligned_dimension` uses on the server. */
function roundTiesEven(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function alignedDimension(value: number): number {
  return Math.max(
    REFERENCE_ALIGNMENT,
    roundTiesEven(value / REFERENCE_ALIGNMENT) * REFERENCE_ALIGNMENT,
  );
}

export interface ReferencePadEstimate {
  normalizedWidth: number;
  normalizedHeight: number;
  /** Qwen vision pads = 32 px cells of the normalized canvas. */
  pads: number;
}

/**
 * Preview of the reference's admission cost — the TS mirror of
 * `reference_image_dimensions` (down-only 2048-short-edge, 32-aligned) followed by
 * `rows_per_video_latent` (`(w/32)*(h/32)`). A hint only; the server's
 * `reference_prepared_shapes_for_target` stays the authority.
 */
export function referencePadEstimate(
  source: Size,
  crop?: ReferenceCrop | null,
): ReferencePadEstimate {
  const size = crop ?? source;
  // Down-only, mirroring the server (and ComfyUI's `min(1.0, 2048/short)`):
  // a reference already inside the 2048-short-edge canvas keeps its native
  // geometry rather than being upscaled onto it.
  const scale = Math.min(
    1,
    REFERENCE_SHORT_EDGE / Math.min(size.width, size.height),
  );
  const normalizedWidth = alignedDimension(size.width * scale);
  const normalizedHeight = alignedDimension(size.height * scale);
  return {
    normalizedWidth,
    normalizedHeight,
    pads:
      (normalizedWidth / REFERENCE_ALIGNMENT) *
      (normalizedHeight / REFERENCE_ALIGNMENT),
  };
}

/** The additive per-reference provenance the wire carries once applied. */
export function referenceCropProvenance(
  crop: ReferenceCrop,
  source: Size,
  sourceSha256: string,
): GenerationReferenceCrop {
  return {
    x: crop.x,
    y: crop.y,
    width: crop.width,
    height: crop.height,
    source_width: source.width,
    source_height: source.height,
    source_sha256: sourceSha256,
  };
}

/** Read the draft rect back out of saved provenance. */
export function referenceCropFromProvenance(
  provenance: GenerationReferenceCrop,
): ReferenceCrop {
  return {
    x: provenance.x,
    y: provenance.y,
    width: provenance.width,
    height: provenance.height,
  };
}

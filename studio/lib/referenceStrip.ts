/**
 * The ordered reference strip every Create surface renders (web, desktop,
 * iPhone/Android) through `studio/components/ReferenceImageStrip.vue`.
 *
 * Order is part of the request: `edit_images` ships in strip order, the
 * prompt addresses the pictures by position ("the jacket from image 1"), and
 * on a `canvas: last-reference` recipe (Qwen Image 2.1) the LAST picture sets
 * the canvas shape. So each tile is numbered exactly the way the expander's
 * `GENERATION CONTEXT` block numbers it (`mold_core::prompting::
 * render_generation_context` — "image N" over the request's references in
 * order, `source_image` before `edit_images`), and the one tile that sets the
 * canvas says so.
 *
 * The list helpers are non-mutating; a no-op (same index, out of range)
 * returns the INPUT array so callers can cheaply skip an update.
 */

import type { SourceImageMode } from "./generationCapabilities";
import type { ReferenceCanvasRule } from "./generated/generationProfileV1";

export function reorderReference<T>(
  list: readonly T[],
  fromIndex: number,
  toIndex: number,
): T[] {
  if (fromIndex === toIndex) return list as T[];
  if (fromIndex < 0 || fromIndex >= list.length) return list as T[];
  if (toIndex < 0 || toIndex >= list.length) return list as T[];
  const next = list.slice();
  const [item] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, item as T);
  return next;
}

export function moveReference<T>(
  list: readonly T[],
  index: number,
  delta: -1 | 1,
): T[] {
  return reorderReference(list, index, index + delta);
}

export interface ReferenceStripItem {
  /** Position in the strip's own list. */
  index: number;
  /** The 1-based position the prompt and the expander use ("image 2"). */
  ordinal: number;
  /** `Image <ordinal>` — the tile's name, matching the prompt's addressing. */
  label: string;
  /** `Target` for a target-first recipe's first picture (Qwen edit). */
  role: "Target" | "Reference";
  /** This picture sets the canvas shape (`canvas: last-reference`). */
  setsCanvas: boolean;
}

export interface ReferenceStripInput {
  count: number;
  /** The first picture is the edit Target (`primary_is_target`). */
  firstIsTarget?: boolean;
  /** The recipe's canvas follows the LAST reference (`stripSetsCanvas`). */
  setsCanvas?: boolean;
  /** Images the request carries BEFORE the strip (`referenceOrdinalBase`). */
  ordinalBase?: number;
}

export function referenceStripItems(
  input: ReferenceStripInput,
): ReferenceStripItem[] {
  const base = input.ordinalBase ?? 0;
  return Array.from({ length: Math.max(0, input.count) }, (_, index) => {
    const ordinal = base + index + 1;
    return {
      index,
      ordinal,
      label: `Image ${ordinal}`,
      role: input.firstIsTarget && index === 0 ? "Target" : "Reference",
      setsCanvas: Boolean(input.setsCanvas) && index === input.count - 1,
    };
  });
}

/**
 * How many images the request carries ahead of the strip. Only an ADDITIVE
 * (IP-Adapter) recipe ships `source_image` beside `edit_images`, and the
 * expander lists the source first, so its reference is image 2 while a source
 * is held. Klein ships one or the other; Qwen edit's Target is
 * `edit_images[0]`; FLUX.2 [dev] and Qwen Image 2.1 have no source at all.
 */
export function referenceOrdinalBase(
  mode: SourceImageMode,
  hasSource: boolean,
): number {
  return mode === "single-and-references" && hasSource ? 1 : 0;
}

/**
 * Whether the strip's last picture sets the canvas: the recipe advertises
 * `canvas: last-reference` and the strip is the whole conditioning — exactly
 * the case every surface's `referenceCanvasSize` watcher applies.
 */
export function stripSetsCanvas(
  mode: SourceImageMode,
  canvas: ReferenceCanvasRule | null | undefined,
): boolean {
  return mode === "references" && canvas === "last-reference";
}

/** The strip's canvas sentence, beside the "Sets canvas" badge. */
export const REFERENCE_CANVAS_NOTE =
  "The last image sets the canvas shape unless you pick a size.";

// ── Pointer drag-to-reorder ───────────────────────────────────────────────
// The strip reorders with Pointer Events, never HTML5 drag-and-drop: under
// Tauri's default `dragDropEnabled` the native layer can swallow HTML5 DnD
// inside the webview (WebView2 on Windows), and an HTML5 drag is also what
// an OS FILE drop arrives as — one mechanism for each keeps them apart.

/** How far (CSS px, either axis) a press travels before it is a drag, so a
 * click on a tile never reorders anything. */
const REORDER_DRAG_THRESHOLD_PX = 5;

export function pastReorderThreshold(dx: number, dy: number): boolean {
  return (
    Math.abs(dx) > REORDER_DRAG_THRESHOLD_PX ||
    Math.abs(dy) > REORDER_DRAG_THRESHOLD_PX
  );
}

interface TileRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** The tile under the pointer, by index into `rects`; `null` over a gap or
 * outside the strip. */
export function referenceIndexAtPoint(
  rects: readonly TileRect[],
  x: number,
  y: number,
): number | null {
  const index = rects.findIndex(
    (r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom,
  );
  return index < 0 ? null : index;
}

/** Which edge of the target tile the insertion mark sits on: a picture
 * moved earlier lands before the target, one moved later lands after it
 * (`reorderReference` semantics). */
export function referenceDropSide(
  from: number,
  to: number,
): "before" | "after" {
  return from > to ? "before" : "after";
}

/** The live-region sentence after a pointer reorder, in the prompt's
 * numbering. */
export function referenceMoveAnnouncement(
  from: number,
  to: number,
  ordinalBase: number,
): string {
  return `Moved image ${ordinalBase + from + 1} to position ${ordinalBase + to + 1}.`;
}

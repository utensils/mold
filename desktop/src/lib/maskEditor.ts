/**
 * Pure helpers for the desktop mask editor. Extracted from the paint-canvas
 * component so the fiddly bits — filename derivation, pixel inversion, bounded
 * undo history, and pointer→canvas coordinate scaling — are unit-testable
 * without a live 2D context (happy-dom has none). Logic mirrors the web SPA's
 * `MaskEditorModal.vue`.
 */

/** Derive the mask filename from the source filename: `stem + "-mask.png"`.
 * Falls back to `source-mask.png` when the source name is missing or blank. */
export function maskFilenameFor(filename: string | null | undefined): string {
  const base = filename && filename.trim() ? filename : "source.png";
  const stem = base.replace(/\.[^.]+$/, "");
  return `${stem}-mask.png`;
}

/**
 * Invert the RGB channels of RGBA pixel data in place. A fully transparent
 * pixel (alpha 0) is promoted to opaque so the inverted region actually paints
 * — otherwise inverting an empty canvas would produce nothing visible.
 */
export function invertPixels(data: Uint8ClampedArray): void {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255 - (data[i] ?? 0);
    data[i + 1] = 255 - (data[i + 1] ?? 0);
    data[i + 2] = 255 - (data[i + 2] ?? 0);
    const alpha = data[i + 3] ?? 0;
    data[i + 3] = alpha === 0 ? 255 : alpha;
  }
}

/** Append `item` to `stack`, keeping at most `cap` (default 20) most-recent
 * entries. Pure — returns a new array, never mutates the input. */
export function pushCapped<T>(stack: T[], item: T, cap = 20): T[] {
  return [...stack, item].slice(-cap);
}

export interface CanvasPoint {
  x: number;
  y: number;
}

/**
 * Map a pointer's client coordinates into canvas pixel coordinates, accounting
 * for the CSS scaling between the rendered rect and the canvas's intrinsic
 * (drawing-buffer) size. Guards against a zero-size rect (divide-by-zero).
 */
export function pointerToCanvas(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  canvas: { width: number; height: number },
): CanvasPoint {
  const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
  const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

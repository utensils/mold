/**
 * img2img source-fit policies — how a source image whose dimensions differ
 * from the requested width x height is mapped onto the generation canvas.
 * Verbatim port of the web SPA's `web/src/lib/sourceFit.ts` (framework-free);
 * the `SourceFitPolicy` union lives here because the desktop app keeps wire
 * types in `api/types.ts` and this is a client-side concept.
 */

export type SourceFitPolicy =
  | { mode: "pad-repaint" }
  | { mode: "pad-fit" }
  | {
      mode: "crop-fill";
      alignX?: "left" | "center" | "right";
      alignY?: "top" | "center" | "bottom";
    }
  | { mode: "lanczos-resize" }
  | {
      mode: "upscale-then-fit";
      upscalerModel: string;
      fit: Exclude<SourceFitPolicy, { mode: "upscale-then-fit" }>;
    };

export type SourceFitMode = SourceFitPolicy["mode"];

/**
 * Source-fit choices shared by every source-image surface. Sequence video is
 * maskless, so callers omit Pad + repaint and use the remaining four modes.
 */
export const SOURCE_FIT_OPTIONS: readonly {
  value: SourceFitMode;
  label: string;
}[] = [
  { value: "pad-repaint", label: "Pad + repaint" },
  { value: "crop-fill", label: "Crop fill" },
  { value: "pad-fit", label: "Scale to fit" },
  { value: "lanczos-resize", label: "Resize to fill" },
  { value: "upscale-then-fit", label: "Upscale + crop" },
];

const ALIGN_X = new Set(["left", "center", "right"]);
const ALIGN_Y = new Set(["top", "center", "bottom"]);

/**
 * Defensive parse of a policy recovered from provenance (the additive
 * `source_fit` metadata field, echoed verbatim by the server). Anything that
 * is not exactly a wire-shaped policy returns null so a corrupt or future
 * value can never poison the live form.
 */
export function parseSourceFitPolicy(value: unknown): SourceFitPolicy | null {
  if (typeof value !== "object" || value === null) return null;
  const candidate = value as Record<string, unknown>;
  switch (candidate.mode) {
    case "pad-repaint":
    case "pad-fit":
    case "lanczos-resize":
      return { mode: candidate.mode };
    case "crop-fill": {
      const alignX = candidate.alignX;
      const alignY = candidate.alignY;
      if (alignX !== undefined && !ALIGN_X.has(alignX as string)) return null;
      if (alignY !== undefined && !ALIGN_Y.has(alignY as string)) return null;
      return {
        mode: "crop-fill",
        ...(alignX !== undefined
          ? { alignX: alignX as "left" | "center" | "right" }
          : {}),
        ...(alignY !== undefined
          ? { alignY: alignY as "top" | "center" | "bottom" }
          : {}),
      };
    }
    case "upscale-then-fit": {
      if (typeof candidate.upscalerModel !== "string") return null;
      const fit = parseSourceFitPolicy(candidate.fit);
      if (!fit || fit.mode === "upscale-then-fit") return null;
      return {
        mode: "upscale-then-fit",
        upscalerModel: candidate.upscalerModel,
        fit,
      };
    }
    default:
      return null;
  }
}

/** Build the complete policy represented by a compact mode control. */
export function sourceFitPolicyForMode(
  mode: SourceFitMode,
  options: { supportsMask: boolean; upscalerModel?: string },
): SourceFitPolicy {
  if (mode === "crop-fill") {
    return { mode, alignX: "center", alignY: "center" };
  }
  if (mode === "upscale-then-fit") {
    return {
      mode,
      upscalerModel: options.upscalerModel ?? "",
      fit: options.supportsMask
        ? { mode: "pad-repaint" }
        : { mode: "crop-fill", alignX: "center", alignY: "center" },
    };
  }
  if (mode === "pad-repaint" && !options.supportsMask) {
    return { mode: "crop-fill", alignX: "center", alignY: "center" };
  }
  return { mode };
}

export interface Size {
  width: number;
  height: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SourceFitTransform {
  outputWidth: number;
  outputHeight: number;
  drawWidth: number;
  drawHeight: number;
  offsetX: number;
  offsetY: number;
  maskPaddedPixels: boolean;
}

function alignOffset(
  available: number,
  align: "left" | "center" | "right" | "top" | "bottom" | undefined,
): number {
  if (align === "left" || align === "top") return 0;
  if (align === "right" || align === "bottom") return available;
  return Math.round(available / 2);
}

export function resolveSourceFitTransform(
  source: Size,
  target: Size,
  policy: SourceFitPolicy,
): SourceFitTransform {
  const fit = policy.mode === "upscale-then-fit" ? policy.fit : policy;
  const outputWidth = target.width;
  const outputHeight = target.height;
  if (fit.mode === "lanczos-resize") {
    return {
      outputWidth,
      outputHeight,
      drawWidth: outputWidth,
      drawHeight: outputHeight,
      offsetX: 0,
      offsetY: 0,
      maskPaddedPixels: false,
    };
  }

  const sourceRatio = source.width / source.height;
  const targetRatio = target.width / target.height;
  const crop = fit.mode === "crop-fill";
  const scale = crop
    ? targetRatio > sourceRatio
      ? target.width / source.width
      : target.height / source.height
    : targetRatio < sourceRatio
      ? target.width / source.width
      : target.height / source.height;
  const drawWidth = Math.round(source.width * scale);
  const drawHeight = Math.round(source.height * scale);
  const availableX = outputWidth - drawWidth;
  const availableY = outputHeight - drawHeight;
  let offsetX = crop
    ? -alignOffset(-availableX, fit.alignX)
    : Math.round(availableX / 2);
  let offsetY = crop
    ? -alignOffset(-availableY, fit.alignY)
    : Math.round(availableY / 2);
  if (Object.is(offsetX, -0)) offsetX = 0;
  if (Object.is(offsetY, -0)) offsetY = 0;

  return {
    outputWidth,
    outputHeight,
    drawWidth,
    drawHeight,
    offsetX,
    offsetY,
    maskPaddedPixels: fit.mode === "pad-repaint",
  };
}

export function maskPaddingRectangles(t: SourceFitTransform): Rect[] {
  if (!t.maskPaddedPixels) return [];
  const rects: Rect[] = [];
  const left = Math.max(0, t.offsetX);
  const top = Math.max(0, t.offsetY);
  const right = Math.max(0, t.outputWidth - (t.offsetX + t.drawWidth));
  const bottom = Math.max(0, t.outputHeight - (t.offsetY + t.drawHeight));
  if (top > 0) rects.push({ x: 0, y: 0, width: t.outputWidth, height: top });
  if (bottom > 0)
    rects.push({
      x: 0,
      y: t.outputHeight - bottom,
      width: t.outputWidth,
      height: bottom,
    });
  if (left > 0)
    rects.push({
      x: 0,
      y: top,
      width: left,
      height: t.outputHeight - top - bottom,
    });
  if (right > 0)
    rects.push({
      x: t.outputWidth - right,
      y: top,
      width: right,
      height: t.outputHeight - top - bottom,
    });
  return rects.filter((r) => r.width > 0 && r.height > 0);
}

/**
 * Rewrite a fit policy for families that can't ship a repaint mask (video
 * img2img — LTX-2): pad-repaint would paint pad bands the model can never
 * repaint, so it becomes a centered crop-fill (which always fills the target).
 * Applied both when entering such a family and defensively on submit.
 */
export function coerceSourceFitForMaskless(
  policy: SourceFitPolicy,
): SourceFitPolicy {
  if (policy.mode === "pad-repaint") {
    return { mode: "crop-fill", alignX: "center", alignY: "center" };
  }
  if (policy.mode === "upscale-then-fit" && policy.fit.mode === "pad-repaint") {
    return {
      ...policy,
      fit: { mode: "crop-fill", alignX: "center", alignY: "center" },
    };
  }
  return policy;
}

export function describeSourceFit(policy: SourceFitPolicy): string {
  switch (policy.mode) {
    case "pad-repaint":
      return "Pad to fit and repaint added pixels";
    case "pad-fit":
      return "Pad to fit";
    case "crop-fill":
      return "Crop fill";
    case "lanczos-resize":
      return "Lanczos resize";
    case "upscale-then-fit":
      return `Upscale with ${policy.upscalerModel}, then ${describeSourceFit(policy.fit)}`;
  }
}

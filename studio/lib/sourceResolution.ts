import {
  dimensionAlignmentForFamily,
  maxAxisPixelsForModel,
  sourceMaxPixelsForModel,
  type ModelResolutionContract,
} from "./resolutions";
import { effectiveGenerationRecipe } from "./generationProfile";

export interface SourceDimensions {
  width: number;
  height: number;
}

export type SourceResolutionReason =
  "exact" | "aligned" | "downscaled" | "minimum";

export interface SourceResolutionResult {
  source: SourceDimensions;
  output: SourceDimensions;
  maxPixels: number;
  alignment: number;
  minimumDimension: number;
  reason: SourceResolutionReason;
  /** False only for a contradictory model contract whose pixel ceiling is
   * smaller than its minimum aligned canvas. */
  fitsModel: boolean;
}

export interface SourceResolutionStatus {
  label: "Source" | "Adjusted" | "Downscaled";
  detail: string;
}

function positiveInteger(
  value: number | null | undefined,
  fallback: number,
): number {
  return Number.isFinite(value) && Number(value) > 0
    ? Math.max(1, Math.round(Number(value)))
    : fallback;
}

function floorAligned(value: number, alignment: number): number {
  return Math.floor(value / alignment) * alignment;
}

function minimumAligned(alignment: number): number {
  return Math.ceil(64 / alignment) * alignment;
}

function familyOf(model: ModelResolutionContract | string): string | undefined {
  return typeof model === "string" ? model : model.family;
}

function familyAlignment(model: ModelResolutionContract | string): number {
  return dimensionAlignmentForFamily(familyOf(model));
}

/**
 * Resolve a source image into a model-safe generation canvas.
 *
 * Eligible source images are never enlarged: exact aligned dimensions remain
 * exact, while over-budget or unaligned dimensions only move downward. The
 * sole exception is an input dimension below the generation contract's 64 px
 * minimum, where the minimum valid aligned canvas necessarily wins.
 *
 * Recommended dimensions are presets, not bounds. The server-advertised
 * source ceiling (falling back to the output ceiling) and alignment are the
 * actual custom-canvas contract, so preserving the source aspect is the
 * authority here.
 */
export function resolveSourceResolution(
  source: SourceDimensions,
  model: ModelResolutionContract | string,
  pipeline?: string | null,
): SourceResolutionResult {
  const sourceWidth = positiveInteger(source.width, 1);
  const sourceHeight = positiveInteger(source.height, 1);
  const contract = typeof model === "string" ? null : model;
  const recipe = contract
    ? effectiveGenerationRecipe(contract, pipeline)
    : null;
  const alignment = positiveInteger(
    recipe?.resolution.alignment ?? contract?.dimension_alignment,
    familyAlignment(model),
  );
  const minimumDimension = Math.max(
    minimumAligned(alignment),
    recipe?.resolution.min_width ?? 0,
    recipe?.resolution.min_height ?? 0,
  );
  const maxPixels = positiveInteger(
    sourceMaxPixelsForModel(model, pipeline),
    sourceMaxPixelsForModel(familyOf(model)),
  );
  const sourcePixels = sourceWidth * sourceHeight;
  // The per-axis span is a second, independent bound. It matters most for an
  // extreme aspect ratio: an 8000x600 source is under any pixel budget once
  // scaled, and its long edge would still land far outside the span LTX-2
  // normalizes RoPE positions by.
  const maxAxis = maxAxisPixelsForModel(model, pipeline);
  const longestSourceAxis = Math.max(sourceWidth, sourceHeight);
  const scale = Math.min(
    1,
    Math.sqrt(maxPixels / sourcePixels),
    maxAxis === null ? 1 : maxAxis / longestSourceAxis,
  );

  let width = Math.max(
    minimumDimension,
    floorAligned(sourceWidth * scale, alignment),
  );
  let height = Math.max(
    minimumDimension,
    floorAligned(sourceHeight * scale, alignment),
  );

  // Flooring both axes normally guarantees the cap. The minimum can make an
  // extremely thin source exceed it, so reduce whichever non-minimum axis
  // least harms the original aspect until no further valid reduction exists.
  while (
    width * height > maxPixels ||
    (maxAxis !== null && Math.max(width, height) > maxAxis)
  ) {
    const canReduceWidth = width > minimumDimension;
    const canReduceHeight = height > minimumDimension;
    if (!canReduceWidth && !canReduceHeight) break;
    // An axis violation has one cure: shorten that axis. The aspect-preserving
    // heuristic below would otherwise shave the *other* one, distorting the
    // frame without ever satisfying the bound.
    if (maxAxis !== null && Math.max(width, height) > maxAxis) {
      if (width >= height && canReduceWidth) width -= alignment;
      else if (canReduceHeight) height -= alignment;
      else width -= alignment;
      continue;
    }
    if (!canReduceHeight) {
      width -= alignment;
      continue;
    }
    if (!canReduceWidth) {
      height -= alignment;
      continue;
    }
    const sourceRatio = sourceWidth / sourceHeight;
    const widthRatioError = Math.abs(
      (width - alignment) / height - sourceRatio,
    );
    const heightRatioError = Math.abs(
      width / (height - alignment) - sourceRatio,
    );
    if (widthRatioError <= heightRatioError) width -= alignment;
    else height -= alignment;
  }

  const belowMinimum =
    sourceWidth < minimumDimension || sourceHeight < minimumDimension;
  const downscaled = scale < 1;
  const exact = width === sourceWidth && height === sourceHeight;
  const reason: SourceResolutionReason = exact
    ? "exact"
    : belowMinimum
      ? "minimum"
      : downscaled
        ? "downscaled"
        : "aligned";

  return {
    source: { width: sourceWidth, height: sourceHeight },
    output: { width, height },
    maxPixels,
    alignment,
    minimumDimension,
    reason,
    fitsModel:
      width * height <= maxPixels &&
      (maxAxis === null || Math.max(width, height) <= maxAxis),
  };
}

/** True when the current canvas still follows the model-safe source result. */
export function canvasMatchesSourceResolution(
  canvas: SourceDimensions,
  result: SourceResolutionResult,
): boolean {
  return (
    canvas.width === result.output.width &&
    canvas.height === result.output.height
  );
}

/** Human-ready copy shared by the three Create surfaces. */
export function sourceResolutionStatus(
  result: SourceResolutionResult,
): SourceResolutionStatus {
  const source = `${result.source.width}×${result.source.height}`;
  const output = `${result.output.width}×${result.output.height}`;
  if (result.reason === "exact") {
    return { label: "Source", detail: `Matches source · ${output}` };
  }
  if (result.reason === "downscaled") {
    return {
      label: "Downscaled",
      detail: `${source} → ${output} · ${formatMegapixels(result.maxPixels)} limit, ${result.alignment} px aligned`,
    };
  }
  if (result.reason === "minimum") {
    return {
      label: "Adjusted",
      detail: `${source} → ${output} · minimum ${result.minimumDimension} px, ${result.alignment} px aligned`,
    };
  }
  return {
    label: "Adjusted",
    detail: `${source} → ${output} · ${result.alignment} px aligned`,
  };
}

function formatMegapixels(pixels: number): string {
  const value = (pixels / 1_000_000).toFixed(1).replace(/\.0$/, "");
  return `${value} MP`;
}

// Keep the model contract structurally visible to API consumers that build a
// synthetic contract from a family plus server-advertised dimensions.
export type { ModelResolutionContract };

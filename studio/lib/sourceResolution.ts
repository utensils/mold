import {
  dimensionAlignmentForFamily,
  maxAxisPixelsForModel,
  presetsForModel,
  sourceMaxPixelsForModel,
  type ModelResolutionContract,
} from "./resolutions";
import { effectiveGenerationRecipe } from "./generationProfile";
import type { CanvasIntent } from "./outputShape";

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

export interface SourceCanvasTransition {
  source: SourceResolutionResult;
  automatic: SourceDimensions;
  replaced: boolean;
  intent: CanvasIntent;
  preserveReplacement?: boolean;
}

/** A preset outside this log-ratio distance is a different shape, not a
 * source-following canvas. This matches the canonical family tolerance used
 * by the shared output-shape resolver without importing it circularly. */
export const SOURCE_PRESET_ASPECT_LOG_TOLERANCE = 0.06;

/**
 * Preserve authored/restored canvases while keeping source-driven defaults
 * live across model changes. All Create controllers use this transition so a
 * programmatic Reuse/edit import is never mistaken for a fresh drag.
 *
 * The decision is the recorded {@link CanvasIntent} and nothing else. It used
 * to also require the live canvas to still equal the value this function last
 * returned, which no surface can honour: selecting a model runs
 * `applyModelDefaults` — writing the new model's default width/height —
 * *before* the source watcher runs, so the comparison always failed and the
 * canvas stopped following the source on the first model switch (#1166).
 */
export function resolveSourceCanvasTransition({
  source,
  automatic,
  replaced,
  intent,
  preserveReplacement = false,
}: SourceCanvasTransition): SourceDimensions | null {
  if (intent === "source-exact") return source.output;
  if (replaced) return preserveReplacement ? null : automatic;
  return intent === "source" ? automatic : null;
}

/**
 * Pick the model-authored canvas a newly attached source should use.
 *
 * Aspect distance is measured logarithmically so portrait and landscape are
 * treated symmetrically. Once the closest aspect is known, prefer the tier
 * nearest the recipe's default pixel area; older hosts without recipe
 * defaults use the largest advertised tier. A model with no advertised
 * presets retains the safe custom-source fallback.
 *
 * This is only the automatic choice. `resolveSourceResolution` remains the
 * authority behind the explicit "Source" / "Match source" control.
 */
export function resolveDefaultSourceResolution(
  source: SourceDimensions,
  model: ModelResolutionContract | string,
  pipeline?: string | null,
): SourceDimensions {
  const presets = presetsForModel(model, pipeline);
  if (presets.length === 0) {
    return resolveSourceResolution(source, model, pipeline).output;
  }

  const sourceRatio =
    positiveInteger(source.width, 1) / positiveInteger(source.height, 1);
  const recipe =
    typeof model === "string"
      ? null
      : effectiveGenerationRecipe(model, pipeline);
  const defaultWidth =
    recipe?.defaults.width ??
    (typeof model === "string" ? null : model.default_width);
  const defaultHeight =
    recipe?.defaults.height ??
    (typeof model === "string" ? null : model.default_height);
  const defaultArea =
    defaultWidth && defaultHeight ? defaultWidth * defaultHeight : null;

  const ranked = [...presets].sort((left, right) => {
    const leftRatioDistance = Math.abs(
      Math.log(left.width / left.height / sourceRatio),
    );
    const rightRatioDistance = Math.abs(
      Math.log(right.width / right.height / sourceRatio),
    );
    if (Math.abs(leftRatioDistance - rightRatioDistance) > Number.EPSILON) {
      return leftRatioDistance - rightRatioDistance;
    }

    const leftArea = left.width * left.height;
    const rightArea = right.width * right.height;
    return defaultArea === null
      ? rightArea - leftArea
      : Math.abs(leftArea - defaultArea) - Math.abs(rightArea - defaultArea);
  });
  const selected = ranked[0];
  const sourceIsSquare =
    Math.abs(Math.log(sourceRatio)) <= SOURCE_PRESET_ASPECT_LOG_TOLERANCE;
  if (
    sourceIsSquare &&
    selected &&
    Math.abs(Math.log(selected.width / selected.height / sourceRatio)) >
      SOURCE_PRESET_ASPECT_LOG_TOLERANCE
  ) {
    // Wan checkpoints commonly advertise only landscape and portrait tiers.
    // A square source is not either shape: keep its aspect on the model's
    // aligned custom-canvas contract instead of silently cropping it to 16:9.
    return resolveSourceResolution(source, model, pipeline).output;
  }
  return selected
    ? { width: selected.width, height: selected.height }
    : resolveSourceResolution(source, model, pipeline).output;
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
  // Exact square caps such as sqrt(1024² / 1328²) can land one floating-point
  // ulp below 1024 and otherwise lose an entire 16 px alignment step. Nudge
  // only that machine-precision boundary before flooring; real fractions are
  // unchanged.
  const boundaryEpsilon = Number.EPSILON * Math.max(1, Math.abs(value)) * 8;
  return Math.floor((value + boundaryEpsilon) / alignment) * alignment;
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

/**
 * Cap a request canvas to the model's independent source-conditioning
 * contract. Qwen Image Edit can render at the ordinary output ceiling while
 * its VAE input stays at roughly 1 MP; clients use this immediately before
 * source-fit preprocessing so they never upload a larger intermediate only
 * for the server to shrink it again.
 */
export function resolveSourceConditioningTarget(
  canvas: SourceDimensions,
  model: ModelResolutionContract | string,
  pipeline?: string | null,
): SourceDimensions {
  return resolveSourceResolution(canvas, model, pipeline).output;
}

/** Human-ready source ceiling from the same model contract preprocessing uses. */
export function sourceConditioningLimitLabel(
  model: ModelResolutionContract | string,
  pipeline?: string | null,
): string {
  return formatMegapixels(sourceMaxPixelsForModel(model, pipeline));
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

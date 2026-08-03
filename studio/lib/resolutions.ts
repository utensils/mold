/** Shared, server-aligned resolution presets for web, desktop, and iPhone. */

export interface ResolutionPreset {
  label: string;
  aspect: string;
  width: number;
  height: number;
}

export interface RecommendedDimensions {
  width: number;
  height: number;
}

export interface ModelResolutionContract {
  family: string;
  default_width?: number;
  default_height?: number;
  max_pixels?: number | null;
  recommended_dimensions?: RecommendedDimensions[] | null;
  dimension_alignment?: number | null;
}

export const MAX_GENERATION_PIXELS = 1_800_000;
/**
 * LTX-2's own ceiling: upstream's shipped LTX_2_3_HQ_PARAMS renders 1920x1088.
 * The server advertises this per model on `/api/models.max_pixels`, which is
 * authoritative — these constants are only the fallback for a host that
 * predates the raised ceiling.
 */
export const LTX2_MAX_GENERATION_PIXELS = 1_920 * 1_088;
/**
 * Per-axis span, independent of the pixel budget. The checkpoints normalize
 * RoPE pixel positions by 2048, so a longer axis is out of distribution even
 * when the frame is small.
 */
export const LTX2_MAX_AXIS_PIXELS = 2_048;

export function maxPixelsForFamily(family: string | null | undefined): number {
  return family === "ltx2" ? LTX2_MAX_GENERATION_PIXELS : MAX_GENERATION_PIXELS;
}

export function maxAxisPixelsForFamily(
  family: string | null | undefined,
): number | null {
  return family === "ltx2" ? LTX2_MAX_AXIS_PIXELS : null;
}

export function dimensionAlignmentForFamily(
  family: string | null | undefined,
): number {
  return family === "ltx-video" || family === "ltx2" ? 32 : 16;
}

function gcd(a: number, b: number): number {
  let x = Math.max(1, Math.round(Math.abs(a)));
  let y = Math.max(1, Math.round(Math.abs(b)));
  while (y !== 0) [x, y] = [y, x % y];
  return x;
}

export function rawAspectRatioLabel(width: number, height: number): string {
  const divisor = gcd(width, height);
  return `${Math.round(width) / divisor}:${Math.round(height) / divisor}`;
}

const p = (
  width: number,
  height: number,
  aspect = rawAspectRatioLabel(width, height),
) => ({
  label: `${aspect} · ${width}×${height}`,
  aspect,
  width,
  height,
});

const SD15 = [p(512, 512), p(512, 768), p(768, 512), p(384, 512), p(512, 384)];
const SDXL = [
  p(1024, 1024),
  p(1152, 896),
  p(896, 1152),
  p(1216, 832),
  p(832, 1216),
  p(1344, 768),
  p(768, 1344),
  p(1536, 640),
  p(640, 1536),
];
const SD3 = SDXL.slice(0, 7);
const FLUX = [
  p(1024, 1024),
  p(1024, 768),
  p(768, 1024),
  p(1024, 576),
  p(576, 1024),
  p(768, 768),
];
const ZIMAGE = [p(1024, 1024), p(1024, 768), p(768, 1024)];
const QWEN_IMAGE = [
  p(1328, 1328),
  p(1024, 1024),
  p(1152, 896),
  p(896, 1152),
  p(1216, 832),
  p(832, 1216),
  p(1344, 768),
  p(768, 1344),
  p(1664, 928, "≈16:9"),
  p(928, 1664, "≈9:16"),
  p(768, 768),
  p(512, 512),
];
const VIDEO = [
  p(704, 480, "22:15"),
  p(768, 512),
  p(512, 512),
  p(1024, 576),
  p(1216, 704, "16:9"),
  p(576, 1024),
  p(768, 768),
  p(512, 768),
];
/** LTX-2 adds upstream's 1080p HQ pair and a 9:16 transpose of the default. */
const LTX2_VIDEO = [
  p(704, 480, "22:15"),
  p(768, 512),
  p(512, 512),
  p(1024, 576),
  p(1216, 704, "16:9"),
  p(704, 1216, "9:16"),
  p(576, 1024),
  p(768, 768),
  p(512, 768),
  p(1920, 1088, "16:9"),
  p(1088, 1920, "9:16"),
];

const BY_FAMILY: Record<string, ResolutionPreset[]> = {
  sd15: SD15,
  sdxl: SDXL,
  sd3: SD3,
  flux: FLUX,
  flux2: FLUX,
  zimage: ZIMAGE,
  "z-image": ZIMAGE,
  "qwen-image": QWEN_IMAGE,
  "qwen-image-edit": QWEN_IMAGE,
  wuerstchen: [p(1024, 1024)],
  "ltx-video": VIDEO,
  ltx2: LTX2_VIDEO,
};

export function presetsForFamily(family: string): ResolutionPreset[] {
  return BY_FAMILY[family] ?? FLUX;
}

export function presetsForModel(
  model: ModelResolutionContract | string,
): ResolutionPreset[] {
  if (typeof model === "string") return presetsForFamily(model);
  const advertised = model.recommended_dimensions ?? [];
  if (advertised.length === 0) return presetsForFamily(model.family);
  // The server's advertised values win; the family fallbacks only cover a
  // host that predates them.
  const maxPixels = model.max_pixels ?? maxPixelsForFamily(model.family);
  const alignment =
    model.dimension_alignment ?? dimensionAlignmentForFamily(model.family);
  return advertised
    .filter(
      ({ width, height }) =>
        width > 0 &&
        height > 0 &&
        width % alignment === 0 &&
        height % alignment === 0 &&
        width * height <= maxPixels,
    )
    .map(({ width, height }) => p(width, height));
}

export function matchPreset(
  width: number,
  height: number,
  model: ModelResolutionContract | string,
): ResolutionPreset | null {
  return (
    presetsForModel(model).find(
      (r) => r.width === width && r.height === height,
    ) ?? null
  );
}

export function aspectRatioLabel(
  width: number,
  height: number,
  model: ModelResolutionContract | string,
): string {
  return (
    matchPreset(width, height, model)?.aspect ??
    rawAspectRatioLabel(width, height)
  );
}

export function orientationLabel(
  width: number,
  height: number,
): "Landscape" | "Portrait" | "Square" {
  if (width === height) return "Square";
  return width > height ? "Landscape" : "Portrait";
}

export function presetsNearRatio(
  presets: readonly ResolutionPreset[],
  ratio: number,
  tolerance = 0.09,
): ResolutionPreset[] {
  if (!(ratio > 0) || presets.length === 0) return [];
  const ranked = presets
    .map((preset) => ({
      preset,
      diff: Math.abs(preset.width / preset.height - ratio) / ratio,
    }))
    .sort((a, b) => a.diff - b.diff);
  const best = ranked[0]?.diff ?? Number.POSITIVE_INFINITY;
  return ranked
    .filter(({ diff }) => diff <= Math.max(best + Number.EPSILON, tolerance))
    .map(({ preset }) => preset)
    .sort((a, b) => a.width * a.height - b.width * b.height);
}

export function closestResolutionPreset(
  presets: readonly ResolutionPreset[],
  width: number,
  height: number,
): ResolutionPreset | null {
  if (presets.length === 0) return null;
  const area = width * height;
  return (
    [...presets].sort(
      (a, b) =>
        Math.abs(a.width * a.height - area) -
        Math.abs(b.width * b.height - area),
    )[0] ?? null
  );
}

export function megapixelLabel(width: number, height: number): string {
  const value = ((width * height) / 1_000_000).toFixed(1).replace(/\.0$/, "");
  return `${value} MP`;
}

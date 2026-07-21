/**
 * Common, family-appropriate resolutions for the quick-select in the Size
 * block. Every entry is a multiple of 16 (the engines' latent stride — the
 * manual inputs snap to 16 for the same reason). The manual W/H inputs stay
 * for anything not listed.
 */
import { ASPECTS, dimsForMp, nearestMp } from "@ui/lib/resolution";

export interface ResolutionPreset {
  label: string;
  aspect: string;
  width: number;
  height: number;
}

const p = (aspect: string, width: number, height: number): ResolutionPreset => ({
  label: `${aspect} · ${width}×${height}`,
  aspect,
  width,
  height,
});

/** SD 1.5 was trained at 512; going far beyond it doubles subjects. */
const SD15: ResolutionPreset[] = [
  p("1:1", 512, 512),
  p("2:3", 512, 768),
  p("3:2", 768, 512),
  p("3:4", 640, 832),
  p("4:3", 832, 640),
];

/** SDXL's official bucket list (all ~1 MP). */
const SDXL: ResolutionPreset[] = [
  p("1:1", 1024, 1024),
  p("3:4", 896, 1152),
  p("4:3", 1152, 896),
  p("2:3", 832, 1216),
  p("3:2", 1216, 832),
  p("16:9", 1344, 768),
  p("9:16", 768, 1344),
];

/** Modern ~1 MP transformers (FLUX, SD3.5, Z-Image, Qwen-Image, Flux.2). */
const MODERN: ResolutionPreset[] = [
  p("1:1", 1024, 1024),
  p("3:4", 896, 1152),
  p("4:3", 1152, 896),
  p("2:3", 832, 1216),
  p("3:2", 1216, 832),
  p("16:9", 1344, 768),
  p("9:16", 768, 1344),
  p("21:9", 1536, 640),
];

/** Qwen's published buckets, kept in lockstep with mold-core validation. */
const QWEN_IMAGE: ResolutionPreset[] = [
  p("1:1", 1328, 1328),
  p("1:1", 1024, 1024),
  p("9:7", 1152, 896),
  p("7:9", 896, 1152),
  p("19:13", 1216, 832),
  p("13:19", 832, 1216),
  p("7:4", 1344, 768),
  p("4:7", 768, 1344),
  p("≈16:9", 1664, 928),
  p("≈9:16", 928, 1664),
  p("1:1", 768, 768),
  p("1:1", 512, 512),
];

/** Video works in smaller frames; these match the LTX sample configs. */
const VIDEO: ResolutionPreset[] = [
  p("22:15", 704, 480),
  p("3:2", 768, 512),
  p("16:9", 1024, 576),
  p("16:9", 1216, 704),
];

const BY_FAMILY: Record<string, ResolutionPreset[]> = {
  sd15: SD15,
  sdxl: SDXL,
  flux: MODERN,
  flux2: MODERN,
  sd3: MODERN,
  zimage: MODERN,
  "z-image": MODERN,
  "qwen-image": QWEN_IMAGE,
  "qwen-image-edit": QWEN_IMAGE,
  wuerstchen: SDXL,
  "ltx-video": VIDEO,
  ltx2: VIDEO,
};

export function presetsForFamily(family: string): ResolutionPreset[] {
  return BY_FAMILY[family] ?? MODERN;
}

/** The preset matching the current W/H exactly, or null (= "Custom"). */
export function matchPreset(
  width: number,
  height: number,
  family: string,
): ResolutionPreset | null {
  return presetsForFamily(family).find((r) => r.width === width && r.height === height) ?? null;
}

function gcd(a: number, b: number): number {
  let x = Math.max(1, Math.round(Math.abs(a)));
  let y = Math.max(1, Math.round(Math.abs(b)));
  while (y !== 0) [x, y] = [y, x % y];
  return x;
}

/** Human-readable ratio, preferring a known model bucket over raw arithmetic. */
export function aspectRatioLabel(width: number, height: number, family: string): string {
  const preset = matchPreset(width, height, family);
  if (preset) return preset.aspect;
  const divisor = gcd(width, height);
  return `${Math.round(width) / divisor}:${Math.round(height) / divisor}`;
}

export function orientationLabel(
  width: number,
  height: number,
): "Landscape" | "Portrait" | "Square" {
  if (width === height) return "Square";
  return width > height ? "Landscape" : "Portrait";
}

// ── Mold Studio inspector: megapixel + shape projection ─────────────────────
// The redesigned inspector drives Size with a five-swatch Shape picker (the
// shared @ui ASPECTS set) and an MP-budget Resolution selector rather than raw
// W/H. These helpers bridge that @ui vocabulary onto the desktop form, keeping
// the engines' /16 latent-stride constraint the manual inputs already enforce.

/** /16 latent-stride snap (floored at 64) shared with the manual W/H inputs. */
function snap16(value: number): number {
  return Math.max(64, Math.round(value / 16) * 16);
}

/**
 * The canonical shape id (from @ui `ASPECTS`) whose ratio matches width/height
 * within `tolerance` (relative), or null for a size the five swatches can't
 * represent (e.g. 2:3, 3:2, 21:9). Lets the Shape picker reflect the form's
 * live dimensions without forcing every bucket onto one of the five shapes.
 */
export function aspectIdFor(width: number, height: number, tolerance = 0.075): string | null {
  if (!(width > 0) || !(height > 0)) return null;
  const ratio = width / height;
  let best: { id: string; diff: number } | null = null;
  for (const aspect of ASPECTS) {
    const diff = Math.abs(ratio - aspect.ratio) / aspect.ratio;
    if (!best || diff < best.diff) best = { id: aspect.id, diff };
  }
  return best && best.diff <= tolerance ? best.id : null;
}

/**
 * Reproject the form's dimensions onto a new megapixel budget at the CURRENT
 * aspect ratio. Uses @ui `dimsForMp` for the megapixel math, then snaps to the
 * /16 grid so the result is a runnable size (a no-op on `dimsForMp`'s /64
 * output, but the constraint the desktop engines actually require).
 */
export function applyMp(form: { width: number; height: number }, mp: number): void {
  if (!(form.width > 0) || !(form.height > 0)) return;
  const { width, height } = dimsForMp(mp, form.width / form.height);
  form.width = snap16(width);
  form.height = snap16(height);
}

/**
 * Switch the form to a canonical shape (a swatch id from @ui `ASPECTS`) while
 * keeping the closest megapixel budget, snapped to /16. Unknown ids are a
 * no-op, so a custom size the picker can't select never gets clobbered.
 */
export function applyAspectId(form: { width: number; height: number }, id: string): void {
  const aspect = ASPECTS.find((option) => option.id === id);
  if (!aspect) return;
  const { width, height } = dimsForMp(nearestMp(form.width, form.height), aspect.ratio);
  form.width = snap16(width);
  form.height = snap16(height);
}

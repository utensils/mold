/**
 * Common, family-appropriate resolutions for the quick-select in the Size
 * block. Every entry is a multiple of 16 (the engines' latent stride — the
 * manual inputs snap to 16 for the same reason). The manual W/H inputs stay
 * for anything not listed.
 */

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
  "qwen-image": MODERN,
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

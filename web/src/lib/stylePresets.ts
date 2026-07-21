/*
 * Style presets (Mold Studio Create — style row). Each preset appends a fixed
 * bundle of descriptive "extras" to the prompt at request time; the textarea
 * content is never rewritten. `STYLE_ANGLES` cycle through variation angles
 * when Expand fans a batch out into editable variations (spec §03/§06).
 *
 * The strings are the prototype's exact `STYLE_EXTRAS` / `STYLE_ANGLES`
 * (docs/design/mold-studio-proposed-ui.html) — keep them verbatim so a saved
 * `stylePreset` reproduces the same appended text across surfaces.
 */

export interface StylePreset {
  /** Stable id persisted in the form (`stylePreset`). */
  id: string;
  /** Chip label. */
  name: string;
  /** Phrase appended to the outgoing prompt when this preset is active. */
  extras: string;
}

/** The eight presets, in chip order. */
export const STYLE_PRESETS: readonly StylePreset[] = [
  {
    id: "photoreal",
    name: "Photoreal",
    extras: "sharp focus, natural light, 50mm, fine texture",
  },
  {
    id: "cinematic",
    name: "Cinematic",
    extras: "cinematic lighting, anamorphic, dramatic mood, subtle film grain",
  },
  {
    id: "portrait",
    name: "Portrait",
    extras: "studio portrait, soft key light, 85mm, crisp detail",
  },
  {
    id: "illustration",
    name: "Illustration",
    extras: "bold linework, flat color, clean composition",
  },
  {
    id: "anime",
    name: "Anime",
    extras: "anime key art, cel shading, vibrant palette",
  },
  {
    id: "3d",
    name: "3D Render",
    extras: "octane render, subsurface scattering, studio HDRI lighting",
  },
  {
    id: "watercolor",
    name: "Watercolor",
    extras: "loose watercolor washes, textured paper, soft edges",
  },
  {
    id: "product",
    name: "Product",
    extras: "product photography, seamless backdrop, softbox lighting",
  },
] as const;

/** Variation angles Expand cycles through for a multi-print batch. */
export const STYLE_ANGLES: readonly string[] = [
  "at golden hour",
  "in moody overcast light",
  "in dramatic low-key light",
  "with soft diffused daylight",
  "backlit with warm rim light",
];

const BY_ID = new Map(STYLE_PRESETS.map((p) => [p.id, p]));

/** Resolve a preset by id. Returns `null` for `null` or an unknown id so a
 * stale persisted preset never crashes the request path. */
export function stylePresetById(id: string | null): StylePreset | null {
  if (!id) return null;
  return BY_ID.get(id) ?? null;
}

/** The extras string for an active preset id, or `null` when none applies. */
export function stylePresetExtras(id: string | null): string | null {
  return stylePresetById(id)?.extras ?? null;
}

/** The variation angle for a zero-based index, cycling the fixed list. */
export function angleForIndex(index: number): string {
  const count = STYLE_ANGLES.length;
  return STYLE_ANGLES[((index % count) + count) % count];
}

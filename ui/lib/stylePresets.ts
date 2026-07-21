/*
 * Unified style presets (spec §03 — Create style row), shared by web, desktop,
 * and iOS. This replaces the two divergent per-surface tables; the canonical
 * ids and fragments come from the prototype's STYLE_EXTRAS / STYLE_ANGLES
 * (docs/design/mold-studio-proposed-ui.html).
 *
 * Composition follows the bake-and-clear decision: the style is baked into the
 * outgoing request at submit time (A1111 semantics — a "{prompt}" placeholder
 * is substituted, otherwise the positive is comma-appended) and quick Batch-1
 * apply clears the chip afterwards; prepared Batch>1 keeps the chip as the
 * frozen-style indicator and its submit path never suffixes. The server-side
 * expansion directive gets the style as natural language via `styleHint`, not
 * as the literal suffix.
 */

const PROMPT_PLACEHOLDER = "{prompt}";

export interface StylePreset {
  /** Stable id persisted in form state (`stylePreset`). */
  id: string;
  /** Chip label. */
  label: string;
  /**
   * A1111-style positive template. Contains "{prompt}" where the look reads
   * better as a prefix; otherwise it is appended to the prompt with ", ".
   */
  positive: string;
  /**
   * Curated negative fragments, merged into the user's negative only when the
   * model supports a negative prompt. Omitted where a negative does not
   * genuinely help the look.
   */
  negative?: string;
}

/** The eight canonical presets, in chip order (prototype ids win). */
export const STYLE_PRESETS: readonly StylePreset[] = [
  {
    id: "photoreal",
    label: "Photoreal",
    positive:
      "photograph of {prompt}, sharp focus, natural light, 50mm, fine texture",
    negative: "drawing, painting, illustration, deformed, blurry",
  },
  {
    id: "cinematic",
    label: "Cinematic",
    positive:
      "cinematic film still of {prompt}, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    negative: "anime, cartoon, graphic, washed out",
  },
  {
    id: "portrait",
    label: "Portrait",
    positive: "studio portrait of {prompt}, soft key light, 85mm, crisp detail",
    negative: "deformed features, distorted face, blurry",
  },
  {
    id: "illustration",
    label: "Illustration",
    positive: "bold linework, flat color, clean composition",
    negative: "photo, photorealistic, 3d render",
  },
  {
    id: "anime",
    label: "Anime",
    positive: "anime key art of {prompt}, cel shading, vibrant palette",
    negative: "photo, photorealistic, muted colors",
  },
  {
    id: "3d",
    label: "3D Render",
    positive: "octane render, subsurface scattering, studio HDRI lighting",
    negative: "low poly, noisy, blurry",
  },
  {
    id: "watercolor",
    label: "Watercolor",
    positive:
      "watercolor painting of {prompt}, loose washes, textured paper, soft edges",
  },
  {
    id: "product",
    label: "Product",
    positive: "product photography, seamless backdrop, softbox lighting",
  },
] as const;

/**
 * Every id the pre-kit desktop and web tables ever persisted, mapped to its
 * canonical preset. Desktop's `neon` and `minimal` have no canonical twin, so
 * they re-home to the nearest look (dramatic lighting → cinematic; clean flat
 * composition → illustration) rather than silently dropping the user's pick.
 */
export const LEGACY_STYLE_IDS: Record<string, string> = {
  // Desktop table (desktop/src/lib/stylePresets.ts).
  photographic: "photoreal",
  "3d-render": "3d",
  neon: "cinematic",
  minimal: "illustration",
  // Shared / web table (web/src/lib/stylePresets.ts) — already canonical.
  photoreal: "photoreal",
  cinematic: "cinematic",
  portrait: "portrait",
  illustration: "illustration",
  anime: "anime",
  "3d": "3d",
  watercolor: "watercolor",
  product: "product",
};

/**
 * Canonical id for a stored preset id. Unknown or empty ids pass through
 * unchanged so callers keep their existing "no preset" handling.
 */
export function resolveStyleId(id: string): string {
  return LEGACY_STYLE_IDS[id] ?? id;
}

const BY_ID = new Map(STYLE_PRESETS.map((preset) => [preset.id, preset]));

/** Preset for a stored id (legacy aliases included), or null. */
export function stylePresetById(id: string): StylePreset | null {
  return BY_ID.get(resolveStyleId(id)) ?? null;
}

/** Chip label for a stored id — "None" when nothing (known) is picked. */
export function stylePresetLabel(id: string): string {
  return stylePresetById(id)?.label ?? "None";
}

export interface ComposeStyleOptions {
  /** Whether the target model accepts a negative prompt. */
  supportsNegativePrompt: boolean;
  /** The user's own negative prompt, if any. */
  negative?: string;
}

export interface ComposedStyle {
  prompt: string;
  negative?: string | undefined;
}

function splitFragments(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((fragment) => fragment.trim())
    .filter((fragment) => fragment.length > 0);
}

/**
 * Bake a preset into the outgoing request. Pure — callers pass a draft so the
 * live textarea is never mutated. An empty prompt, or an empty/unknown preset
 * id, returns the inputs unchanged. The preset negative is merged after the
 * user's fragments (exact duplicates dropped) and only when the model supports
 * a negative prompt.
 */
export function composeStyle(
  prompt: string,
  presetId: string,
  opts: ComposeStyleOptions,
): ComposedStyle {
  const preset = stylePresetById(presetId);
  const base = prompt.trim();
  if (!preset || !base) return { prompt, negative: opts.negative };

  const styled = preset.positive.includes(PROMPT_PLACEHOLDER)
    ? preset.positive.replaceAll(PROMPT_PLACEHOLDER, base)
    : `${base}, ${preset.positive}`;

  let negative = opts.negative;
  if (opts.supportsNegativePrompt && preset.negative) {
    const fragments = splitFragments(opts.negative);
    const seen = new Set(fragments);
    for (const fragment of splitFragments(preset.negative)) {
      if (!seen.has(fragment)) {
        seen.add(fragment);
        fragments.push(fragment);
      }
    }
    if (fragments.length > 0) negative = fragments.join(", ");
  }

  return { prompt: styled, negative };
}

/**
 * Natural-language description of a preset for the server expansion directive
 * ("{label} look — {fragments}"). The "{prompt}" placeholder is stripped and
 * the remaining fragments rejoined, so the style travels as an instruction —
 * never as the literal suffix. Null when no (known) preset is active.
 */
export function styleHint(presetId: string): string | null {
  const preset = stylePresetById(presetId);
  if (!preset) return null;
  const fragments = preset.positive
    .split(PROMPT_PLACEHOLDER)
    .map((segment) =>
      segment
        .replace(/^[\s,]+/, "")
        .replace(/[\s,]+$/, "")
        .replace(/\s+of$/, ""),
    )
    .filter((segment) => segment.length > 0);
  return `${preset.label} look — ${fragments.join(", ")}`;
}

/**
 * Variation angles a multi-print batch cycles through (prototype
 * STYLE_ANGLES, verbatim — today's web Expand fan-out behavior).
 */
export const STYLE_ANGLES = [
  "at golden hour",
  "in moody overcast light",
  "in dramatic low-key light",
  "with soft diffused daylight",
  "backlit with warm rim light",
] as const;

/** The variation angle for a zero-based index, cycling the fixed list. */
export function angleForIndex(index: number): string {
  const count = STYLE_ANGLES.length;
  return STYLE_ANGLES[((index % count) + count) % count] ?? STYLE_ANGLES[0];
}

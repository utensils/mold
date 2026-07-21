/**
 * Composer style presets (Mold Studio Create). Each preset appends a short
 * look-and-feel fragment to the outgoing prompt at submit time — the textarea
 * itself is never mutated, so the user's words stay theirs and the style is a
 * reversible, one-click modifier. `""` (no preset) is the default and composes
 * to the prompt unchanged.
 */
export interface StylePreset {
  id: string;
  label: string;
  /** Appended to the prompt (comma-joined) when this preset is active. */
  extra: string;
}

export const STYLE_PRESETS: readonly StylePreset[] = [
  {
    id: "photographic",
    label: "Photographic",
    extra: "photorealistic, sharp focus, natural light",
  },
  {
    id: "cinematic",
    label: "Cinematic",
    extra: "cinematic lighting, film grain, shallow depth of field",
  },
  { id: "anime", label: "Anime", extra: "anime style, cel shading, clean linework, vibrant" },
  { id: "illustration", label: "Illustration", extra: "digital illustration, painterly, detailed" },
  {
    id: "3d-render",
    label: "3D render",
    extra: "3d render, physically based, soft studio lighting",
  },
  {
    id: "watercolor",
    label: "Watercolor",
    extra: "watercolor painting, soft washes, textured paper",
  },
  { id: "neon", label: "Neon", extra: "neon-lit, high contrast, glowing accents" },
  { id: "minimal", label: "Minimal", extra: "minimalist, clean composition, negative space" },
] as const;

/** Human label for the header's active chip — "None" when nothing is picked. */
export function stylePresetLabel(id: string): string {
  return STYLE_PRESETS.find((preset) => preset.id === id)?.label ?? "None";
}

/**
 * Compose the outgoing prompt with the active style preset. Pure — the caller
 * passes a draft/clone so the live textarea is untouched. An unknown or empty
 * id, or an empty prompt, returns the prompt verbatim.
 */
export function composeStylePrompt(prompt: string, id: string): string {
  const preset = STYLE_PRESETS.find((candidate) => candidate.id === id);
  const base = prompt.trim();
  if (!preset || !base) return prompt;
  return `${base}, ${preset.extra}`;
}

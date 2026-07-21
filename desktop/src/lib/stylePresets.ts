/**
 * Desktop view of the shared Mold Studio style-preset kit (repo-root
 * `ui/lib/stylePresets.ts`). The kit owns the canonical eight presets, the
 * legacy-id aliases for everything this table used to persist (photographic,
 * 3d-render, neon, minimal), submit-time composition, and the
 * natural-language `styleHint` the server expansion directive uses. This
 * module re-exports that surface and keeps `composeStylePrompt` as the
 * prompt-only helper existing desktop call sites use — now templating-aware:
 * a "{prompt}" placeholder in the preset substitutes the prompt, otherwise
 * the look is comma-appended as before.
 */
import { composeStyle } from "@ui/lib/stylePresets";

export {
  LEGACY_STYLE_IDS,
  STYLE_ANGLES,
  STYLE_PRESETS,
  angleForIndex,
  composeStyle,
  mergeStyleNegative,
  resolveStyleId,
  styleHint,
  stylePresetById,
  stylePresetLabel,
} from "@ui/lib/stylePresets";
export type { ComposeStyleOptions, ComposedStyle, StylePreset } from "@ui/lib/stylePresets";

/**
 * Compose the outgoing prompt with the active style preset. Pure — the caller
 * passes a draft/clone so the live textarea is untouched. An unknown or empty
 * id, or an empty prompt, returns the prompt verbatim. Negative-prompt merging
 * deliberately does not happen here: submit sites use `composeStyle` directly,
 * where the family's negative-prompt capability gate lives.
 */
export function composeStylePrompt(prompt: string, id: string): string {
  return composeStyle(prompt, id, { supportsNegativePrompt: false }).prompt;
}

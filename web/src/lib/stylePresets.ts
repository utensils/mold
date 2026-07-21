/*
 * Web view of the shared Mold Studio style-preset kit (repo-root
 * `ui/lib/stylePresets.ts`). The kit owns the canonical eight presets, their
 * A1111-style positive templates, the curated negatives, the legacy-id
 * aliases, submit-time composition, and the natural-language `styleHint` the
 * server expansion directive uses — so one preset produces the same request on
 * web, desktop, and iOS.
 *
 * This module only adapts the surface web already consumes: a `name` alias for
 * the kit's `label`, and a null-tolerant `stylePresetById` (the form stores
 * `stylePreset: string | null`). The variation angles Expand cycles through
 * for a client-side batch fan-out come straight from the kit.
 */
import {
  STYLE_PRESETS as KIT_STYLE_PRESETS,
  resolveStyleId,
  type StylePreset as KitStylePreset,
} from "@ui/lib/stylePresets";

export {
  LEGACY_STYLE_IDS,
  STYLE_ANGLES,
  angleForIndex,
  composeStyle,
  mergeStyleNegative,
  resolveStyleId,
  styleHint,
} from "@ui/lib/stylePresets";
export type { ComposeStyleOptions, ComposedStyle } from "@ui/lib/stylePresets";

export interface StylePreset extends KitStylePreset {
  /** Alias of the kit's `label`, kept for existing web chip consumers. */
  name: string;
}

/** The eight presets, in chip order. */
export const STYLE_PRESETS: readonly StylePreset[] = KIT_STYLE_PRESETS.map(
  (preset) => ({ ...preset, name: preset.label }),
);

const BY_ID = new Map(STYLE_PRESETS.map((preset) => [preset.id, preset]));

/** Resolve a preset by id, following legacy aliases. Returns `null` for
 * `null` or an unknown id so a stale persisted preset never crashes the
 * request path. */
export function stylePresetById(id: string | null): StylePreset | null {
  if (!id) return null;
  return BY_ID.get(resolveStyleId(id)) ?? null;
}

/** Chip label for a stored id — "None" when nothing (known) is picked. */
export function stylePresetLabel(id: string | null): string {
  return stylePresetById(id)?.label ?? "None";
}

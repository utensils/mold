/**
 * The advertised reference-image contract: whether a checkpoint takes ordered
 * reference images (`GenerateRequest.edit_images`), how many, whether the
 * first one is the edit TARGET, and how references relate to `source_image`.
 *
 * `capabilities.reference_images` on the generation profile is the single
 * authority — `mold_core::generation_profile::reference_images_for_recipe`
 * answers it once for the server, admission, the CLI, the TUI and every GUI.
 * Absence of the block means an OLDER SERVER, never a refusal (the
 * `supports_strength` lesson): a client falls back to the pre-profile family
 * sniff in `legacyRecipeRules.ts`.
 *
 * The cross-surface expectations are pinned in
 * `tests/fixtures/flux2/reference-parity-v1.json`, read by both a mold-core
 * test and `flux2ReferenceParity.test.ts`.
 *
 * INTERIM: the two WIRE types below are hand-written to the field names of
 * `ReferenceImagesProfile` / `ReferenceSourceRelation` in
 * `crates/mold-core/src/generation_profile.rs`. When the Rust workstream
 * lands, `studio/lib/generated/generationProfileV1.ts` gains both and the two
 * declarations here become re-exports of the generated ones — this file stays
 * the import site so nothing else has to change.
 */

/** How `edit_images` relates to `source_image` on the same request. */
export type ReferenceSourceRelation = "replaces" | "exclusive" | "combines";

/** The wire block, kebab-case exactly as `mold-core` serializes it. */
export interface ReferenceImagesProfile {
  mode: "adjustable" | "fixed" | "hidden";
  required: boolean;
  /** Absent means no count bound (Qwen edit); `0` accompanies `hidden`. */
  max_count?: number | null;
  /** The first image is the edit target rather than a peer reference. */
  primary_is_target: boolean;
  source_relation: ReferenceSourceRelation;
  max_pixels_single?: number | null;
  max_pixels_multi?: number | null;
  reason?: string | null;
}

/** The client-side projection every surface reads. `null` where the recipe
 * (or the legacy rule standing in for an older host) offers no references. */
export interface ReferenceImagesCapabilities {
  /** Generate stays gated until at least one reference is attached. */
  required: boolean;
  /** Strip ceiling; `null` is unbounded (Qwen edit). */
  max: number | null;
  /** Index 0 is the edit target, rendered through the shared Target well. */
  primaryIsTarget: boolean;
  sourceRelation: ReferenceSourceRelation;
  maxPixelsSingle: number | null;
  maxPixelsMulti: number | null;
  /** The server's own sentence for a hidden block, for refusal copy. */
  reason: string | null;
}

/**
 * Project the advertised block onto the client shape. A `hidden` block is the
 * server SAYING NO — it answers `null` exactly like an absent one, and the
 * caller must not then fall back to a family sniff (only absence does that).
 */
export function referenceImagesFromProfile(
  profile: ReferenceImagesProfile,
): ReferenceImagesCapabilities | null {
  if (profile.mode === "hidden") return null;
  return {
    required: profile.required,
    max: profile.max_count ?? null,
    primaryIsTarget: profile.primary_is_target,
    sourceRelation: profile.source_relation,
    maxPixelsSingle: profile.max_pixels_single ?? null,
    maxPixelsMulti: profile.max_pixels_multi ?? null,
    reason: profile.reason ?? null,
  };
}

/**
 * Read `capabilities.reference_images` off an advertised recipe's capability
 * block. INTERIM: the cast disappears with the generated type; the ONE read
 * site is here so the swap is a one-line edit.
 */
export function advertisedReferenceImages(
  capabilities: object | null | undefined,
): ReferenceImagesProfile | null {
  const block = (
    capabilities as { reference_images?: ReferenceImagesProfile | null } | null
  )?.reference_images;
  return block ?? null;
}

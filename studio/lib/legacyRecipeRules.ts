/**
 * The family rules every client carried BEFORE the generation profile
 * advertised `capabilities.prompt` and `capabilities.supports_strength`.
 *
 * They are no longer an authority: a host that sends the profile answers
 * both questions itself (`generation_profile::prompt_requirement_for_family`
 * in `mold-core` is the ONE source, and validation reads the same function).
 * They survive here as the answer for a host that predates those fields —
 * the client-side legacy adapter fills its recipe from them so behaviour on
 * an old host is exactly what it was, and `baseGenerationCapabilities` falls
 * back to them when a recipe is silent.
 *
 * This module is a LEAF on purpose: `generationProfile.ts` imports it for the
 * legacy adapter, and everything richer (`generationCapabilities.ts`,
 * `promptRequirement.ts`) reaches `generationProfile.ts` at runtime, so any
 * shared helper that lived there would close an import cycle.
 */

import type { PromptRequirement } from "./generated/generationProfileV1";
import { isMinimaxH3Identity } from "./minimaxH3Identity";

/** Families whose engines could render from visual conditioning alone. */
const PROMPT_OPTIONAL_FAMILIES: ReadonlySet<string> = new Set([
  "ltx2",
  "ltx-2",
  "ltx-video",
]);

/**
 * The pre-profile prompt rule, spelled as the ADVERTISED mode: the answer for
 * a conditioned request, exactly as a host that emits the field would say it.
 * Never `ignored` — no family that predates the field lacked a text encoder.
 */
export function legacyPromptRequirementForFamily(
  family: string | null | undefined,
): PromptRequirement {
  return PROMPT_OPTIONAL_FAMILIES.has((family ?? "").trim().toLowerCase())
    ? "optional"
    : "required";
}

export function isWanFamily(family: string): boolean {
  return family.trim().toLowerCase() === "wan";
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

/**
 * The pre-profile mesh rule: Hunyuan3D is the only family that stores a
 * mesh. A host that advertises a recipe answers through its `mesh` block and
 * `output.formats`; this is the fallback for one that predates the profile.
 */
export function isMeshFamily(family: string | null | undefined): boolean {
  return (family ?? "").trim().toLowerCase() === "hunyuan3d";
}

export function isFlux2DevModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized.includes("flux2-dev") || normalized.includes("flux.2-dev");
}

/**
 * The pre-profile strength rule. Wan pins its conditioning frames exactly,
 * Qwen-Image-Edit and Flux.2 Dev condition through references rather than a
 * denoised source, H3 has no source-image path at all, and a 3-D family
 * reconstructs geometry rather than denoising pixels — none of them read
 * `strength`. Every other family denoises from the source image.
 */
export function legacySupportsStrength(family: string, model = ""): boolean {
  const normalized = family.trim().toLowerCase();
  return (
    !isMinimaxH3Identity(normalized, model) &&
    !isQwenImageEditFamily(normalized) &&
    !isFlux2DevModel(model) &&
    !isWanFamily(normalized) &&
    !isMeshFamily(normalized)
  );
}

/**
 * The pre-profile container rule for a 3-D family: binary glTF is the only
 * thing a mesh engine stores, exactly as the advertised recipe's
 * `output.formats` says. Without it the fallback offered `png`/`jpeg`/`webp`
 * for a print that has no raster at all.
 */
export const LEGACY_MESH_OUTPUT_FORMATS: readonly string[] = ["glb"];

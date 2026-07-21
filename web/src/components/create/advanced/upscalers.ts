/*
 * Post-generate upscalers — pure selection helpers shared by the Advanced
 * "Upscale after generate" section and its tests.
 *
 * The server publishes upscalers through the same `/api/models` listing as
 * generation models, tagged `family: "upscaler"` (Real-ESRGAN today). They are
 * never runnable as generators, so every other model surface filters them out
 * (`isStandaloneGenerationModel`); this module is the inverse view.
 *
 * `GenerateRequest.upscale_model` is the ONLY wire field — there is no separate
 * factor. The scale is a property of the chosen checkpoint (x4plus → 4×), so we
 * derive it for display instead of inventing a control the server would ignore.
 */
import type { ModelInfoExtended } from "../../../types";

const UPSCALER_FAMILIES = new Set(["upscaler", "real-esrgan", "realesrgan"]);
const UPSCALER_NAME = /real[-_ ]?esrgan|upscal(e|er|ing)/i;

/** True when a `/api/models` row is a post-generate upscaler. */
export function isUpscalerModel(model: ModelInfoExtended): boolean {
  if (UPSCALER_FAMILIES.has(model.family.toLowerCase())) return true;
  return UPSCALER_NAME.test(model.name);
}

/**
 * The upscalers in a model listing, on-disk ones first (they run without a
 * download) and name-sorted within each group.
 */
export function selectUpscalers(
  models: ModelInfoExtended[],
): ModelInfoExtended[] {
  return models
    .filter(isUpscalerModel)
    .slice()
    .sort((a, b) => {
      if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
}

/**
 * Model name to select when the switch is flipped on. Prefers an on-disk
 * upscaler (no download stall), then the smallest known scale factor — 2× is
 * the conservative enlargement, 4× quadruples the canvas and the VRAM the
 * post-pass needs. `""` when the host has none, which the section renders as
 * an explicit empty state instead of a dead switch.
 */
export function defaultUpscaler(models: ModelInfoExtended[]): string {
  const ranked = selectUpscalers(models)
    .slice()
    .sort((a, b) => {
      if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1;
      const fa = upscaleFactor(a) ?? Number.POSITIVE_INFINITY;
      const fb = upscaleFactor(b) ?? Number.POSITIVE_INFINITY;
      if (fa !== fb) return fa - fb;
      return a.name.localeCompare(b.name);
    });
  return ranked[0]?.name ?? "";
}

/**
 * Scale factor implied by the checkpoint (`…-x4plus:fp16` → 4). Falls back to
 * the manifest description (`"fast anime 4x upscaler"`) for names that do not
 * carry it, and returns null rather than guessing.
 */
export function upscaleFactor(model: ModelInfoExtended): number | null {
  const fromName = model.name.match(/[-_ ]x(\d+(?:\.\d+)?)/i)?.[1];
  if (fromName) return Number(fromName);
  const fromDescription = (model.description ?? "").match(
    /(\d+(?:\.\d+)?)\s*x\b/i,
  )?.[1];
  return fromDescription ? Number(fromDescription) : null;
}

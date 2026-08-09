import {
  baseGenerationCapabilities,
  isFlux2DevModel,
  isQwenImageEditFamily,
  type BaseGenerationCapabilities,
} from "@studio/lib/generationCapabilities";
import type { Scheduler } from "../types";

export type { SourceImageMode } from "@studio/lib/generationCapabilities";
export { isFlux2DevModel, isQwenImageEditFamily };

export interface GenerationCapabilities extends Omit<
  BaseGenerationCapabilities,
  "schedulerOptions"
> {
  schedulerOptions: Scheduler[];
}

/**
 * `advertisedSourceImage` is the selected `/api/models` row's additive
 * `source_image` field (#772). Pass it wherever that row is in scope — the
 * shared kit owns both the absent-field fallback to the family heuristic and
 * wan's first/last-frame gate, so no view should read the raw field or a
 * family set of its own.
 */
export function generationCapabilitiesForFamily(
  family: string,
  model = "",
  pipeline?: string | null,
  advertisedGuidance?: Parameters<typeof baseGenerationCapabilities>[3],
  advertisedSourceImage?: string | null,
): GenerationCapabilities {
  return baseGenerationCapabilities(
    family,
    model,
    pipeline,
    advertisedGuidance,
    advertisedSourceImage,
  );
}

export function schedulerOptionsForFamily(family: string): Scheduler[] {
  return generationCapabilitiesForFamily(family).schedulerOptions.slice();
}

export function isVideoFamily(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsVideo;
}

export function supportsNegativePrompt(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsNegativePrompt;
}

export function supportsScheduler(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsScheduler;
}

export function supportsLora(family: string): boolean {
  return generationCapabilitiesForFamily(family).supportsLora;
}

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

export function generationCapabilitiesForFamily(
  family: string,
  model = "",
  pipeline?: string | null,
  advertisedGuidance?: Parameters<typeof baseGenerationCapabilities>[3],
): GenerationCapabilities {
  return baseGenerationCapabilities(
    family,
    model,
    pipeline,
    advertisedGuidance,
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

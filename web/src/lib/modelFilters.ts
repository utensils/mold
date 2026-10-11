import { isGenerationModel } from "@studio/lib/generationModels";

import type { ModelInfoExtended } from "../types";

const VARIANT_RANKS: Record<string, number> = {
  bf16: 0,
  fp16: 1,
  q8: 2,
  fp8: 3,
  q6: 4,
  q5: 5,
  q4: 6,
};

/**
 * Whether a row is a style a person can pick.
 *
 * Shared with the other surfaces so support models never appear as styles.
 */
export function isStandaloneGenerationModel(model: ModelInfoExtended): boolean {
  return isGenerationModel(model);
}

export function modelQuantization(model: ModelInfoExtended): string {
  const lower = model.name.toLowerCase();
  const colonVariant = lower.match(/:([a-z0-9.-]+)$/)?.[1];
  if (colonVariant) return normalizeQuantization(colonVariant);
  const suffixVariant = lower.match(/[-_](bf16|fp16|fp8|q[0-9])\b/)?.[1];
  return suffixVariant ? normalizeQuantization(suffixVariant) : "unknown";
}

function normalizeQuantization(value: string): string {
  const match = value.match(/^(bf16|fp16|fp8|q[0-9])/);
  return match?.[1] ?? value;
}

export function variantRank(model: ModelInfoExtended): number {
  return VARIANT_RANKS[modelQuantization(model)] ?? 999;
}

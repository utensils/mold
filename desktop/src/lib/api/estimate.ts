import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { GenerateRequest, GenerationMemoryEstimate } from "./types";
import {
  classifyGenerationFit,
  displayGenerationMemory,
  generationEstimateLabel,
  GENERATION_ESTIMATE_TOOLTIP,
  type EstimateFit,
} from "@studio/lib/generationMemoryEstimate";

export type { EstimateFit };

/**
 * VRAM preflight for a pending generation. Takes the full request; only the
 * model + dimensions materially move the estimate, but the server accepts the
 * whole shape. Pass `target` to ask the host the batch will actually run on —
 * its VRAM is the one that matters, not the primary's.
 */
export function estimateGeneration(
  req: GenerateRequest,
  target?: ApiTarget | null,
): Promise<GenerationMemoryEstimate> {
  const init = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  };
  return target
    ? apiJsonTo<GenerationMemoryEstimate>(target, "/api/generate/estimate", init)
    : apiJson<GenerationMemoryEstimate>("/api/generate/estimate", init);
}

/**
 * Bucket an estimate into a stable hardware-fit verdict for the badge.
 * Moment-to-moment free VRAM is intentionally ignored: active/queued work owns
 * that memory temporarily and made this label alternate while a host worked.
 * Older servers without capacity fields report an estimate without claiming a
 * fit verdict.
 */
export function classifyFit(est: GenerationMemoryEstimate): EstimateFit {
  return classifyGenerationFit(est);
}

/** Stable values used by the badge copy; legacy servers retain estimate-only copy. */
export function displayEstimateMemory(est: GenerationMemoryEstimate): {
  peakBytes: number;
  capacityBytes: number | null;
} {
  return displayGenerationMemory(est);
}

/**
 * Badge copy. Every string leads with "VRAM" so the reader knows what is
 * being estimated — the old bare "Fits · est. 2.3 GB" read as a mystery.
 */
export function estimateLabel(
  fit: EstimateFit,
  peakBytes: number,
  availableBytes: number | null,
): string {
  return generationEstimateLabel(fit, peakBytes, availableBytes);
}

/** Plain-language tooltip for the badge. */
export const ESTIMATE_TOOLTIP = GENERATION_ESTIMATE_TOOLTIP;

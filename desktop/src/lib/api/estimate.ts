import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { GenerateRequest, GenerationMemoryEstimate } from "./types";
import type { EstimateFit } from "@studio/lib/generationMemoryEstimate";

export type { EstimateFit };

/**
 * VRAM preflight for a pending generation. Takes the full request; only the
 * model + dimensions materially move the estimate, but the server accepts the
 * whole shape. Pass `target` to ask the host the batch will actually run on —
 * its VRAM is the one that matters, not the primary's.
 *
 * The verdict, the numbers and the badge copy are `@studio/lib/
 * generationMemoryEstimate`, read by the shared badge itself; this module is
 * only how the app asks.
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

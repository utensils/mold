import { apiJson } from "./client";
import type { ChainLimits } from "./types";

/** Per-model chain caps (frames/clip, max stages, fade cap, audio support). */
export function fetchChainLimits(model: string): Promise<ChainLimits> {
  return apiJson<ChainLimits>(`/api/capabilities/chain-limits?model=${encodeURIComponent(model)}`);
}

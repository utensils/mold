import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { ChainLimits } from "./types";

/** Per-model chain caps (frames/clip, max stages, fade cap, audio support). */
export function fetchChainLimits(
  model: string,
  target: ApiTarget | null = null,
): Promise<ChainLimits> {
  const path = `/api/capabilities/chain-limits?model=${encodeURIComponent(model)}`;
  return target ? apiJsonTo<ChainLimits>(target, path) : apiJson<ChainLimits>(path);
}

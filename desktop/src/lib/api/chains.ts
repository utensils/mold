import { apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { ChainLimits } from "./types";

/** Per-model chain caps (frames/clip, max stages, fade cap, audio support). */
export function fetchChainLimits(
  model: string,
  target: ApiTarget | null = null,
  fps?: number | null,
): Promise<ChainLimits> {
  const fpsQuery = fps && fps > 0 ? `&fps=${encodeURIComponent(Math.floor(fps))}` : "";
  const path = `/api/capabilities/chain-limits?model=${encodeURIComponent(model)}${fpsQuery}`;
  return target ? apiJsonTo<ChainLimits>(target, path) : apiJson<ChainLimits>(path);
}

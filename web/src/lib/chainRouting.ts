/**
 * Web compatibility facade for the shared browser-safe routing authority.
 * Existing imports stay surface-local while web and desktop use one policy.
 */
export {
  DEFAULT_MOTION_TAIL,
  LTX2_DISTILLED_CLIP_CAP,
  LTX2_TEMPORAL_UPSCALE_MAX_FRAMES,
  MAX_CHAIN_STAGES,
  decideChainRouting,
  decideGenerateRequestRouting,
} from "@studio/lib/chainRouting";
export type { ChainRoutingDecision } from "@studio/lib/chainRouting";

/**
 * Web compatibility facade for the shared browser-safe routing authority.
 * Existing imports stay surface-local while web and desktop use one policy.
 */
export {
  AUTO_CHAIN_FIELD_LABELS,
  DEFAULT_MOTION_TAIL,
  LTX2_DEFAULT_CLIP_FRAMES,
  MAX_CHAIN_STAGES,
  autoChainFieldList,
  decideChainRouting,
  decideGenerateRequestRouting,
  unsupportedAutoChainFields,
} from "@studio/lib/chainRouting";
export type {
  AutoChainUnsupportedField,
  ChainRoutingDecision,
} from "@studio/lib/chainRouting";

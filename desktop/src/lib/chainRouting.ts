/**
 * Desktop-specific request projection layered on the shared browser-safe
 * chain-routing authority.
 */

import { decideGenerateRequestRouting, type ChainRoutingDecision } from "@studio/lib/chainRouting";
import type { AutoChainRequest, GenerateRequest } from "./api/types";

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
export type { AutoChainUnsupportedField, ChainRoutingDecision } from "@studio/lib/chainRouting";

export type AutoChainRoutingDecision = Extract<ChainRoutingDecision, { kind: "chain" }>;

/** Project a normal GenerateRequest onto the chain endpoint's auto-expand
 * wire contract. Only call this after `decideGenerateRequestRouting` returns a
 * chain decision; incompatible requests stay single-shot or are rejected. */
export function buildAutoChainRequest(
  req: GenerateRequest,
  decision: AutoChainRoutingDecision,
): AutoChainRequest {
  return {
    model: req.model,
    prompt: req.prompt,
    total_frames: req.frames!,
    clip_frames: decision.clipFrames,
    motion_tail_frames: decision.motionTail,
    width: req.width,
    height: req.height,
    ...(req.fps !== undefined ? { fps: req.fps } : {}),
    ...(req.seed !== undefined ? { seed: req.seed } : {}),
    steps: req.steps,
    guidance: req.guidance ?? 1,
    ...(req.strength !== undefined ? { strength: req.strength } : {}),
    ...(req.output_format !== undefined ? { output_format: req.output_format } : {}),
    ...(req.source_image !== undefined ? { source_image: req.source_image } : {}),
    ...(req.enable_audio !== undefined ? { enable_audio: req.enable_audio } : {}),
    ...(req.original_prompt !== undefined ? { original_prompt: req.original_prompt } : {}),
    ...(req.batch_id !== undefined ? { batch_id: req.batch_id } : {}),
    ...(req.batch_index !== undefined ? { batch_index: req.batch_index } : {}),
    ...(req.batch_count !== undefined ? { batch_count: req.batch_count } : {}),
  };
}

/** Match the one concrete server render the estimator should size. */
export function buildGenerationEstimateRequest(
  req: GenerateRequest,
  family: string | null | undefined,
): GenerateRequest {
  const decision = decideGenerateRequestRouting(req, family);
  const estimate = { ...req };
  // Optional properties must be absent, not merely `undefined`: aside from
  // satisfying the exact wire type, omission guarantees JSON serializers do
  // not grow an accidental media payload in a future transport change.
  delete estimate.source_image;
  delete estimate.source_image_name;
  delete estimate.mask_image;
  delete estimate.control_image;
  delete estimate.control_model;
  delete estimate.control_scale;
  delete estimate.edit_images;
  delete estimate.source_video;
  delete estimate.keyframes;
  delete estimate.audio_file;
  delete estimate.ic_lora_control;
  return {
    ...estimate,
    batch_size: 1,
    ...(decision.kind === "chain" ? { frames: decision.clipFrames } : {}),
  };
}

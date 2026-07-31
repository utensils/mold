/**
 * Desktop-specific request projection layered on the shared browser-safe
 * chain-routing authority.
 */

import { decideGenerateRequestRouting, type ChainRoutingDecision } from "@studio/lib/chainRouting";
import type { AutoChainRequest, GenerateRequest } from "./api/types";

export {
  DEFAULT_MOTION_TAIL,
  LTX2_DISTILLED_CLIP_CAP,
  LTX2_TEMPORAL_UPSCALE_MAX_FRAMES,
  MAX_CHAIN_STAGES,
  decideChainRouting,
  decideGenerateRequestRouting,
} from "@studio/lib/chainRouting";
export type { ChainRoutingDecision } from "@studio/lib/chainRouting";

export type AutoChainRoutingDecision = Extract<ChainRoutingDecision, { kind: "chain" }>;

/** GenerateRequest features the server's auto-expand chain form cannot carry. */
export type AutoChainUnsupportedField =
  | "negative_prompt"
  | "loras"
  | "audio_file"
  | "source_video"
  | "keyframes"
  | "pipeline"
  | "ic_lora_control"
  | "retake_range"
  | "spatial_upscale"
  | "temporal_upscale"
  | "guidance_overrides";

/** Report meaningful fields that would be lost when a normal generation is
 * routed through the auto-expand chain endpoint. Camera control rides the
 * LoRA wire fields, so it is intentionally reported as `loras` too. */
export function unsupportedAutoChainFields(req: GenerateRequest): AutoChainUnsupportedField[] {
  const unsupported: AutoChainUnsupportedField[] = [];
  if (req.negative_prompt?.trim()) unsupported.push("negative_prompt");
  if ((req.loras?.length ?? 0) > 0 || req.lora) unsupported.push("loras");
  if (req.audio_file) unsupported.push("audio_file");
  if (req.source_video) unsupported.push("source_video");
  if ((req.keyframes?.length ?? 0) > 0) unsupported.push("keyframes");
  if (req.pipeline) unsupported.push("pipeline");
  if (req.ic_lora_control) unsupported.push("ic_lora_control");
  if (req.retake_range) unsupported.push("retake_range");
  if (req.spatial_upscale) unsupported.push("spatial_upscale");
  if (req.temporal_upscale) unsupported.push("temporal_upscale");
  if (req.guidance_overrides && Object.keys(req.guidance_overrides).length > 0) {
    unsupported.push("guidance_overrides");
  }
  return unsupported;
}

/** Project a normal GenerateRequest onto the chain endpoint's auto-expand
 * wire contract. Call `unsupportedAutoChainFields` first so the UI can reject
 * an incompatible request instead of silently dropping it. */
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

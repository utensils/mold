/**
 * Client-side mirror of `crates/mold-cli/src/commands/chain.rs`'s
 * `decide_chain_routing` — a direct port of `web/src/lib/chainRouting.ts`
 * (keep the two in sync). The desktop uses it for the "will render as N
 * chained clips" cue under the Frames input and to block over-budget
 * requests for non-chainable models before they reach the server.
 *
 * Desktop-side normalization delta from the web sync source: the family is
 * lower-cased/trimmed and the "ltx-2" alias is canonicalised to "ltx2" before
 * matching — the same normalization the desktop capabilities layer applies
 * (`capabilities.ts` VIDEO_FAMILIES / ADVANCED_VIDEO_FAMILIES accept both). The
 * server/manifest emit the canonical "ltx2", but a form.family sourced from a
 * catalog entry or older server can arrive as "ltx-2" / "  LTX2  ". The
 * distilled-only chain gate applies to the alias form too. If the web mirror
 * ever grows the same normalization, drop this note and re-sync verbatim.
 *
 * The constants and branch structure match the Rust side exactly — if the
 * engine cap ever diverges from 97 we'd need to bump both (and ideally
 * expose it through a server capability). A regression test in chain.rs
 * asserts `LTX2_DISTILLED_CLIP_CAP % 8 == 1`.
 */

import type { AutoChainRequest, GenerateRequest } from "./api/types";

export const LTX2_DISTILLED_CLIP_CAP = 97;
export const MAX_CHAIN_STAGES = 16;
export const LTX2_TEMPORAL_UPSCALE_MAX_FRAMES = 257;
// 17 pixel frames → 3 LTX-2 latent frames of carryover under the VAE's 8×
// causal temporal compression (causal-first slot + two continuation slots).
// The prior 9-frame default only pinned one continuation slot (≈0.4 s at
// 24 fps), which was too little context to keep scene identity coherent past
// the first clip; bumping to 17 gives the denoiser ≈0.7 s of hard-pinned
// pixel context at the stitch boundary. Keep this in sync with
// `default_value_t` on --motion-tail in mold-cli.
export const DEFAULT_MOTION_TAIL = 17;

export type ChainRoutingDecision =
  | { kind: "single" }
  | {
      kind: "chain";
      clipFrames: number;
      motionTail: number;
      stageCount: number;
    }
  | { kind: "reject"; reason: string };

export type AutoChainRoutingDecision = Extract<ChainRoutingDecision, { kind: "chain" }>;

/** GenerateRequest features the server's auto-expand chain form cannot carry. */
export type AutoChainUnsupportedField =
  | "negative_prompt"
  | "loras"
  | "audio_file"
  | "source_video"
  | "keyframes"
  | "pipeline"
  | "retake_range"
  | "spatial_upscale"
  | "temporal_upscale";

/**
 * Report meaningful fields that would be lost when a normal generation is
 * routed through the auto-expand chain endpoint. Camera control rides the
 * LoRA wire fields, so it is intentionally reported as `loras` too.
 */
export function unsupportedAutoChainFields(req: GenerateRequest): AutoChainUnsupportedField[] {
  const unsupported: AutoChainUnsupportedField[] = [];
  if (req.negative_prompt?.trim()) unsupported.push("negative_prompt");
  if ((req.loras?.length ?? 0) > 0 || req.lora) unsupported.push("loras");
  if (req.audio_file) unsupported.push("audio_file");
  if (req.source_video) unsupported.push("source_video");
  if ((req.keyframes?.length ?? 0) > 0) unsupported.push("keyframes");
  if (req.pipeline) unsupported.push("pipeline");
  if (req.retake_range) unsupported.push("retake_range");
  if (req.spatial_upscale) unsupported.push("spatial_upscale");
  if (req.temporal_upscale) unsupported.push("temporal_upscale");
  return unsupported;
}

/**
 * Project a normal GenerateRequest onto the chain endpoint's auto-expand
 * wire contract. Call `unsupportedAutoChainFields` before this helper so the
 * UI can reject an incompatible request instead of silently dropping it.
 */
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

/** Families that support chain rendering. Mirrors the server-side
 * `chain_limits::family_cap` whitelist. `ltx2` has true latent-handoff chain
 * support; `ltx-video` uses an img2vid-less fallback (independent clips
 * stitched together) so subjects can drift between clips, but it lets users
 * generate videos longer than the per-clip cap. */
const CHAIN_CAPABLE_FAMILIES: ReadonlySet<string> = new Set(["ltx2", "ltx-video"]);

/** Families that have proper latent context handoff between clips. For
 * everything else the server forces motion_tail=0 (Smooth ≡ Cut at stitch
 * level) because there's no overlap region to trim. */
const FAMILIES_WITH_CONTEXT_HANDOFF: ReadonlySet<string> = new Set(["ltx2"]);

/** Live-catalog checkpoints keep an opaque stable id. Unlike built-in LTX-2
 * manifests they do not bundle the spatial upscaler, so the runtime selects
 * the chain-capable one-stage pipeline even though the id cannot say
 * "distilled". */
function isCatalogModel(model: string): boolean {
  return model.startsWith("cv:") || model.startsWith("hf:");
}

/** Normalize a raw family string to the canonical form the sets above use:
 * lower-case, trimmed, with the "ltx-2" alias folded onto "ltx2". Mirrors the
 * alias set the desktop capabilities layer accepts. */
function canonicalizeFamily(family: string | null | undefined): string {
  const fam = (family ?? "").trim().toLowerCase();
  return fam === "ltx-2" ? "ltx2" : fam;
}

/**
 * Request-aware routing for Generate surfaces. Temporal x2 is itself the
 * server's supported way to render an LTX-2 result above one clip: stage 1
 * denoises half as many frames, then the temporal upscaler expands to the
 * requested total. It must stay on the ordinary endpoint instead of being
 * mistaken for a multi-clip chain.
 */
export function decideGenerateRequestRouting(
  req: Pick<GenerateRequest, "frames" | "model" | "temporal_upscale">,
  family: string | null | undefined,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  const frames = req.frames;
  if (canonicalizeFamily(family) === "ltx2" && req.temporal_upscale === "x2" && frames) {
    return frames <= LTX2_TEMPORAL_UPSCALE_MAX_FRAMES
      ? { kind: "single" }
      : {
          kind: "reject",
          reason: `Temporal x2 supports at most ${LTX2_TEMPORAL_UPSCALE_MAX_FRAMES} frames. Reduce the frame count.`,
        };
  }
  return decideChainRouting(frames, family, req.model, motionTail);
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
  return {
    ...estimate,
    batch_size: 1,
    ...(decision.kind === "chain" ? { frames: decision.clipFrames } : {}),
  };
}

export function decideChainRouting(
  frames: number | null | undefined,
  family: string | null | undefined,
  model: string,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  if (!frames || frames <= 0) return { kind: "single" };

  const fam = canonicalizeFamily(family);
  const isChainCapable =
    CHAIN_CAPABLE_FAMILIES.has(fam) &&
    // LTX-2 chain rendering supports one-stage and distilled pipelines.
    // Built-in ids encode distilled explicitly; live-catalog ids are opaque
    // and default to one-stage because they do not bundle a spatial upscaler.
    (fam !== "ltx2" || model.includes("distilled") || isCatalogModel(model));

  if (!isChainCapable) {
    if (frames <= LTX2_DISTILLED_CLIP_CAP) return { kind: "single" };
    return {
      kind: "reject",
      reason: `Model '${model}' does not support chained video generation. Reduce frames to ${LTX2_DISTILLED_CLIP_CAP} or less.`,
    };
  }

  const clipFrames = LTX2_DISTILLED_CLIP_CAP;
  if (frames <= clipFrames) return { kind: "single" };

  // For families without context handoff (ltx-video), motion_tail is forced
  // to 0 server-side. Mirror that here so stage count math matches what the
  // server will actually run.
  const effectiveMotionTail = FAMILIES_WITH_CONTEXT_HANDOFF.has(fam) ? motionTail : 0;

  if (effectiveMotionTail >= clipFrames) {
    return {
      kind: "reject",
      reason: `motion tail (${effectiveMotionTail}) must be strictly less than clip frames (${clipFrames}).`,
    };
  }

  // Stage count mirrors `ChainRequest::normalise` in chain.rs:
  //   1 + ceil((total - clipFrames) / (clipFrames - motionTail))
  // — the first clip emits `clipFrames` frames, every continuation emits
  // `clipFrames - motionTail` new frames after the motion tail is trimmed
  // at stitch time.
  const effective = clipFrames - effectiveMotionTail;
  const remainder = frames - clipFrames;
  const stageCount = 1 + Math.ceil(remainder / effective);

  if (stageCount > MAX_CHAIN_STAGES) {
    const maxFrames = clipFrames + (MAX_CHAIN_STAGES - 1) * effective;
    return {
      kind: "reject",
      reason: `Chained video supports at most ${maxFrames} frames (${MAX_CHAIN_STAGES} clips) for this model. Reduce the frame count.`,
    };
  }

  return {
    kind: "chain",
    clipFrames,
    motionTail: effectiveMotionTail,
    stageCount,
  };
}

/**
 * Shared browser-safe mirror of the server's automatic chain-routing rules.
 * Keep the constants and branch structure aligned with
 * `crates/mold-cli/src/commands/chain.rs` and `ChainRequest::normalise`.
 */

export const LTX2_DISTILLED_CLIP_CAP = 97;
export const MAX_CHAIN_STAGES = 16;
export const LTX2_TEMPORAL_UPSCALE_MAX_FRAMES = 257;

// 17 pixel frames become three LTX-2 latent frames of carryover under the
// VAE's 8x causal temporal compression. Keep this in sync with the CLI's
// `--motion-tail` default.
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

type GenerateRoutingRequest = {
  frames?: number | null;
  model: string;
  temporal_upscale?: string | null;
};

/** Families accepted by the server's `chain_limits::family_cap` whitelist. */
const CHAIN_CAPABLE_FAMILIES: ReadonlySet<string> = new Set([
  "ltx2",
  "ltx-video",
]);

/** Only LTX-2 carries latent context between clips. */
const FAMILIES_WITH_CONTEXT_HANDOFF: ReadonlySet<string> = new Set(["ltx2"]);

/** Live-catalog checkpoints use opaque IDs and default to a chain-capable
 * one-stage pipeline because they do not bundle a spatial upscaler. */
function isCatalogModel(model: string): boolean {
  return model.startsWith("cv:") || model.startsWith("hf:");
}

function canonicalizeFamily(family: string | null | undefined): string {
  const normalized = (family ?? "").trim().toLowerCase();
  return normalized === "ltx-2" ? "ltx2" : normalized;
}

/** Keep LTX-2 temporal x2 on ordinary generation: it expands one latent
 * render and is not a multi-clip chain. */
export function decideGenerateRequestRouting(
  req: GenerateRoutingRequest,
  family: string | null | undefined,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  const frames = req.frames;
  if (
    canonicalizeFamily(family) === "ltx2" &&
    req.temporal_upscale === "x2" &&
    frames
  ) {
    return frames <= LTX2_TEMPORAL_UPSCALE_MAX_FRAMES
      ? { kind: "single" }
      : {
          kind: "reject",
          reason: `Temporal x2 supports at most ${LTX2_TEMPORAL_UPSCALE_MAX_FRAMES} frames. Reduce the frame count.`,
        };
  }
  return decideChainRouting(frames, family, req.model, motionTail);
}

export function decideChainRouting(
  frames: number | null | undefined,
  family: string | null | undefined,
  model: string,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  if (!frames || frames <= 0) return { kind: "single" };

  const normalizedFamily = canonicalizeFamily(family);
  const isChainCapable =
    CHAIN_CAPABLE_FAMILIES.has(normalizedFamily) &&
    // LTX-2 chain rendering supports one-stage and distilled pipelines.
    // Built-in IDs encode distilled explicitly; live-catalog IDs are opaque.
    (normalizedFamily !== "ltx2" ||
      model.includes("distilled") ||
      isCatalogModel(model));

  if (!isChainCapable) {
    if (frames <= LTX2_DISTILLED_CLIP_CAP) return { kind: "single" };
    return {
      kind: "reject",
      reason: `Model '${model}' does not support chained video generation. Reduce frames to ${LTX2_DISTILLED_CLIP_CAP} or less.`,
    };
  }

  const clipFrames = LTX2_DISTILLED_CLIP_CAP;
  if (frames <= clipFrames) return { kind: "single" };

  // Families without context handoff are forced to zero by the server.
  const effectiveMotionTail = FAMILIES_WITH_CONTEXT_HANDOFF.has(
    normalizedFamily,
  )
    ? motionTail
    : 0;

  if (effectiveMotionTail >= clipFrames) {
    return {
      kind: "reject",
      reason: `motion tail (${effectiveMotionTail}) must be strictly less than clip frames (${clipFrames}).`,
    };
  }

  // The first clip emits `clipFrames`; each continuation contributes the
  // clip minus its trimmed motion tail.
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

/**
 * Client-side mirror of `crates/mold-cli/src/commands/chain.rs`'s
 * `decide_chain_routing` so the SPA can auto-promote long video requests to
 * the chain endpoint without a round-trip. Keeping the decision logic pure
 * and out of the composable makes it unit-testable and lets us reuse it in
 * the Composer for the "will render as N chained clips" UX cue.
 *
 * The constants and branch structure match the Rust side exactly — if the
 * engine cap ever diverges from 97 we'd need to bump both (and ideally
 * expose it through a server capability). A regression test in chain.rs
 * asserts `LTX2_DISTILLED_CLIP_CAP % 8 == 1`.
 */

export const LTX2_DISTILLED_CLIP_CAP = 97;
// 9 pixel frames → 2 LTX-2 latent frames of carryover under the VAE's 8× causal
// temporal compression (causal-first slot + one continuation slot). Four frames
// — the prior default — only pinned the causal slot, which the decoder
// reconstructs as a single pixel frame, leaving the inter-clip stitch visibly
// jumpy. Keep this in sync with `default_value_t` on --motion-tail in mold-cli.
export const DEFAULT_MOTION_TAIL = 9;

export type ChainRoutingDecision =
  | { kind: "single" }
  | {
      kind: "chain";
      clipFrames: number;
      motionTail: number;
      stageCount: number;
    }
  | { kind: "reject"; reason: string };

export function decideChainRouting(
  frames: number | null | undefined,
  family: string | null | undefined,
  model: string,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  if (!frames || frames <= 0) return { kind: "single" };

  const isLtx2Distilled = family === "ltx2" && model.includes("distilled");

  if (!isLtx2Distilled) {
    if (frames <= LTX2_DISTILLED_CLIP_CAP) return { kind: "single" };
    return {
      kind: "reject",
      reason: `Model '${model}' does not support chained video generation (only LTX-2 distilled families do). Reduce frames to ${LTX2_DISTILLED_CLIP_CAP} or less.`,
    };
  }

  const clipFrames = LTX2_DISTILLED_CLIP_CAP;
  if (frames <= clipFrames) return { kind: "single" };

  if (motionTail >= clipFrames) {
    return {
      kind: "reject",
      reason: `motion tail (${motionTail}) must be strictly less than clip frames (${clipFrames}).`,
    };
  }

  // Stage count mirrors `ChainRequest::normalise` in chain.rs:
  //   1 + ceil((total - clipFrames) / (clipFrames - motionTail))
  // — the first clip emits `clipFrames` frames, every continuation emits
  // `clipFrames - motionTail` new frames after the motion tail is trimmed
  // at stitch time.
  const effective = clipFrames - motionTail;
  const remainder = frames - clipFrames;
  const stageCount = 1 + Math.ceil(remainder / effective);

  return { kind: "chain", clipFrames, motionTail, stageCount };
}

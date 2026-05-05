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

export function decideChainRouting(
  frames: number | null | undefined,
  family: string | null | undefined,
  model: string,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  if (!frames || frames <= 0) return { kind: "single" };

  const fam = family ?? "";
  const isChainCapable =
    CHAIN_CAPABLE_FAMILIES.has(fam) &&
    // ltx2 still requires a distilled checkpoint — only the distilled path
    // implements `as_chain_renderer` on the server. ltx-video accepts any
    // model in the family because the fallback wraps the standard t2v path.
    (fam !== "ltx2" || model.includes("distilled"));

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
  const effectiveMotionTail = FAMILIES_WITH_CONTEXT_HANDOFF.has(fam)
    ? motionTail
    : 0;

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

  return {
    kind: "chain",
    clipFrames,
    motionTail: effectiveMotionTail,
    stageCount,
  };
}

/**
 * Wan's chain routing (#783).
 *
 * Two things separate wan from the LTX families and both are load-bearing:
 *
 * 1. Wan has **no latent motion tail**. Its smooth handoff is last-frame image
 *    conditioning, which only an image-conditioned checkpoint accepts — so the
 *    carryover is a property of the checkpoint, read from the advertised
 *    `source_image` contract, never of the family. Offering "Continue motion"
 *    on a T2V-only checkpoint would promise continuity the model cannot
 *    produce.
 * 2. Wan's grid is `4k+1`, not `8k+1`, and its auto-chain clip length is a
 *    VRAM envelope rather than a ceiling.
 *
 * This mirrors `mold_inference::chain::wan_carryover`; the Rust side has the
 * matching test.
 */

import { describe, expect, it } from "vitest";

import {
  WAN_DEFAULT_CLIP_FRAMES,
  WAN_HANDOFF_DUPLICATED_FRAMES,
  WAN_SINGLE_EXPERT_CLIP_FRAMES,
  decideChainRouting,
  wanCarriesContext,
} from "./chainRouting";

describe("wan chain routing", () => {
  it("carries context only for an image-conditioned checkpoint", () => {
    expect(wanCarriesContext("required")).toBe(true);
    expect(wanCarriesContext("optional")).toBe(true);
    expect(wanCarriesContext("unsupported")).toBe(false);
    // Unclassified is unknown, never an assumed handoff — which is also what
    // an older server that does not advertise the field yields.
    expect(wanCarriesContext(null)).toBe(false);
    expect(wanCarriesContext(undefined)).toBe(false);
  });

  it("auto-chains past the clip envelope instead of rejecting", () => {
    // The pre-#783 behaviour: any wan request over the cap was rejected
    // outright because the family was not chain-capable.
    const decision = decideChainRouting(
      300,
      "wan",
      "wan22-ti2v-5b:fp16",
      undefined,
      24,
      "optional",
    );
    expect(decision.kind).toBe("chain");
    if (decision.kind !== "chain") return;
    expect(decision.clipFrames).toBe(WAN_SINGLE_EXPERT_CLIP_FRAMES);
    expect(decision.stageCount).toBeGreaterThan(1);
  });

  it("uses the two-expert envelope for A14B and the wider one for 5B", () => {
    const a14b = decideChainRouting(
      300,
      "wan",
      "wan22-t2v-a14b:q5",
      undefined,
      16,
      "unsupported",
    );
    expect(a14b.kind === "chain" && a14b.clipFrames).toBe(
      WAN_DEFAULT_CLIP_FRAMES,
    );

    const fiveB = decideChainRouting(
      300,
      "wan",
      "wan22-ti2v-5b:q8",
      undefined,
      24,
      "optional",
    );
    expect(fiveB.kind === "chain" && fiveB.clipFrames).toBe(
      WAN_SINGLE_EXPERT_CLIP_FRAMES,
    );
  });

  it("keeps every clip length on wan's 4k+1 grid", () => {
    for (const model of ["wan22-t2v-a14b:q5", "wan22-ti2v-5b:fp16"]) {
      const decision = decideChainRouting(
        300,
        "wan",
        model,
        undefined,
        16,
        "optional",
      );
      expect(decision.kind).toBe("chain");
      if (decision.kind !== "chain") continue;
      expect(
        (decision.clipFrames - 1) % 4,
        `${model} clip ${decision.clipFrames} is off the 4k+1 grid`,
      ).toBe(0);
    }
  });

  it("sizes the seam from the checkpoint, not from the caller's LTX tail", () => {
    // The seam is observable directly, so assert it rather than a stage-count
    // side effect: at 53-frame clips a tail of 0 and a tail of 1 need the same
    // number of stages, so a count comparison would pass whatever the tail was.
    const routed = (sourceImage: string | null) => {
      const decision = decideChainRouting(
        300,
        "wan",
        "wan22-t2v-a14b:q5",
        // A caller-supplied LTX-shaped tail must not reach wan in either
        // direction: wan has no latent motion tail to size.
        24,
        16,
        sourceImage,
      );
      if (decision.kind !== "chain") throw new Error("expected a chain");
      return decision.motionTail;
    };

    expect(routed("optional")).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    expect(routed("required")).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    // A text-to-video checkpoint has no conditioning channel, so nothing
    // crosses the seam and nothing is trimmed.
    expect(routed("unsupported")).toBe(0);
    expect(routed(null)).toBe(0);

    // The seeded frame is the only duplicate; 17 would discard sixteen good
    // frames at every boundary.
    expect(WAN_HANDOFF_DUPLICATED_FRAMES).toBe(1);
  });

  it("leaves the LTX families exactly as they were", () => {
    // The added parameter must not reach a family that does not read it: a
    // wan-shaped `source_image` hint must not rewrite LTX-2's real handoff.
    const withHint = decideChainRouting(
      500,
      "ltx2",
      "ltx-2-19b-distilled:fp8",
      24,
      24,
      "unsupported",
    );
    const withoutHint = decideChainRouting(
      500,
      "ltx2",
      "ltx-2-19b-distilled:fp8",
      24,
      24,
    );
    expect(withHint).toEqual(withoutHint);
  });
});

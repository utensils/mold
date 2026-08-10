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

  it("zeroes the motion tail on a text-to-video checkpoint", () => {
    // A T2V checkpoint concatenates independent clips: no frames are trimmed
    // at the seam, so N clips carry N * clipFrames of content. With a tail the
    // same request would need fewer stages, which is the arithmetic that would
    // silently change if the carryover were read from the family.
    const independent = decideChainRouting(
      300,
      "wan",
      "wan22-t2v-a14b:q5",
      24,
      16,
      "unsupported",
    );
    const carrying = decideChainRouting(
      300,
      "wan",
      "wan22-t2v-a14b:q5",
      24,
      16,
      "optional",
    );
    expect(independent.kind).toBe("chain");
    expect(carrying.kind).toBe("chain");
    if (independent.kind !== "chain" || carrying.kind !== "chain") return;
    expect(carrying.stageCount).toBeGreaterThan(independent.stageCount);
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

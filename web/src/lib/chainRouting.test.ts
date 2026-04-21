import { describe, expect, it } from "vitest";
import {
  DEFAULT_MOTION_TAIL,
  LTX2_DISTILLED_CLIP_CAP,
  decideChainRouting,
} from "./chainRouting";

describe("decideChainRouting", () => {
  it("returns single when frames is null/undefined/zero", () => {
    expect(
      decideChainRouting(null, "ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toEqual({ kind: "single" });
    expect(
      decideChainRouting(undefined, "ltx2", "ltx-2.3-22b-distilled:fp8"),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(0, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual({
      kind: "single",
    });
  });

  it("returns single for ltx2-distilled at-or-below the cap", () => {
    expect(
      decideChainRouting(
        LTX2_DISTILLED_CLIP_CAP,
        "ltx2",
        "ltx-2.3-22b-distilled:fp8",
      ),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(25, "ltx2", "ltx-2.3-22b-distilled:fp8")).toEqual(
      { kind: "single" },
    );
  });

  it("chains ltx2-distilled requests above the cap", () => {
    // 241 = 97 + 4*(97-4) - 9  → ceil(144/93) = 2 → 1+2 = 3 stages
    const d = decideChainRouting(241, "ltx2", "ltx-2.3-22b-distilled:fp8");
    expect(d).toEqual({
      kind: "chain",
      clipFrames: 97,
      motionTail: DEFAULT_MOTION_TAIL,
      stageCount: 3,
    });
  });

  it("chain stage count matches Rust normalise() expectations", () => {
    // Mirrors crates/mold-core/src/chain.rs test cases:
    //   (400, 97, 4, 5)  — 97 + 4*93 = 469 ≥ 400
    //   (200, 97, 4, 3)  — 97 + 2*93 = 283 ≥ 200
    //   (97,  97, 4, 1)  — single clip hits 97 exactly (handled as "single")
    expect(
      decideChainRouting(400, "ltx2", "ltx-2.3-22b-distilled:fp8", 4),
    ).toMatchObject({ kind: "chain", stageCount: 5 });
    expect(
      decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 4),
    ).toMatchObject({ kind: "chain", stageCount: 3 });
  });

  it("rejects non-ltx2-distilled models when frames exceed the single-clip budget", () => {
    const d = decideChainRouting(241, "ltx2", "ltx-2-19b:fp8");
    expect(d.kind).toBe("reject");
  });

  it("stays single for non-ltx2-distilled when frames are within budget", () => {
    expect(decideChainRouting(49, "ltx-video", "ltx-video-0.9.6:bf16")).toEqual(
      { kind: "single" },
    );
    expect(decideChainRouting(97, "ltx2", "ltx-2-19b:fp8")).toEqual({
      kind: "single",
    });
  });

  it("rejects when motion tail is >= clip frames", () => {
    const d = decideChainRouting(200, "ltx2", "ltx-2.3-22b-distilled:fp8", 97);
    expect(d.kind).toBe("reject");
  });

  it("returns single when family is missing", () => {
    expect(decideChainRouting(50, null, "anything")).toEqual({
      kind: "single",
    });
    expect(decideChainRouting(50, undefined, "anything")).toEqual({
      kind: "single",
    });
  });
});

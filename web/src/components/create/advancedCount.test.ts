import { describe, expect, it } from "vitest";
import { advancedActiveCount, type AdvancedCountParams } from "./advancedCount";

function params(
  overrides: Partial<AdvancedCountParams> = {},
): AdvancedCountParams {
  return {
    negativePrompt: "",
    hasSource: false,
    loraCount: 0,
    upscaleOn: false,
    scheduler: null,
    customSize: false,
    videoNonDefault: false,
    ...overrides,
  };
}

describe("advancedActiveCount", () => {
  it("is zero when nothing is active", () => {
    expect(advancedActiveCount(params())).toBe(0);
  });

  it("counts a non-empty negative prompt once, ignoring whitespace-only", () => {
    expect(advancedActiveCount(params({ negativePrompt: "blurry" }))).toBe(1);
    expect(advancedActiveCount(params({ negativePrompt: "   " }))).toBe(0);
  });

  it("counts each active LoRA", () => {
    expect(advancedActiveCount(params({ loraCount: 3 }))).toBe(3);
  });

  it("counts a non-default scheduler but not the default", () => {
    expect(advancedActiveCount(params({ scheduler: "ddim" }))).toBe(1);
    expect(advancedActiveCount(params({ scheduler: "default" }))).toBe(0);
    expect(advancedActiveCount(params({ scheduler: null }))).toBe(0);
  });

  it("counts source, upscale, custom size and video non-defaults", () => {
    expect(advancedActiveCount(params({ hasSource: true }))).toBe(1);
    expect(advancedActiveCount(params({ upscaleOn: true }))).toBe(1);
    expect(advancedActiveCount(params({ customSize: true }))).toBe(1);
    expect(advancedActiveCount(params({ videoNonDefault: true }))).toBe(1);
  });

  it("counts an active ControlNet block once", () => {
    expect(advancedActiveCount(params({ controlNet: true }))).toBe(1);
    expect(advancedActiveCount(params({ controlNet: false }))).toBe(0);
    // Absent (older callers) behaves like off.
    expect(advancedActiveCount(params())).toBe(0);
  });

  it("counts the LTX-2 video suite once", () => {
    expect(advancedActiveCount(params({ videoSuite: true }))).toBe(1);
    expect(advancedActiveCount(params({ videoSuite: false }))).toBe(0);
  });

  it("sums independent active fields", () => {
    expect(
      advancedActiveCount(
        params({
          negativePrompt: "low quality",
          hasSource: true,
          loraCount: 2,
          upscaleOn: true,
          scheduler: "uni-pc",
          customSize: true,
          videoNonDefault: true,
          controlNet: true,
          videoSuite: true,
          wanRecipe: 2,
        }),
      ),
    ).toBe(12);
  });
});

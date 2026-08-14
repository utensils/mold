import { describe, expect, it } from "vitest";
import { advancedActiveCount } from "./advancedCount";
import { newGenerateForm } from "./generateForm";

describe("advancedActiveCount", () => {
  it("is zero for a fresh form", () => {
    expect(advancedActiveCount({ ...newGenerateForm(), family: "sdxl" })).toBe(0);
  });

  it("counts each non-default advanced control the family exposes", () => {
    const form = { ...newGenerateForm(), family: "sdxl" };
    form.negativePrompt = "blurry";
    form.scheduler = "ddim";
    form.loras = [{ path: "a", name: "a", scale: 1, trainedWords: [] }];
    form.upscaleModel = "real-esrgan-x4plus";
    expect(advancedActiveCount(form)).toBe(4);
  });

  it("ignores controls the family does not support", () => {
    // FLUX has no negative-prompt or scheduler control, so a stale value there
    // is not "active".
    const form = { ...newGenerateForm(), family: "flux" };
    form.negativePrompt = "blurry";
    form.scheduler = "ddim";
    expect(advancedActiveCount(form)).toBe(0);
  });

  it("never counts a source image — it lives in the primary form, not Advanced", () => {
    const form = { ...newGenerateForm(), family: "ltx2" };
    form.sourceImage = "bytes";
    form.pipeline = "keyframe";
    form.cameraControl = "dolly-in";
    expect(advancedActiveCount(form)).toBe(2);
  });

  it("counts active LTX-2 guidance overrides as one advanced group", () => {
    const form = { ...newGenerateForm(), family: "ltx2" };
    form.guidanceOverrides = {
      ...form.guidanceOverrides,
      stgScale: 1.5,
      stgBlocks: "28, 29",
    };
    expect(advancedActiveCount(form)).toBe(1);

    form.family = "flux";
    expect(advancedActiveCount(form)).toBe(0);
  });

  it("counts each set wan recipe control, and none off-family", () => {
    const form = { ...newGenerateForm(), family: "wan", model: "wan22-t2v-a14b:q5" };
    form.scheduler = "dpm-pp";
    form.wanRecipe = { sampleShift: 12, distillStrengthHigh: 1.8, distillStrengthLow: null };
    // Solver + shift + high strength.
    expect(advancedActiveCount(form)).toBe(3);

    form.family = "flux";
    expect(advancedActiveCount(form)).toBe(0);
  });
});

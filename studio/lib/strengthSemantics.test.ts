import { describe, expect, it } from "vitest";

import {
  strengthSemantics,
  strengthSemanticsForModel,
} from "./strengthSemantics";

const LABEL = "How much to change it";

describe("strengthSemantics", () => {
  it("reads LTX-2 as source-preserving (higher keeps the source)", () => {
    for (const family of ["ltx2", "ltx-2", " LTX2 "]) {
      const semantics = strengthSemantics(family);
      expect(semantics.label).toBe(LABEL);
      expect(semantics.higherMeansSource).toBe(true);
      expect(semantics.hint).toMatch(/pins the opening frame/);
    }
  });

  it("keeps the SD denoise direction for every other family", () => {
    for (const family of [
      "stable-diffusion-1.5",
      "sdxl",
      "flux",
      "zimage",
      // ltx-video is deliberately unaudited — SD direction until proven.
      "ltx-video",
    ]) {
      const semantics = strengthSemantics(family);
      expect(semantics.label).toBe(LABEL);
      expect(semantics.higherMeansSource).toBe(false);
    }
  });

  it("says the lexicon phrase and never the engine word", () => {
    for (const family of ["ltx2", "sdxl"]) {
      const { label, hint } = strengthSemantics(family);
      expect(label).toBe(LABEL);
      for (const banned of ["denoise", "img2img", "Denoise", "Img2img"]) {
        expect(label).not.toContain(banned);
        expect(hint).not.toContain(banned);
      }
    }
  });

  it("resolves saved prints by inventory family first, then model-id markers", () => {
    expect(
      strengthSemanticsForModel("cv:12345", "ltx2").higherMeansSource,
    ).toBe(true);
    // Sequences record strength with no pipeline; the model id decides.
    expect(
      strengthSemanticsForModel("ltx-2-19b-distilled:fp8").higherMeansSource,
    ).toBe(true);
    expect(
      strengthSemanticsForModel("ltx2.3-22b-dev:fp8").higherMeansSource,
    ).toBe(true);
    expect(strengthSemanticsForModel("ltx-video-0.9.8").higherMeansSource).toBe(
      false,
    );
    expect(strengthSemanticsForModel("cv:99999").higherMeansSource).toBe(false);
    expect(strengthSemanticsForModel("sdxl-base:fp16").higherMeansSource).toBe(
      false,
    );
    expect(strengthSemanticsForModel("sdxl-base:fp16").label).toBe(LABEL);
  });
});

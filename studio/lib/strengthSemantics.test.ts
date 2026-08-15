import { describe, expect, it } from "vitest";

import {
  strengthSemantics,
  strengthSemanticsForModel,
} from "./strengthSemantics";

describe("strengthSemantics", () => {
  it("labels LTX-2 as source strength (higher preserves the source)", () => {
    for (const family of ["ltx2", "ltx-2", " LTX2 "]) {
      const semantics = strengthSemantics(family);
      expect(semantics.label).toBe("Source strength");
      expect(semantics.higherMeansSource).toBe(true);
      expect(semantics.hint).toMatch(/pins the opening frame/);
    }
  });

  it("keeps the SD denoise convention for every other family", () => {
    for (const family of [
      "stable-diffusion-1.5",
      "sdxl",
      "flux",
      "zimage",
      // ltx-video is deliberately unaudited — SD wording until proven.
      "ltx-video",
    ]) {
      const semantics = strengthSemantics(family);
      expect(semantics.label).toBe("Denoise strength");
      expect(semantics.higherMeansSource).toBe(false);
    }
  });

  it("resolves saved prints by inventory family first, then model-id markers", () => {
    expect(strengthSemanticsForModel("cv:12345", "ltx2").label).toBe(
      "Source strength",
    );
    // Sequences record strength with no pipeline; the model id decides.
    expect(strengthSemanticsForModel("ltx-2-19b-distilled:fp8").label).toBe(
      "Source strength",
    );
    expect(strengthSemanticsForModel("ltx2.3-22b-dev:fp8").label).toBe(
      "Source strength",
    );
    expect(strengthSemanticsForModel("ltx-video-0.9.8").label).toBe(
      "Denoise strength",
    );
    expect(strengthSemanticsForModel("cv:99999").label).toBe(
      "Denoise strength",
    );
    expect(strengthSemanticsForModel("sdxl-base:fp16").label).toBe(
      "Denoise strength",
    );
  });
});

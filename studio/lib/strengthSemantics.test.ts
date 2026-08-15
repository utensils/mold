import { describe, expect, it } from "vitest";

import { strengthSemantics } from "./strengthSemantics";

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
});

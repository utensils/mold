import { describe, expect, it } from "vitest";
import { estimateLabel, type EstimateFit } from "./estimate";

describe("estimateLabel", () => {
  const GB = 1_000_000_000;
  const cases: Array<[EstimateFit, number, number | null, string]> = [
    ["fits", 2.3 * GB, 64 * GB, "VRAM · fits — est. 2.3 GB of 64.0 GB"],
    ["fits", 2.3 * GB, null, "VRAM · est. 2.3 GB"],
    ["tight", 60 * GB, 64 * GB, "VRAM · tight — close other apps"],
    ["wont-fit", 90 * GB, 64 * GB, "VRAM · won't fit on this GPU"],
    ["unknown", 2.3 * GB, null, "VRAM · est. 2.3 GB"],
  ];
  it("always names VRAM so the number isn't a mystery", () => {
    for (const [fit, peak, available, expected] of cases) {
      expect(estimateLabel(fit, peak, available)).toBe(expected);
    }
  });
});

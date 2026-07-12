import { describe, expect, it } from "vitest";
import { classifyFit, estimateLabel, type EstimateFit } from "./estimate";

describe("classifyFit", () => {
  it("buckets fits / tight / wont-fit / unknown", () => {
    const est = (peak: number, available: number | null, fits?: boolean) => ({
      model: "flux-dev:q8",
      peak_memory_bytes: peak,
      activation_memory_bytes: 0,
      available_memory_bytes: available,
      fits_available_memory: fits ?? null,
      load_strategy: "full",
    });
    expect(classifyFit(est(1, 100))).toBe("fits");
    expect(classifyFit(est(93, 100))).toBe("tight");
    expect(classifyFit(est(101, 100))).toBe("wont-fit");
    expect(classifyFit(est(1, null))).toBe("unknown");
    expect(classifyFit(est(1, 100, false))).toBe("wont-fit");
  });
});

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

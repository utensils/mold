import { describe, expect, it } from "vitest";
import {
  classifyGenerationFit,
  displayGenerationMemory,
  generationEstimateLabel,
  type EstimateFit,
} from "./generationMemoryEstimate";

describe("classifyGenerationFit", () => {
  it("uses the server's stable capacity estimate", () => {
    const estimate = (peak: number, capacity: number | null) => ({
      peak_memory_bytes: 99,
      available_memory_bytes: 100,
      fits_available_memory: true,
      capacity_peak_memory_bytes: peak,
      device_capacity_bytes: capacity,
      fits_device_capacity: capacity == null ? null : peak <= capacity * 0.9,
    });

    expect(classifyGenerationFit(estimate(1, 100))).toBe("fits");
    expect(classifyGenerationFit(estimate(81, 100))).toBe("tight");
    expect(classifyGenerationFit(estimate(91, 100))).toBe("wont-fit");
    expect(classifyGenerationFit(estimate(1, null))).toBe("unknown");
  });

  it("does not oscillate when active work changes currently available VRAM", () => {
    const stable = {
      peak_memory_bytes: 16,
      capacity_peak_memory_bytes: 21,
      device_capacity_bytes: 25,
      fits_device_capacity: true,
    };

    expect(
      classifyGenerationFit({
        ...stable,
        available_memory_bytes: 24,
        fits_available_memory: true,
      }),
    ).toBe("tight");
    expect(
      classifyGenerationFit({
        ...stable,
        available_memory_bytes: 10,
        fits_available_memory: false,
      }),
    ).toBe("tight");
  });

  it("keeps legacy responses estimate-only instead of guessing from transient free VRAM", () => {
    const legacy = {
      peak_memory_bytes: 12,
      available_memory_bytes: 24,
      fits_available_memory: true,
    };
    expect(classifyGenerationFit(legacy)).toBe("unknown");
    expect(displayGenerationMemory(legacy)).toEqual({
      peakBytes: 12,
      capacityBytes: null,
    });
  });
});

describe("generationEstimateLabel", () => {
  const GB = 1_000_000_000;
  const cases: Array<[EstimateFit, number, number | null, string]> = [
    ["fits", 2.3 * GB, 64 * GB, "VRAM · fits — est. 2.3 GB of 64.0 GB"],
    ["fits", 2.3 * GB, null, "VRAM · est. 2.3 GB"],
    ["tight", 60 * GB, 64 * GB, "VRAM · tight — close other apps"],
    ["wont-fit", 90 * GB, 64 * GB, "VRAM · won't fit on this GPU"],
    ["unknown", 2.3 * GB, null, "VRAM · est. 2.3 GB"],
  ];

  it("always names VRAM so the number is unambiguous", () => {
    for (const [fit, peak, capacity, expected] of cases) {
      expect(generationEstimateLabel(fit, peak, capacity)).toBe(expected);
    }
  });
});

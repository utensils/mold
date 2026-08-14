import { describe, expect, it } from "vitest";
import { unifiedMemoryHost } from "./telemetryMemory";

describe("unifiedMemoryHost", () => {
  it("is true when every GPU reports the metal backend", () => {
    expect(
      unifiedMemoryHost([{ backend: "metal", name: "Apple Metal GPU" }]),
    ).toBe(true);
  });

  it("infers metal from an Apple GPU name when the backend field is absent (older servers)", () => {
    expect(unifiedMemoryHost([{ name: "Apple M4 Max" }])).toBe(true);
    expect(
      unifiedMemoryHost([{ backend: null, name: "Apple Metal GPU" }]),
    ).toBe(true);
  });

  it("is false for CUDA hosts, mixed fleets, CPU backends, and empty lists", () => {
    expect(unifiedMemoryHost([{ backend: "cuda", name: "NVIDIA L40S" }])).toBe(
      false,
    );
    expect(
      unifiedMemoryHost([
        { backend: "metal", name: "Apple Metal GPU" },
        { backend: "cuda", name: "NVIDIA L40S" },
      ]),
    ).toBe(false);
    expect(unifiedMemoryHost([{ backend: "cpu", name: "CPU" }])).toBe(false);
    expect(unifiedMemoryHost([{ name: "NVIDIA RTX 4090" }])).toBe(false);
    expect(unifiedMemoryHost([])).toBe(false);
  });
});

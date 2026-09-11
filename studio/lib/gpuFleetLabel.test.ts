import { describe, expect, it } from "vitest";
import { gpuFleetLabel, type GpuSnapshot } from "./gpuFleetLabel";

const gpu = (name: string): GpuSnapshot => ({ name });

describe("gpuFleetLabel", () => {
  it("counts identical cards instead of listing them", () => {
    expect(
      gpuFleetLabel([gpu("L40S"), gpu("L40S"), gpu("L40S"), gpu("L40S")]),
    ).toBe("4× L40S");
  });

  it("names mixed cards, deduped, in the order reported", () => {
    expect(gpuFleetLabel([gpu("RTX 4090"), gpu("B200"), gpu("RTX 4090")])).toBe(
      "RTX 4090 + B200",
    );
  });

  it("says the bare name for a single card", () => {
    expect(gpuFleetLabel([gpu("Apple M3 Max")])).toBe("Apple M3 Max");
  });

  it("says nothing before telemetry has arrived", () => {
    expect(gpuFleetLabel([])).toBe("");
  });
});

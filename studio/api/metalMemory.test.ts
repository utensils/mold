import { describe, expect, it } from "vitest";
import { parseMetalMemory, metalLimitLabel } from "./metalMemory";

const sample = {
  wired_limit: { mode: "automatic" },
  physical_bytes: 48 * 1024 ** 3,
  available_host_bytes: 32 * 1024 ** 3,
  recommended_bytes: 37 * 1024 ** 3,
  allocated_bytes: 4 * 1024 ** 3,
  effective_capacity_bytes: 37 * 1024 ** 3,
  allocation_headroom_bytes: 24 * 1024 ** 3,
  error: null,
};
describe("optional Metal memory telemetry", () => {
  it("accepts automatic, explicit and failed probes without inventing zero", () => {
    expect(metalLimitLabel(parseMetalMemory(sample)!)).toBe("automatic");
    expect(
      metalLimitLabel(
        parseMetalMemory({
          ...sample,
          wired_limit: { mode: "explicit", mib: 16384 },
        })!,
      ),
    ).toBe("16384 MiB");
    expect(
      parseMetalMemory({
        ...sample,
        effective_capacity_bytes: null,
        allocation_headroom_bytes: null,
        error: "probe failed",
      })?.effective_capacity_bytes,
    ).toBeNull();
  });
  it("ignores old, future and malformed extensions", () => {
    for (const value of [
      undefined,
      null,
      {},
      { ...sample, wired_limit: { mode: "future" } },
      { ...sample, allocated_bytes: -1 },
      { ...sample, allocated_bytes: Infinity },
      { ...sample, wired_limit: { mode: "explicit", mib: 0 } },
    ]) {
      expect(parseMetalMemory(value)).toBeUndefined();
    }
  });
});

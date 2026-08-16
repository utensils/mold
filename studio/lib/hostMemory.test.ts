import { describe, expect, it } from "vitest";
import { hostMemoryLevel, parseHostMemory } from "./hostMemory";

const snapshot = {
  total_bytes: 64_000_000_000,
  available_bytes: 40_000_000_000,
  headroom_bytes: 32_000_000_000,
  safety_floor_bytes: 4_000_000_000,
};

describe("parseHostMemory", () => {
  it("reads a complete additive snapshot", () => {
    expect(parseHostMemory(snapshot)).toEqual(snapshot);
  });

  it("reads a partial or malformed payload as absent", () => {
    expect(parseHostMemory(undefined)).toBeNull();
    expect(parseHostMemory(null)).toBeNull();
    expect(parseHostMemory([])).toBeNull();
    expect(parseHostMemory({ ...snapshot, headroom_bytes: "lots" })).toBeNull();
    expect(
      parseHostMemory({ ...snapshot, safety_floor_bytes: NaN }),
    ).toBeNull();
    const { total_bytes: _dropped, ...missing } = snapshot;
    expect(parseHostMemory(missing)).toBeNull();
  });
});

describe("hostMemoryLevel", () => {
  it("stays silent when the host does not report host memory", () => {
    expect(hostMemoryLevel(null)).toBeNull();
    expect(hostMemoryLevel(undefined)).toBeNull();
  });

  it("is ok while headroom clears the safety floor", () => {
    expect(hostMemoryLevel(snapshot)).toBe("ok");
  });

  it("warns within one safety floor of the wall", () => {
    expect(
      hostMemoryLevel({ ...snapshot, headroom_bytes: 3_000_000_000 }),
    ).toBe("warn");
  });

  it("is critical once nothing is spendable", () => {
    expect(hostMemoryLevel({ ...snapshot, headroom_bytes: 0 })).toBe(
      "critical",
    );
    expect(hostMemoryLevel({ ...snapshot, headroom_bytes: -1 })).toBe(
      "critical",
    );
  });

  it("treats a zero safety floor as no warn band rather than constant warning", () => {
    expect(
      hostMemoryLevel({
        ...snapshot,
        headroom_bytes: 1,
        safety_floor_bytes: 0,
      }),
    ).toBe("ok");
  });
});

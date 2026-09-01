import { describe, expect, it } from "vitest";
import { formatByteProgress, formatBytes } from "./formatBytes";

describe("formatBytes", () => {
  it("uses compact decimal units from bytes through terabytes", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(999)).toBe("999 B");
    expect(formatBytes(1_250)).toBe("1.3 KB");
    expect(formatBytes(1_250_000)).toBe("1.3 MB");
    expect(formatBytes(31_375_569_558)).toBe("31.4 GB");
    expect(formatBytes(2_400_000_000_000)).toBe("2.4 TB");
  });

  it("formats readable progress and rejects invalid counters", () => {
    expect(formatByteProgress(2_539_086_011, 31_375_569_558)).toBe(
      "2.5 GB / 31.4 GB",
    );
    expect(formatBytes(Number.NaN)).toBe("—");
    expect(formatBytes(-1)).toBe("—");
  });
});

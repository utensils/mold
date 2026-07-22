import { describe, expect, it } from "vitest";
import { formatGB } from "./format";

describe("formatGB", () => {
  it("uses decimal GB consistently for byte counts", () => {
    expect(formatGB(1_000_000_000)).toBe("1.0 GB");
    expect(formatGB(8_500_000_000)).toBe("8.5 GB");
    expect(formatGB(null)).toBe("—");
  });
});

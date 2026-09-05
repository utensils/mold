import { describe, expect, it } from "vitest";
import { formatGenerationTime } from "./generationTime";

/*
 * One spelling for "how long it took" on every surface: the caption's
 * `1024² · 4.0s`, the Recent row's `flux-dev:q8 · 4.0s`, the Lightbox's
 * "Took 4.0s". Tenths under a minute, minutes and whole seconds above it,
 * and nothing at all when the print does not know.
 */
describe("formatGenerationTime", () => {
  it("says nothing for an unknown or unmeasured time", () => {
    expect(formatGenerationTime(undefined)).toBeNull();
    expect(formatGenerationTime(null)).toBeNull();
    expect(formatGenerationTime(0)).toBeNull();
    expect(formatGenerationTime(-5)).toBeNull();
    expect(formatGenerationTime(Number.NaN)).toBeNull();
  });

  it("reads tenths of a second under a minute", () => {
    expect(formatGenerationTime(4_000)).toBe("4.0s");
    expect(formatGenerationTime(4_049)).toBe("4.0s");
    expect(formatGenerationTime(12_345)).toBe("12.3s");
    expect(formatGenerationTime(59_949)).toBe("59.9s");
  });

  it("reads minutes and whole seconds from a minute up", () => {
    expect(formatGenerationTime(60_000)).toBe("1m 00s");
    expect(formatGenerationTime(72_400)).toBe("1m 12s");
    expect(formatGenerationTime(3_599_999)).toBe("59m 59s");
    expect(formatGenerationTime(3_600_000)).toBe("60m 00s");
  });
});

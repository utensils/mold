import { describe, expect, it } from "vitest";
import { formatTypicalTime, typicalGenerationTimes } from "./styleSpeed";

/*
 * The Styles view's SPEED column has no server field behind it: a style's
 * typical time is read from the prints already made with it. The median of
 * the newest few timed prints, per style, so a cold first render cannot
 * stand for the style, and nothing at all for a style nobody has timed.
 */
describe("typicalGenerationTimes", () => {
  it("takes the median of the newest timed prints per style", () => {
    const times = typicalGenerationTimes([
      { model: "flux-dev:q8", generation_time_ms: 4_000 },
      { model: "flux-dev:q8", generation_time_ms: 40_000 }, // one cold render
      { model: "flux-dev:q8", generation_time_ms: 5_000 },
      { model: "ltx-video", generation_time_ms: 90_000 },
    ]);
    expect(times.get("flux-dev:q8")).toBe(5_000);
    expect(times.get("ltx-video")).toBe(90_000);
  });

  it("averages the two middle values of an even sample", () => {
    const times = typicalGenerationTimes([
      { model: "m", generation_time_ms: 2_000 },
      { model: "m", generation_time_ms: 4_000 },
    ]);
    expect(times.get("m")).toBe(3_000);
  });

  it("reads only the newest N per style, newest first as listed", () => {
    const prints = Array.from({ length: 15 }, (_, i) => ({
      model: "m",
      // Ten recent 2 s prints, then five old 60 s ones the sample must skip.
      generation_time_ms: i < 10 ? 2_000 : 60_000,
    }));
    expect(typicalGenerationTimes(prints, 10).get("m")).toBe(2_000);
  });

  it("ignores prints that do not know how long they took", () => {
    const times = typicalGenerationTimes([
      { model: "m", generation_time_ms: 0 },
      { model: "m", generation_time_ms: null },
      { model: "m" },
      { model: "", generation_time_ms: 4_000 },
    ]);
    expect(times.size).toBe(0);
  });
});

describe("formatTypicalTime", () => {
  it("rounds to whole seconds under a minute, never below one", () => {
    expect(formatTypicalTime(3_400)).toBe("~3s");
    expect(formatTypicalTime(19_800)).toBe("~20s");
    expect(formatTypicalTime(200)).toBe("~1s");
  });
  it("keeps minutes and seconds above a minute", () => {
    expect(formatTypicalTime(72_400)).toBe("~1m 12s");
  });
  it("says nothing for an unknown time", () => {
    expect(formatTypicalTime(null)).toBeNull();
    expect(formatTypicalTime(0)).toBeNull();
  });
});

import { describe, expect, it } from "vitest";

import { timeAgo } from "./relativeTime";

/*
 * These words are what MRU lists across the app say — the desktop's own lists
 * and the 3-D Studio's Recent panel both read them, which is why the function
 * lives in `studio/` rather than in a shell. Its only coverage used to be a
 * desktop test reaching through the re-export, so the module that owns the
 * words was never exercised by the studio suite.
 */
describe("timeAgo", () => {
  const now = Date.UTC(2026, 8, 9, 12, 0, 0);
  const ago = (ms: number) => timeAgo(now - ms, now);

  it("says just now for anything under a minute", () => {
    expect(ago(0)).toBe("just now");
    expect(ago(59_000)).toBe("just now");
  });

  it("counts minutes, then hours, then days", () => {
    expect(ago(60_000)).toBe("1m ago");
    expect(ago(59 * 60_000)).toBe("59m ago");
    expect(ago(60 * 60_000)).toBe("1h ago");
    expect(ago(23 * 3_600_000)).toBe("23h ago");
    expect(ago(24 * 3_600_000)).toBe("1d ago");
    expect(ago(29 * 86_400_000)).toBe("29d ago");
  });

  /* Past a month a relative count stops meaning anything; show the date. */
  it("falls back to a date beyond thirty days", () => {
    expect(ago(30 * 86_400_000)).not.toMatch(/ago$/);
    expect(ago(400 * 86_400_000)).not.toMatch(/ago$/);
  });

  /* A clock that disagrees must not produce "-3m ago". */
  it("never counts backwards for a timestamp in the future", () => {
    expect(timeAgo(now + 60_000, now)).toBe("just now");
  });
});

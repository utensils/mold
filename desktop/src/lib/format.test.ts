import { describe, expect, it } from "vitest";
import { formatGB, percent, timeAgo, vramLevel } from "./format";

describe("formatGB", () => {
  it("formats decimal gigabytes with one decimal", () => {
    expect(formatGB(38_200_000_000)).toBe("38.2 GB");
    expect(formatGB(0)).toBe("0.0 GB");
  });
});

describe("vramLevel", () => {
  it("is critical at or past 92%", () => {
    expect(vramLevel(91, 100)).toBe("ok");
    expect(vramLevel(92, 100)).toBe("critical");
    expect(vramLevel(100, 100)).toBe("critical");
  });

  it("treats an empty meter as ok", () => {
    expect(vramLevel(0, 0)).toBe("ok");
  });
});

describe("timeAgo", () => {
  it("buckets into minutes, hours, and days", () => {
    const now = 1_700_000_000_000;
    expect(timeAgo(now - 30_000, now)).toBe("just now");
    expect(timeAgo(now - 5 * 60_000, now)).toBe("5m ago");
    expect(timeAgo(now - 3 * 3_600_000, now)).toBe("3h ago");
    expect(timeAgo(now - 2 * 86_400_000, now)).toBe("2d ago");
  });

  it("never goes negative for future stamps", () => {
    const now = 1_700_000_000_000;
    expect(timeAgo(now + 60_000, now)).toBe("just now");
  });
});

describe("percent", () => {
  it("clamps to 0–100", () => {
    expect(percent(50, 100)).toBe(50);
    expect(percent(200, 100)).toBe(100);
    expect(percent(-5, 100)).toBe(0);
    expect(percent(1, 0)).toBe(0);
  });
});

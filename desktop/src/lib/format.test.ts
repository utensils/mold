import { describe, expect, it } from "vitest";
import {
  formatCount,
  formatEta,
  formatGB,
  formatUptime,
  percent,
  timeAgo,
  vramLevel,
} from "./format";

describe("formatUptime", () => {
  it("keeps sub-minute uptimes in seconds", () => {
    expect(formatUptime(0)).toBe("0s");
    expect(formatUptime(42)).toBe("42s");
    expect(formatUptime(-5)).toBe("0s");
  });

  it("uses whole minutes below an hour", () => {
    expect(formatUptime(60)).toBe("1m");
    expect(formatUptime(3_599)).toBe("59m");
  });

  it("shows hours with minutes, then days with hours", () => {
    expect(formatUptime(5_400)).toBe("1h 30m");
    expect(formatUptime(200_000)).toBe("2d 7h");
  });
});

describe("formatGB", () => {
  it("formats decimal gigabytes with one decimal", () => {
    expect(formatGB(38_200_000_000)).toBe("38.2 GB");
    expect(formatGB(0)).toBe("0.0 GB");
  });
});

describe("formatCount", () => {
  it("passes small counts through", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(999)).toBe("999");
  });

  it("compacts thousands and millions with one trimmed decimal", () => {
    expect(formatCount(12_300)).toBe("12.3k");
    expect(formatCount(1_000)).toBe("1k");
    expect(formatCount(4_200_000)).toBe("4.2M");
    expect(formatCount(950_000)).toBe("950k");
  });

  it("rolls the unit over when rounding reaches the next magnitude", () => {
    // 999_950 / 1000 rounds to "1000.0" — that must render as 1M, not 1000k.
    expect(formatCount(999_950)).toBe("1M");
    expect(formatCount(999_449)).toBe("999.4k");
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

describe("formatEta", () => {
  it("renders seconds, minutes, and hours compactly", () => {
    expect(formatEta(42)).toBe("42s");
    expect(formatEta(192)).toBe("3m 12s");
    expect(formatEta(3_845)).toBe("1h 4m");
  });

  it("renders a dash for unknown ETAs", () => {
    expect(formatEta(null)).toBe("—");
    expect(formatEta(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

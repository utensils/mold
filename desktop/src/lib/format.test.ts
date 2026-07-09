import { describe, expect, it } from "vitest";
import { formatGB, percent, vramLevel } from "./format";

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

describe("percent", () => {
  it("clamps to 0–100", () => {
    expect(percent(50, 100)).toBe(50);
    expect(percent(200, 100)).toBe(100);
    expect(percent(-5, 100)).toBe(0);
    expect(percent(1, 0)).toBe(0);
  });
});

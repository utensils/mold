import { describe, expect, it } from "vitest";
import { virtualGridWindow } from "./virtualGrid";

describe("virtualGridWindow", () => {
  it("keeps only viewport-adjacent rows mounted in a large library", () => {
    const window = virtualGridWindow({
      itemCount: 10_000,
      containerWidth: 1200,
      minimumItemWidth: 220,
      gap: 12,
      viewportStart: 50_000,
      viewportSize: 900,
      overscanRows: 2,
    });
    expect(window.columns).toBe(5);
    expect(window.startIndex).toBeGreaterThan(900);
    expect(window.endIndex - window.startIndex).toBeLessThan(50);
    expect(window.totalSize).toBeGreaterThan(400_000);
  });

  it("clamps the first and last window without dropping items", () => {
    const first = virtualGridWindow({
      itemCount: 11,
      containerWidth: 500,
      minimumItemWidth: 150,
      gap: 10,
      viewportStart: 0,
      viewportSize: 200,
    });
    expect(first.startIndex).toBe(0);
    const last = virtualGridWindow({
      itemCount: 11,
      containerWidth: 500,
      minimumItemWidth: 150,
      gap: 10,
      viewportStart: 10_000,
      viewportSize: 200,
    });
    expect(last.endIndex).toBe(11);
  });
});

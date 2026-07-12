import { describe, expect, it } from "vitest";
import { dockBadgeValue } from "./dockBadge";

describe("dockBadgeValue", () => {
  it("shows this app's pending job count", () => {
    expect(dockBadgeValue(3, true)).toBe(3);
    expect(dockBadgeValue(1, true)).toBe(1);
  });

  it("clears when idle or disabled", () => {
    expect(dockBadgeValue(0, true)).toBeNull();
    expect(dockBadgeValue(5, false)).toBeNull();
  });
});

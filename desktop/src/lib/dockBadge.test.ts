import { describe, expect, it } from "vitest";
import { dockBadgeValue } from "./dockBadge";

describe("dockBadgeValue", () => {
  it("shows how many prints landed while the app was away", () => {
    expect(dockBadgeValue(3, true)).toBe(3);
    expect(dockBadgeValue(1, true)).toBe(1);
  });

  it("clears when nothing landed or the badge is switched off", () => {
    expect(dockBadgeValue(0, true)).toBeNull();
    expect(dockBadgeValue(5, false)).toBeNull();
  });
});

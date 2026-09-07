import { describe, expect, it } from "vitest";
import { mobileQueueSection } from "./queueSections";
describe("mobile queue sections", () => {
  it("gives stale and blocked work attention without pretending it is still running", () => {
    expect(mobileQueueSection("running", true)).toBe("attention");
    expect(mobileQueueSection("queued", false, true)).toBe("attention");
    expect(mobileQueueSection("paused")).toBe("attention");
  });
  it("separates admission and waiting from active preparation or denoising", () => {
    expect(mobileQueueSection("accepted")).toBe("waiting");
    expect(mobileQueueSection("queued")).toBe("waiting");
    expect(mobileQueueSection("loading")).toBe("making");
    expect(mobileQueueSection("preparing")).toBe("making");
    expect(mobileQueueSection("running")).toBe("making");
  });
});

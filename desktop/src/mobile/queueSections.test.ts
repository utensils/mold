import { describe, expect, it } from "vitest";
import { mobileQueueSection } from "./queueSections";
describe("mobile queue sections", () => {
  it("gives stale and blocked work attention without pretending it is still running", () => {
    expect(mobileQueueSection("running", true)).toBe("attention");
    expect(mobileQueueSection("queued", false, true)).toBe("attention");
    expect(mobileQueueSection("paused")).toBe("attention");
  });
  it("keeps normal serialization in Waiting and flags actionable or unknown reasons", () => {
    for (const reason of [
      "no_schedulable_device",
      "no_idle_device",
      "warm_wait",
      "dependency_wait",
      "lower_priority_opening",
    ]) {
      expect(mobileQueueSection("queued", false, reason)).toBe("waiting");
    }
    expect(mobileQueueSection("queued", false, "license_required")).toBe("attention");
    expect(mobileQueueSection("queued", false, "future_reason")).toBe("attention");
  });
  it("separates admission and waiting from active preparation or denoising", () => {
    expect(mobileQueueSection("accepted")).toBe("waiting");
    expect(mobileQueueSection("queued")).toBe("waiting");
    expect(mobileQueueSection("loading")).toBe("making");
    expect(mobileQueueSection("preparing")).toBe("making");
    expect(mobileQueueSection("running")).toBe("making");
  });
});

import { describe, it, expect } from "vitest";
import { queueTransferEligible } from "./queueTransferEligibility";

describe("queue transfer eligibility", () => {
  it.each(["queued", "paused", "held"])(
    "offers %s only before rendering on modern servers",
    (state) => {
      expect(queueTransferEligible(state, true)).toBe(true);
    },
  );
  it.each([
    "running",
    "loading",
    "denoising",
    "finishing",
    "cancelling",
    "complete",
    "failed",
    "cancelled",
    "unknown",
  ])("never offers %s", (state) => {
    expect(queueTransferEligible(state, true)).toBe(false);
  });
  it("retains held-only behavior on older servers", () => {
    expect(queueTransferEligible("held", false)).toBe(true);
    expect(queueTransferEligible("queued", false)).toBe(false);
    expect(queueTransferEligible("paused", false)).toBe(false);
  });
});

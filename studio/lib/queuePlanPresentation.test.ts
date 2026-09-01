import { describe, expect, it } from "vitest";
import {
  queueCompletionLabel,
  queueLanePositionLabel,
  queuePlanUpdateLabel,
  queueScopeLabel,
} from "./queuePlanPresentation";

describe("human queue-plan presentation", () => {
  it("keeps loaded rows distinct from the durable queue total", () => {
    expect(queueScopeLabel(1, 7)).toBe("Showing 1 of 7 jobs");
    expect(queueScopeLabel(7, 7)).toBe("All 7 jobs loaded");
    expect(queueScopeLabel(1, null)).toBe("1 job loaded");
    expect(queueScopeLabel(0, 200, false)).toBe("Queue details unavailable");
  });

  it("never leaves an expired optimization countdown at zero seconds", () => {
    expect(queuePlanUpdateLabel(14_000, 10_000)).toBe("Updating order in 4s");
    expect(queuePlanUpdateLabel(10_000, 10_000)).toBe("Updating order…");
  });

  it("translates raw seconds and confidence into an understandable completion estimate", () => {
    expect(queueCompletionLabel(285_000, "low", 10_000)).toBe(
      "Done in about 5 minutes · estimate may change",
    );
    expect(queueCompletionLabel(70_000, "high", 10_000)).toBe(
      "Done in about 1 minute",
    );
    expect(queueCompletionLabel(null, "low", 10_000)).toBe(
      "Completion time is still being estimated",
    );
  });

  it("uses lane order instead of exposing internal work ids", () => {
    expect([0, 1, 2, 3, 10].map(queueLanePositionLabel)).toEqual([
      "Next",
      "After that",
      "3rd in this lane",
      "4th in this lane",
      "11th in this lane",
    ]);
  });
});

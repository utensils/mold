import { describe, expect, it } from "vitest";
import { countedFn, expectOpsUnder } from "./galleryPerfBudget";

describe("expectOpsUnder", () => {
  it("passes at or under the budget", () => {
    expect(() => expectOpsUnder("x", 10, 10)).not.toThrow();
    expect(() => expectOpsUnder("x", 0, 10)).not.toThrow();
  });
  it("names the label, observed count, budget and overage", () => {
    expect(() => expectOpsUnder("unionOrganization", 12_000, 2_000)).toThrow(
      /unionOrganization: 12000 operations exceeds the budget of 2000 \(\+10000\)/,
    );
  });
});

describe("countedFn", () => {
  it("delegates and counts", () => {
    const counted = countedFn((a: number, b: number) => a + b);
    expect(counted.fn(1, 2)).toBe(3);
    expect(counted.fn(2, 2)).toBe(4);
    expect(counted.count()).toBe(2);
    counted.reset();
    expect(counted.count()).toBe(0);
  });
});

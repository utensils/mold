import { describe, expect, it } from "vitest";
import { PromptCycler } from "./promptCycler";

const cyclerWith = (prompts: string[]): PromptCycler => {
  const c = new PromptCycler();
  c.setEntries(prompts);
  return c;
};

describe("PromptCycler", () => {
  it("walks newest to oldest on prev and clamps at the oldest", () => {
    const c = cyclerWith(["newest", "middle", "oldest"]);
    expect(c.prev("draft")).toBe("newest");
    expect(c.prev("newest")).toBe("middle");
    expect(c.prev("middle")).toBe("oldest");
    expect(c.prev("oldest")).toBeNull();
    expect(c.navigating).toBe(true);
  });

  it("walks back down and restores the draft past the newest", () => {
    const c = cyclerWith(["newest", "older"]);
    c.prev("my draft");
    c.prev("newest");
    expect(c.next()).toBe("newest");
    expect(c.next()).toBe("my draft");
    expect(c.navigating).toBe(false);
    // A further ↓ is inert.
    expect(c.next()).toBeNull();
  });

  it("returns null with no history", () => {
    const c = cyclerWith([]);
    expect(c.prev("draft")).toBeNull();
    expect(c.next()).toBeNull();
  });

  it("reset abandons navigation so the next prev re-captures the draft", () => {
    const c = cyclerWith(["a", "b"]);
    c.prev("first draft");
    c.reset();
    expect(c.navigating).toBe(false);
    expect(c.prev("edited draft")).toBe("a");
    c.prev("a");
    expect(c.next()).toBe("a");
    expect(c.next()).toBe("edited draft");
  });

  it("setEntries mid-navigation restarts navigation", () => {
    const c = cyclerWith(["a", "b"]);
    c.prev("draft");
    c.setEntries(["fresh", "a", "b"]);
    expect(c.navigating).toBe(false);
    expect(c.prev("draft")).toBe("fresh");
  });

  it("records a just-submitted remote prompt before its host history refetch completes", () => {
    const c = cyclerWith(["older"]);
    c.record("just submitted");
    expect(c.prev("")).toBe("just submitted");
    expect(c.prev("just submitted")).toBe("older");
  });
});

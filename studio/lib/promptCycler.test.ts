import { describe, expect, it } from "vitest";
import {
  PromptCycler,
  caretOnFirstLine,
  caretOnLastLine,
} from "./promptCycler";

describe("PromptCycler", () => {
  it("walks newest→oldest on prev and restores the draft on next", () => {
    const c = new PromptCycler();
    c.setEntries(["newest", "middle", "oldest"]);
    expect(c.prev("draft")).toBe("newest");
    expect(c.prev("draft")).toBe("middle");
    expect(c.prev("draft")).toBe("oldest");
    // At the oldest, prev stays put.
    expect(c.prev("draft")).toBeNull();
    // next walks back toward newer, then restores the pre-navigation draft.
    expect(c.next()).toBe("middle");
    expect(c.next()).toBe("newest");
    expect(c.next()).toBe("draft");
    // Past the draft, next is inert.
    expect(c.next()).toBeNull();
  });

  it("returns null from prev when there is no history", () => {
    const c = new PromptCycler();
    expect(c.prev("x")).toBeNull();
    expect(c.navigating).toBe(false);
  });

  it("reset abandons navigation (hand-edit)", () => {
    const c = new PromptCycler();
    c.setEntries(["a", "b"]);
    c.prev("draft");
    expect(c.navigating).toBe(true);
    c.reset();
    expect(c.navigating).toBe(false);
    // A fresh prev captures the new draft.
    expect(c.prev("edited")).toBe("a");
  });

  it("record pushes a prompt to the front and de-dupes", () => {
    const c = new PromptCycler();
    c.setEntries(["a", "b"]);
    c.record("b"); // moves b to front
    expect(c.prev("d")).toBe("b");
    expect(c.prev("d")).toBe("a");
  });

  it("setEntries mid-navigation resets the cursor", () => {
    const c = new PromptCycler();
    c.setEntries(["a", "b"]);
    c.prev("d");
    c.setEntries(["x", "y"]);
    expect(c.navigating).toBe(false);
    expect(c.prev("d")).toBe("x");
  });
});

describe("caret line helpers", () => {
  function ta(value: string, start: number, end = start): HTMLTextAreaElement {
    return {
      value,
      selectionStart: start,
      selectionEnd: end,
    } as HTMLTextAreaElement;
  }

  it("caretOnFirstLine is true only before any newline", () => {
    expect(caretOnFirstLine(ta("one\ntwo", 2))).toBe(true);
    expect(caretOnFirstLine(ta("one\ntwo", 5))).toBe(false);
  });

  it("caretOnLastLine is true only after the last newline", () => {
    expect(caretOnLastLine(ta("one\ntwo", 5))).toBe(true);
    expect(caretOnLastLine(ta("one\ntwo", 2))).toBe(false);
  });
});

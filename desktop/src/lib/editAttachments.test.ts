import { describe, expect, it } from "vitest";
import {
  attachmentRoleLabel,
  attachmentTitleLabel,
  moveAttachment,
  reorderAttachment,
} from "./editAttachments";

describe("reorderAttachment", () => {
  it("moves an item from one index to another, preserving the rest", () => {
    expect(reorderAttachment(["a", "b", "c", "d"], 0, 2)).toEqual(["b", "c", "a", "d"]);
    expect(reorderAttachment(["a", "b", "c", "d"], 3, 0)).toEqual(["d", "a", "b", "c"]);
    expect(reorderAttachment(["a", "b", "c"], 2, 1)).toEqual(["a", "c", "b"]);
  });

  it("returns the input unchanged for a no-op or out-of-range move", () => {
    const list = ["a", "b", "c"];
    expect(reorderAttachment(list, 1, 1)).toBe(list);
    expect(reorderAttachment(list, -1, 0)).toBe(list);
    expect(reorderAttachment(list, 0, 3)).toBe(list);
    expect(reorderAttachment(list, 3, 0)).toBe(list);
  });

  it("never mutates the input list", () => {
    const list = ["a", "b", "c"];
    reorderAttachment(list, 0, 2);
    expect(list).toEqual(["a", "b", "c"]);
  });
});

describe("moveAttachment", () => {
  it("swaps with the neighbor in the given direction", () => {
    expect(moveAttachment(["a", "b", "c"], 1, -1)).toEqual(["b", "a", "c"]);
    expect(moveAttachment(["a", "b", "c"], 1, 1)).toEqual(["a", "c", "b"]);
  });

  it("returns the input unchanged when the move would leave the list", () => {
    const list = ["a", "b", "c"];
    expect(moveAttachment(list, 0, -1)).toBe(list);
    expect(moveAttachment(list, 2, 1)).toBe(list);
  });
});

describe("role / title labels (web Composer parity)", () => {
  it("labels index 0 Target and everything after Reference", () => {
    expect(attachmentRoleLabel(0)).toBe("Target");
    expect(attachmentRoleLabel(1)).toBe("Reference");
    expect(attachmentRoleLabel(5)).toBe("Reference");
  });

  it("titles tiles Picture N (1-based)", () => {
    expect(attachmentTitleLabel(0)).toBe("Picture 1");
    expect(attachmentTitleLabel(2)).toBe("Picture 3");
  });
});

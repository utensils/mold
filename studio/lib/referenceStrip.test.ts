import { describe, expect, it } from "vitest";
import {
  moveReference,
  pastReorderThreshold,
  REFERENCE_CANVAS_NOTE,
  referenceDropSide,
  referenceIndexAtPoint,
  referenceMoveAnnouncement,
  referenceOrdinalBase,
  referenceStripItems,
  reorderReference,
  stripSetsCanvas,
} from "./referenceStrip";

describe("reorderReference", () => {
  it("moves an item from one index to another, preserving the rest", () => {
    expect(reorderReference(["a", "b", "c", "d"], 0, 2)).toEqual([
      "b",
      "c",
      "a",
      "d",
    ]);
    expect(reorderReference(["a", "b", "c", "d"], 3, 0)).toEqual([
      "d",
      "a",
      "b",
      "c",
    ]);
  });

  it("returns the input itself for a no-op or out-of-range move", () => {
    const list = ["a", "b", "c"];
    expect(reorderReference(list, 1, 1)).toBe(list);
    expect(reorderReference(list, -1, 0)).toBe(list);
    expect(reorderReference(list, 0, 3)).toBe(list);
  });

  it("never mutates the input", () => {
    const list = ["a", "b", "c"];
    reorderReference(list, 0, 2);
    expect(list).toEqual(["a", "b", "c"]);
  });
});

describe("moveReference", () => {
  it("steps one place earlier or later and clamps at the ends", () => {
    expect(moveReference(["a", "b", "c"], 1, -1)).toEqual(["b", "a", "c"]);
    expect(moveReference(["a", "b", "c"], 1, 1)).toEqual(["a", "c", "b"]);
    const list = ["a", "b"];
    expect(moveReference(list, 0, -1)).toBe(list);
    expect(moveReference(list, 1, 1)).toBe(list);
  });
});

describe("referenceStripItems", () => {
  it("numbers every reference 1..N the way the prompt addresses it", () => {
    const items = referenceStripItems({ count: 3 });
    expect(items.map((item) => item.label)).toEqual([
      "Image 1",
      "Image 2",
      "Image 3",
    ]);
    expect(items.map((item) => item.role)).toEqual([
      "Reference",
      "Reference",
      "Reference",
    ]);
    expect(items.every((item) => !item.setsCanvas)).toBe(true);
  });

  it("names the first picture the Target on a target-first recipe (Qwen edit)", () => {
    const items = referenceStripItems({ count: 3, firstIsTarget: true });
    expect(items.map((item) => item.role)).toEqual([
      "Target",
      "Reference",
      "Reference",
    ]);
    // The Target is `edit_images[0]`, so the expander calls it image 1 too.
    expect(items[0]!.label).toBe("Image 1");
  });

  it("marks exactly the LAST reference as the one that sets the canvas", () => {
    const items = referenceStripItems({ count: 3, setsCanvas: true });
    expect(items.map((item) => item.setsCanvas)).toEqual([false, false, true]);
    expect(referenceStripItems({ count: 0, setsCanvas: true })).toEqual([]);
  });

  it("offsets the ordinal past a source image that ships first", () => {
    const items = referenceStripItems({ count: 1, ordinalBase: 1 });
    expect(items[0]!.label).toBe("Image 2");
    expect(items[0]!.ordinal).toBe(2);
    expect(items[0]!.index).toBe(0);
  });
});

describe("referenceOrdinalBase", () => {
  it("counts the source first only where it ships beside the references", () => {
    // IP-Adapter: `source_image` precedes `edit_images` in the request, so the
    // expander's context names the reference image 2.
    expect(referenceOrdinalBase("single-and-references", true)).toBe(1);
    expect(referenceOrdinalBase("single-and-references", false)).toBe(0);
    // Klein never ships both: the strip is the whole request.
    expect(referenceOrdinalBase("single-or-references", true)).toBe(0);
    // Qwen edit's Target is edit_images[0]; FLUX.2 [dev]/Qwen 2.1 have no source.
    expect(referenceOrdinalBase("qwen-edit", true)).toBe(0);
    expect(referenceOrdinalBase("references", false)).toBe(0);
  });
});

describe("stripSetsCanvas", () => {
  it("is the last-reference rule on a references-only strip, nothing else", () => {
    expect(stripSetsCanvas("references", "last-reference")).toBe(true);
    expect(stripSetsCanvas("references", null)).toBe(false);
    expect(stripSetsCanvas("references", undefined)).toBe(false);
    // The canvas watchers apply the rule only in `references` mode.
    expect(stripSetsCanvas("qwen-edit", "last-reference")).toBe(false);
    expect(stripSetsCanvas("single-or-references", "last-reference")).toBe(
      false,
    );
  });

  it("explains the rule in one sentence", () => {
    expect(REFERENCE_CANVAS_NOTE).toMatch(/last image sets the canvas/i);
  });
});

describe("pointer reorder helpers", () => {
  const rect = (left: number, top: number, w = 100, h = 80) => ({
    left,
    top,
    right: left + w,
    bottom: top + h,
  });

  it("starts a drag only once the pointer has travelled past the threshold", () => {
    expect(pastReorderThreshold(0, 0)).toBe(false);
    expect(pastReorderThreshold(3, 3)).toBe(false);
    expect(pastReorderThreshold(6, 0)).toBe(true);
    expect(pastReorderThreshold(0, -6)).toBe(true);
  });

  it("hit-tests the tile under the pointer, across wrapped rows", () => {
    const rects = [rect(0, 0), rect(110, 0), rect(0, 90)];
    expect(referenceIndexAtPoint(rects, 50, 40)).toBe(0);
    expect(referenceIndexAtPoint(rects, 150, 10)).toBe(1);
    expect(referenceIndexAtPoint(rects, 20, 120)).toBe(2);
    // The gap between tiles and the space past the strip hit nothing.
    expect(referenceIndexAtPoint(rects, 105, 40)).toBeNull();
    expect(referenceIndexAtPoint(rects, 500, 500)).toBeNull();
  });

  it("draws the insertion mark on the side the picture will land", () => {
    expect(referenceDropSide(2, 0)).toBe("before");
    expect(referenceDropSide(0, 2)).toBe("after");
  });

  it("announces a move by the numbers the prompt uses", () => {
    expect(referenceMoveAnnouncement(2, 0, 0)).toBe(
      "Moved image 3 to position 1.",
    );
    expect(referenceMoveAnnouncement(0, 1, 1)).toBe(
      "Moved image 2 to position 3.",
    );
  });
});

import { describe, expect, it } from "vitest";
import { viewerAfterRemoval } from "./viewerRemoval";

describe("viewerAfterRemoval", () => {
  it("keeps a surviving current item, including a failed deletion", () => {
    expect(viewerAfterRemoval("b", ["a", "b", "c"], ["b", "c"])).toBe("b");
  });
  it("chooses the next survivor in the previous filtered order", () => {
    expect(viewerAfterRemoval("b", ["a", "b", "c", "d"], ["d", "a"])).toBe("d");
  });
  it("falls back to the previous survivor at the end", () => {
    expect(viewerAfterRemoval("c", ["a", "b", "c"], ["a", "b"])).toBe("b");
  });
  it("preserves a surviving renamed copy rather than jumping to its neighbor", () => {
    const current = {
      file: "original.png",
      copies: ["host-a/original.png", "host-b/mirror.png"],
    };
    const mirror = { file: "mirror.png", copies: ["host-b/mirror.png"] };
    const neighbor = { file: "next.png", copies: ["host-a/next.png"] };
    const same = (a: typeof current, b: typeof current) =>
      a.copies.some((id) => b.copies.includes(id));
    expect(
      viewerAfterRemoval(
        current,
        [current, neighbor],
        [mirror, neighbor],
        same,
      ),
    ).toBe(mirror);
  });
  it("closes an empty viewer and does not open a closed viewer", () => {
    expect(viewerAfterRemoval("a", ["a"], [])).toBeUndefined();
    expect(viewerAfterRemoval(undefined, ["a"], ["a"])).toBeUndefined();
  });
});

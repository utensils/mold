import { describe, expect, it } from "vitest";
import { invertPixels, maskFilenameFor, pointerToCanvas, pushCapped } from "./maskEditor";

describe("maskFilenameFor", () => {
  it("swaps the extension for -mask.png", () => {
    expect(maskFilenameFor("portrait.jpg")).toBe("portrait-mask.png");
    expect(maskFilenameFor("a.b.png")).toBe("a.b-mask.png");
  });

  it("falls back to source.png when no filename is given", () => {
    expect(maskFilenameFor(null)).toBe("source-mask.png");
    expect(maskFilenameFor(undefined)).toBe("source-mask.png");
    expect(maskFilenameFor("")).toBe("source-mask.png");
    expect(maskFilenameFor("   ")).toBe("source-mask.png");
  });

  it("handles a filename with no extension", () => {
    expect(maskFilenameFor("noext")).toBe("noext-mask.png");
  });
});

describe("invertPixels", () => {
  it("inverts RGB channels in place", () => {
    const data = new Uint8ClampedArray([10, 20, 30, 200]);
    invertPixels(data);
    expect(Array.from(data)).toEqual([245, 235, 225, 200]);
  });

  it("promotes a fully transparent pixel to opaque so the invert is visible", () => {
    const data = new Uint8ClampedArray([0, 0, 0, 0]);
    invertPixels(data);
    expect(Array.from(data)).toEqual([255, 255, 255, 255]);
  });

  it("leaves partially transparent alpha untouched", () => {
    const data = new Uint8ClampedArray([100, 100, 100, 128]);
    invertPixels(data);
    expect(data[3]).toBe(128);
  });
});

describe("pushCapped", () => {
  it("appends without exceeding the cap, dropping the oldest", () => {
    let stack: number[] = [];
    for (let i = 0; i < 25; i++) stack = pushCapped(stack, i, 20);
    expect(stack).toHaveLength(20);
    expect(stack[0]).toBe(5);
    expect(stack.at(-1)).toBe(24);
  });

  it("does not mutate the input array", () => {
    const original = [1, 2];
    const next = pushCapped(original, 3, 20);
    expect(original).toEqual([1, 2]);
    expect(next).toEqual([1, 2, 3]);
  });

  it("defaults the cap to 20", () => {
    let stack: number[] = [];
    for (let i = 0; i < 30; i++) stack = pushCapped(stack, i);
    expect(stack).toHaveLength(20);
  });
});

describe("pointerToCanvas", () => {
  it("maps client coords straight through when the rect matches the canvas", () => {
    const p = pointerToCanvas(
      64,
      80,
      { left: 0, top: 0, width: 512, height: 512 },
      { width: 512, height: 512 },
    );
    expect(p).toEqual({ x: 64, y: 80 });
  });

  it("scales when the canvas is rendered smaller than its intrinsic size", () => {
    // canvas is 1024 intrinsic but rendered at 512 → 2x scale, and offset by the rect origin.
    const p = pointerToCanvas(
      110,
      60,
      { left: 10, top: 10, width: 512, height: 512 },
      { width: 1024, height: 1024 },
    );
    expect(p).toEqual({ x: 200, y: 100 });
  });

  it("avoids divide-by-zero when the rect has no size", () => {
    const p = pointerToCanvas(
      5,
      5,
      { left: 0, top: 0, width: 0, height: 0 },
      { width: 512, height: 512 },
    );
    expect(p).toEqual({ x: 5, y: 5 });
  });
});

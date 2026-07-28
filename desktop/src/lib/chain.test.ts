import { describe, expect, it } from "vitest";
import { frames8n1Error, isLtx2FrameCount, snapFrames } from "./chain";

describe("frame-count validation", () => {
  it("accepts only 8n+1 counts", () => {
    for (const ok of [1, 9, 17, 25, 33, 97]) expect(isLtx2FrameCount(ok)).toBe(true);
    for (const bad of [0, 2, 8, 50, 96, 98, 100]) expect(isLtx2FrameCount(bad)).toBe(false);
  });

  it("returns the design-copy error for invalid counts", () => {
    expect(frames8n1Error(97)).toBeNull();
    expect(frames8n1Error(50)).toBe("Frames must be 8n+1 — try 97.");
  });

  it("snaps to the nearest valid count", () => {
    expect(snapFrames(50)).toBe(49);
    expect(snapFrames(52)).toBe(49);
    expect(snapFrames(54)).toBe(57);
    expect(snapFrames(0)).toBe(1);
    expect(snapFrames(97)).toBe(97);
  });
});

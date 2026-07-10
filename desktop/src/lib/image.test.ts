import { describe, expect, it } from "vitest";
import { isStillImageFile } from "./image";

describe("isStillImageFile", () => {
  it("accepts PNG and JPEG regardless of case", () => {
    for (const name of ["a.png", "a.PNG", "a.jpg", "a.JPG", "a.jpeg", "photo.JPEG"]) {
      expect(isStillImageFile(name)).toBe(true);
    }
  });

  it("rejects the animated / video / lossy formats the engine won't accept", () => {
    for (const name of ["clip.mp4", "loop.gif", "frame.webp", "anim.apng", "notes.txt", "noext"]) {
      expect(isStillImageFile(name)).toBe(false);
    }
  });

  it("keys off the final extension and tolerates surrounding whitespace", () => {
    expect(isStillImageFile("a.mp4.png")).toBe(true);
    expect(isStillImageFile("a.png.mp4")).toBe(false);
    expect(isStillImageFile("  spaced.jpg  ")).toBe(true);
  });
});

import { describe, expect, it } from "vitest";
import { isStillImageFile, isStillImageGalleryItem } from "./image";

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

describe("isStillImageGalleryItem", () => {
  const metadata = {
    prompt: "",
    model: "m",
    seed: 1,
    steps: 4,
    guidance: 3,
    width: 8,
    height: 8,
  };

  it("requires both the filename and metadata to describe a still image", () => {
    expect(isStillImageGalleryItem({ filename: "still.png", format: "png", metadata })).toBe(true);
    expect(isStillImageGalleryItem({ filename: "mislabelled.png", format: "mp4", metadata })).toBe(
      false,
    );
    expect(
      isStillImageGalleryItem({
        filename: "legacy-video.png",
        format: null,
        metadata: { ...metadata, video_frames: 25 },
      }),
    ).toBe(false);
    expect(
      isStillImageGalleryItem({
        filename: "current-video.png",
        format: null,
        metadata: { ...metadata, frames: 97 },
      }),
    ).toBe(false);
  });
});

import { describe, expect, it } from "vitest";
import { base64ToDataUrl, isStillImageFile, isStillImageGalleryItem } from "./image";

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

describe("still-image predicates with an advertised container list", () => {
  const metadata = {
    prompt: "",
    model: "m",
    seed: 1,
    steps: 4,
    guidance: 3,
    width: 8,
    height: 8,
  };
  const qwen21 = ["png", "jpeg", "webp"] as const;

  it("accepts WebP only where the recipe advertises it", () => {
    expect(isStillImageFile("cutout.webp")).toBe(false);
    expect(isStillImageFile("cutout.webp", qwen21)).toBe(true);
    expect(isStillImageFile("clip.gif", qwen21)).toBe(false);
  });

  it("still refuses an animated WebP clip on a WebP-accepting recipe", () => {
    expect(
      isStillImageGalleryItem({ filename: "cutout.webp", format: "webp", metadata }, qwen21),
    ).toBe(true);
    expect(
      isStillImageGalleryItem(
        { filename: "clip.webp", format: "webp", metadata: { ...metadata, frames: 97 } },
        qwen21,
      ),
    ).toBe(false);
    expect(isStillImageGalleryItem({ filename: "cutout.webp", format: "webp", metadata })).toBe(
      false,
    );
  });
});

describe("base64ToDataUrl", () => {
  it("labels the payload with its own container unless told otherwise", () => {
    expect(base64ToDataUrl("UklGRjAHAABXRUJQ")).toBe("data:image/webp;base64,UklGRjAHAABXRUJQ");
    expect(base64ToDataUrl("/9j/4AAQ")).toBe("data:image/jpeg;base64,/9j/4AAQ");
    expect(base64ToDataUrl("iVBORw0K")).toBe("data:image/png;base64,iVBORw0K");
    expect(base64ToDataUrl("????")).toBe("data:image/png;base64,????");
    expect(base64ToDataUrl("AAAA", "image/gif")).toBe("data:image/gif;base64,AAAA");
  });
});

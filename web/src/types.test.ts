import { describe, expect, it } from "vitest";
import { inferFormatFromName, mediaKind } from "./types";

describe("inferFormatFromName", () => {
  it.each([
    ["cat.png", "png"],
    ["CAT.PNG", "png"],
    ["shot.jpg", "jpeg"],
    ["shot.jpeg", "jpeg"],
    ["loop.gif", "gif"],
    ["loop.apng", "apng"],
    ["loop.webp", "webp"],
    ["clip.mp4", "mp4"],
  ] as const)("maps %s → %s", (name, expected) => {
    expect(inferFormatFromName(name)).toBe(expected);
  });

  it("returns null for unrecognized extensions", () => {
    expect(inferFormatFromName("something.txt")).toBeNull();
    expect(inferFormatFromName("noext")).toBeNull();
    expect(inferFormatFromName("")).toBeNull();
  });
});

describe("mediaKind", () => {
  it("uses the explicit format when provided", () => {
    expect(mediaKind("mp4", "anything")).toBe("video");
    expect(mediaKind("gif", "anything")).toBe("animated");
    expect(mediaKind("png", "anything")).toBe("image");
  });

  it("falls back to the filename extension when format is null/undefined", () => {
    expect(mediaKind(null, "clip.mp4")).toBe("video");
    expect(mediaKind(undefined, "loop.webp")).toBe("animated");
    expect(mediaKind(null, "still.jpg")).toBe("image");
  });

  it("defaults to 'image' when nothing identifies the media type", () => {
    expect(mediaKind(null, "mystery")).toBe("image");
  });
});

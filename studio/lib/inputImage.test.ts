import { describe, expect, it } from "vitest";
import {
  boundInputImage,
  INPUT_IMAGE_MAX_BYTES,
  inputImageFacts,
  normalizeInputImage,
} from "./inputImage";

describe("input image bounds", () => {
  it("keeps small supported images unchanged", async () => {
    const input = "original";
    let renders = 0;
    const result = await boundInputImage(
      input,
      100,
      { width: 800, height: 600 },
      async () => {
        renders++;
        return "unused";
      },
    );
    expect(result).toBe(input);
    expect(renders).toBe(0);
  });
  it("bounds axes and keeps the original aspect without cropping", async () => {
    const sizes: number[][] = [];
    const result = await boundInputImage(
      "original",
      100,
      { width: 8000, height: 4000 },
      async (width, height) => {
        sizes.push([width, height]);
        return "small";
      },
    );
    expect(sizes).toEqual([[4096, 2048]]);
    expect(result).toBe("small");
  });
  it("reduces dimensions again until PNG bytes fit, including small noisy images", async () => {
    const sizes: number[][] = [];
    await boundInputImage(
      "original",
      INPUT_IMAGE_MAX_BYTES + 1,
      { width: 1000, height: 500 },
      async (width, height) => {
        sizes.push([width, height]);
        return sizes.length === 1
          ? "x".repeat(4 * INPUT_IMAGE_MAX_BYTES)
          : "small";
      },
    );
    expect(sizes).toEqual([
      [1000, 500],
      [750, 375],
    ]);
  });
});

it("uses transformed bytes for MIME and filename", () => {
  expect(inputImageFacts("iVBORw0KGgo=", "photo.jpg")).toEqual({
    mimeType: "image/png",
    filename: "photo.png",
  });
  expect(inputImageFacts("/9j/4AAQ", "photo.jpeg")).toEqual({
    mimeType: "image/jpeg",
    filename: "photo.jpeg",
  });
  expect(inputImageFacts("unknown", "photo.tiff")).toEqual({
    mimeType: "application/octet-stream",
    filename: "photo.tiff",
  });
});

it("keeps unrecognized input bytes and filename without guessing a WebP container", async () => {
  expect(await normalizeInputImage("unknown", "photo.tiff")).toEqual({
    base64: "unknown",
    filename: "photo.tiff",
    mimeType: "application/octet-stream",
  });
});

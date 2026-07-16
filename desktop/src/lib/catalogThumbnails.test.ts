import { describe, expect, it } from "vitest";
import { catalogThumbnailUrl } from "./catalogThumbnails";

describe("catalogThumbnailUrl", () => {
  it("replaces Civitai source-resolution transforms with a cache-stable card size", () => {
    expect(
      catalogThumbnailUrl(
        "https://image.civitai.com/token/id/original=true,quality=90/preview.jpeg",
      ),
    ).toBe("https://image.civitai.com/token/id/width=512/preview.jpeg");
  });

  it("normalizes oversized legacy derivatives without creating a second layout URL", () => {
    expect(
      catalogThumbnailUrl("https://imagecache.civitai.com/token/id/width=1024/preview.jpeg"),
    ).toBe("https://imagecache.civitai.com/token/id/width=512/preview.jpeg");
  });

  it("leaves unrelated image hosts untouched", () => {
    const url = "https://example.com/original=true/preview.jpeg";
    expect(catalogThumbnailUrl(url)).toBe(url);
  });
});

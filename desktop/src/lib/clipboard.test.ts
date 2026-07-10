import { describe, expect, it, vi } from "vitest";
import { copyImageBytesToClipboard } from "./clipboard";

describe("copyImageBytesToClipboard", () => {
  it("copies fetched full-resolution bytes through the native image clipboard", async () => {
    const fromBytes = vi.fn(async () => ({ rid: 7 }));
    const writeImage = vi.fn(async () => {});
    const fetchImage = vi.fn(async () => new Uint8Array([137, 80, 78, 71]));

    await copyImageBytesToClipboard("/api/gallery/image/print.png", {
      fetchImage,
      native: true,
      fromBytes,
      writeImage,
    });

    expect(fetchImage).toHaveBeenCalledWith("/api/gallery/image/print.png");
    expect(fromBytes).toHaveBeenCalledWith(new Uint8Array([137, 80, 78, 71]));
    expect(writeImage).toHaveBeenCalledWith({ rid: 7 });
  });

  it("uses ClipboardItem in browser mode", async () => {
    const write = vi.fn(async () => {});
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: "image/png" });

    await copyImageBytesToClipboard("/image.png", {
      fetchImage: async () => new Uint8Array(await blob.arrayBuffer()),
      native: false,
      browserWrite: write,
      mimeType: "image/png",
    });

    expect(write).toHaveBeenCalledTimes(1);
  });
});

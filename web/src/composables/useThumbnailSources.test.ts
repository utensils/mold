import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  resolveThumbnailSrc: vi.fn(),
}));

vi.mock("../lib/hostRegistry", () => ({
  getHost: () => ({
    id: "remote",
    name: "remote",
    url: "http://remote:7680",
    apiKey: "secret",
  }),
}));

vi.mock("../lib/galleryMedia", () => ({
  resolveThumbnailSrc: mocks.resolveThumbnailSrc,
}));

import { useThumbnailSources } from "./useThumbnailSources";
import type { GalleryImage } from "../types";

async function settle(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}

function entry(mediaVersion: string, filename = "same.png"): GalleryImage {
  return {
    filename,
    timestamp: 1,
    media_type: "image",
    media_version: mediaVersion,
    hostId: "remote",
  } as GalleryImage;
}

describe("useThumbnailSources", () => {
  it("refetches when the same filename receives a new media version", async () => {
    mocks.resolveThumbnailSrc
      .mockResolvedValueOnce("blob:v1")
      .mockResolvedValueOnce("blob:v2");
    const sources = useThumbnailSources();

    expect(sources.srcFor(entry("v1"))).toBe("");
    await settle();
    expect(sources.srcFor(entry("v1"))).toBe("blob:v1");

    expect(sources.srcFor(entry("v2"))).toBe("");
    await settle();
    expect(sources.srcFor(entry("v2"))).toBe("blob:v2");
    expect(mocks.resolveThumbnailSrc).toHaveBeenCalledTimes(2);
  });

  it("evicts the least recently used resolved source without mutating during render", async () => {
    mocks.resolveThumbnailSrc.mockReset();
    mocks.resolveThumbnailSrc.mockImplementation(async (_, filename) =>
      Promise.resolve(`blob:${filename}`),
    );
    const sources = useThumbnailSources(2);
    const first = entry("v1", "first.png");
    const second = entry("v1", "second.png");
    const third = entry("v1", "third.png");

    sources.srcFor(first);
    await settle();
    sources.srcFor(second);
    await vi.waitFor(() =>
      expect(mocks.resolveThumbnailSrc).toHaveBeenCalledTimes(2),
    );
    await settle();
    expect(sources.srcFor(second)).toBe("blob:second.png");
    expect(sources.srcFor(first)).toBe("blob:first.png");
    sources.srcFor(third);
    await vi.waitFor(() =>
      expect(mocks.resolveThumbnailSrc).toHaveBeenCalledTimes(3),
    );
    await settle();

    expect(sources.srcFor(first)).toBe("blob:first.png");
    expect(sources.srcFor(second)).toBe("");
    await settle();
    expect(mocks.resolveThumbnailSrc).toHaveBeenCalledTimes(4);
  });
});

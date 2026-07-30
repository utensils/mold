import { describe, expect, it } from "vitest";
import {
  galleryIdentitySeed,
  galleryPrintIdentity,
  sameLogicalGalleryPrint,
} from "./galleryPrintIdentity";

describe("gallery print identity", () => {
  it("matches exact filenames across devices without requiring legacy metadata", () => {
    expect(
      sameLogicalGalleryPrint(
        { filename: "shared.png", timestamp: 1 },
        { filename: "shared.png", timestamp: 2 },
      ),
    ).toBe(true);
  });

  it("matches legacy mirrors by seed, bytes, model, and time window", () => {
    const origin = {
      filename: "origin.png",
      timestamp: 1_000,
      size_bytes: 4_096,
      metadata: { seed: 42, model: "flux-dev:q8" },
    };
    const mirror = {
      ...origin,
      filename: "old-autosave.png",
      timestamp: 1_010,
    };

    expect(galleryPrintIdentity(origin)).toBe("42:4096:flux-dev-q8");
    expect(sameLogicalGalleryPrint(origin, mirror)).toBe(true);
    expect(
      sameLogicalGalleryPrint(origin, { ...mirror, timestamp: 10_000 }),
    ).toBe(false);
  });

  it("recovers a trustworthy seed from synthetic auto-save filenames only", () => {
    expect(
      galleryIdentitySeed({
        filename: "mold-ltx-2-852494036-1721312349999.mp4",
        timestamp: 1,
        metadata_synthetic: true,
        metadata: { seed: 0 },
      }),
    ).toBe(852494036);
    expect(
      galleryIdentitySeed({
        filename: "legacy.mp4",
        timestamp: 1,
        metadata_synthetic: true,
        metadata: { seed: 0 },
      }),
    ).toBeNull();
  });
});

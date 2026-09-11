import { describe, expect, it } from "vitest";
import {
  catalogFetchCaption,
  catalogIdentityKey,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  isCatalogId,
  type CatalogSizeEntry,
} from "./catalogLabel";

function entry(part: Partial<CatalogSizeEntry> = {}): CatalogSizeEntry {
  return { size_bytes: null, ...part };
}

describe("catalogSizeInfo", () => {
  it("adds shared components to the weights for the fetch total", () => {
    const info = catalogSizeInfo(
      entry({
        size_bytes: 23_900_000_000,
        companion_details: [
          { size_bytes: 9_200_000_000 },
          { size_bytes: 1_700_000_000 },
        ],
      }),
    );
    expect(info.weightsBytes).toBe(23_900_000_000);
    expect(info.fetchBytes).toBe(34_800_000_000);
    expect(info.differs).toBe(true);
  });

  it("does not diverge when there are no companions", () => {
    const info = catalogSizeInfo(entry({ size_bytes: 6_400_000_000 }));
    expect(info.fetchBytes).toBe(6_400_000_000);
    expect(info.differs).toBe(false);
  });

  it("copes with an unknown weight size", () => {
    const info = catalogSizeInfo(entry({ size_bytes: null }));
    expect(info.weightsBytes).toBeNull();
    expect(info.fetchBytes).toBeNull();
    expect(info.differs).toBe(false);
  });
});

describe("catalog labels", () => {
  const two = catalogSizeInfo(
    entry({
      size_bytes: 23_900_000_000,
      companion_details: [{ size_bytes: 9_200_000_000 }],
    }),
  );
  const one = catalogSizeInfo(entry({ size_bytes: 6_400_000_000 }));
  const unknown = catalogSizeInfo(entry({ size_bytes: null }));

  it("shows SIZE · FETCH only when they differ", () => {
    expect(catalogSizeLabel(two)).toBe("SIZE 23.9 GB · FETCH 33.1 GB");
    expect(catalogSizeLabel(one)).toBe("SIZE 6.4 GB");
  });

  it("captions the fetch total only when shared components are added", () => {
    expect(catalogFetchCaption(two)).toBe(
      "33.1 GB to download, including shared components",
    );
    expect(catalogFetchCaption(one)).toBeNull();
  });

  it("labels the acquisition button with the download total", () => {
    expect(catalogPullLabel(two)).toBe("Pull · 33.1 GB");
    expect(catalogPullLabel(one)).toBe("Pull · 6.4 GB");
    // The verb is the caller's; every Studio shell passes the lexicon's.
    expect(catalogPullLabel(one, "Get it")).toBe("Get it · 6.4 GB");
    expect(catalogPullLabel(two, "Get it")).toBe("Get it · 33.1 GB");
  });

  it("falls back to the bare verb when no size is known", () => {
    expect(catalogPullLabel(unknown, "Get it")).toBe("Get it");
  });
});

describe("isCatalogId", () => {
  it("recognizes cv: and hf: ids, not plain names", () => {
    expect(isCatalogId("cv:8001")).toBe(true);
    expect(isCatalogId("hf:author/model")).toBe(true);
    expect(isCatalogId("flux-dev:q8")).toBe(false);
  });
});

describe("catalogIdentityKey", () => {
  it("uses source plus a non-empty upstream identity", () => {
    expect(
      catalogIdentityKey({ source: "hf", source_id: " author/model " }),
    ).toBe("hf:author/model");
  });

  it("does not treat an empty source id as model identity", () => {
    expect(
      catalogIdentityKey({ source: "civitai", source_id: " " }),
    ).toBeNull();
    expect(catalogIdentityKey({ source: "civitai" })).toBeNull();
  });
});

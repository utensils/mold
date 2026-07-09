import { describe, expect, it } from "vitest";
import {
  catalogFetchCaption,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  isCatalogId,
} from "./catalog";
import type { CatalogEntry } from "./api/types";

function entry(part: Partial<CatalogEntry>): CatalogEntry {
  return {
    id: "cv:1",
    source: "civitai",
    name: "x",
    family: "flux",
    kind: "lora",
    nsfw: false,
    installed: false,
    ...part,
  };
}

describe("catalogSizeInfo", () => {
  it("adds shared components to the weights for the fetch total", () => {
    const info = catalogSizeInfo(
      entry({
        size_bytes: 23_900_000_000,
        companion_details: [
          { name: "t5-xxl", size_bytes: 9_200_000_000 },
          { name: "clip-l", size_bytes: 1_700_000_000 },
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
      companion_details: [{ name: "t5", size_bytes: 9_200_000_000 }],
    }),
  );
  const one = catalogSizeInfo(entry({ size_bytes: 6_400_000_000 }));

  it("shows SIZE · FETCH only when they differ", () => {
    expect(catalogSizeLabel(two)).toBe("SIZE 23.9 GB · FETCH 33.1 GB");
    expect(catalogSizeLabel(one)).toBe("SIZE 6.4 GB");
  });

  it("captions the fetch total only when shared components are added", () => {
    expect(catalogFetchCaption(two)).toBe("33.1 GB to download, including shared components");
    expect(catalogFetchCaption(one)).toBeNull();
  });

  it("labels the Pull button with the download total", () => {
    expect(catalogPullLabel(two)).toBe("Pull · 33.1 GB");
    expect(catalogPullLabel(one)).toBe("Pull · 6.4 GB");
  });
});

describe("isCatalogId", () => {
  it("recognizes cv: and hf: ids, not plain names", () => {
    expect(isCatalogId("cv:8001")).toBe(true);
    expect(isCatalogId("hf:author/model")).toBe(true);
    expect(isCatalogId("flux-dev:q8")).toBe(false);
  });
});

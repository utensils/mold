import { describe, expect, it } from "vitest";
import {
  catalogFetchCaption,
  catalogPageUrl,
  catalogPullLabel,
  catalogSizeInfo,
  catalogSizeLabel,
  isCatalogId,
  sortInstalledFirst,
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

describe("catalogPageUrl", () => {
  it("prefers the server-provided page_url", () => {
    const url = catalogPageUrl(
      entry({ page_url: "https://civitai.com/models/9001?modelVersionId=8001" }),
    );
    expect(url).toBe("https://civitai.com/models/9001?modelVersionId=8001");
  });

  it("falls back to the HF repo page from source_id for older servers", () => {
    const url = catalogPageUrl(
      entry({ id: "hf:black-forest-labs/FLUX.1-dev", source_id: "black-forest-labs/FLUX.1-dev" }),
    );
    expect(url).toBe("https://huggingface.co/black-forest-labs/FLUX.1-dev");
  });

  it("derives the HF repo from the id when source_id is missing too", () => {
    expect(catalogPageUrl(entry({ id: "hf:author/model" }))).toBe(
      "https://huggingface.co/author/model",
    );
  });

  it("does not fabricate a page for companion pseudo-ids without a source_id", () => {
    expect(catalogPageUrl(entry({ id: "hf:companion/clip-l" }))).toBeNull();
  });

  it("companion rows with a backing repo link to that repo", () => {
    expect(
      catalogPageUrl(entry({ id: "hf:companion/clip-l", source_id: "openai/clip-vit-large" })),
    ).toBe("https://huggingface.co/openai/clip-vit-large");
  });

  it("cannot recover a Civitai model page without page_url", () => {
    // The wire only carries the version id; the model page needs the
    // parent model id, so older servers get no link rather than a 404.
    expect(catalogPageUrl(entry({ id: "cv:8001", source_id: "8001" }))).toBeNull();
  });
});

describe("sortInstalledFirst", () => {
  it("floats installed entries while preserving relative order in both groups", () => {
    const entries = [
      { id: "a", installed: false },
      { id: "b", installed: true },
      { id: "c", installed: false },
      { id: "d", installed: true },
    ];
    expect(sortInstalledFirst(entries).map((e) => e.id)).toEqual(["b", "d", "a", "c"]);
  });
});

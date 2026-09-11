import { describe, expect, it } from "vitest";
import { catalogPageUrl, catalogPullLabel, catalogSizeInfo, sortInstalledFirst } from "./catalog";
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

/* The size accounting, the acquisition label and the catalog-id helpers moved
 * to `@studio/lib/catalogLabel` and are covered by `catalogLabel.test.ts`. This
 * one case stays: it proves desktop's own wire type still satisfies the shared
 * structural type through the re-export. */
describe("the shared catalog labels reach desktop through the re-export", () => {
  it("labels the acquisition button from a desktop CatalogEntry", () => {
    const info = catalogSizeInfo(
      entry({
        size_bytes: 23_900_000_000,
        companion_details: [{ name: "t5", size_bytes: 9_200_000_000 }],
      }),
    );
    expect(catalogPullLabel(info, "Get it")).toBe("Get it · 33.1 GB");
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

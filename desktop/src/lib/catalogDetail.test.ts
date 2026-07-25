import { describe, expect, it } from "vitest";
import {
  buildDownloadContents,
  canDownloadEntry,
  catalogActionLabel,
  downloadContentsTotalBytes,
  installedModelToEntry,
  mergeCatalogSummaryDetail,
} from "./catalogDetail";
import type { CatalogEntry, ModelEntry } from "./api/types";

function installedModel(part: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name: "flux-dev:q8",
    family: "flux",
    size_gb: 11.8,
    is_loaded: false,
    hf_repo: "org/repo",
    default_steps: 28,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "The dev model.",
    downloaded: true,
    ...part,
  };
}

describe("installedModelToEntry", () => {
  it("adapts an installed model into the drawer's entry shape", () => {
    expect(installedModelToEntry(installedModel())).toMatchObject({
      id: "flux-dev:q8",
      name: "flux-dev:q8",
      family: "flux",
      source: "hf",
      source_id: "org/repo",
      installed: true,
      size_bytes: 11_800_000_000,
      page_url: "https://huggingface.co/org/repo",
      description: "The dev model.",
    });
  });

  it("keeps the upstream version id for Civitai installs and omits unreported fields", () => {
    const entry = installedModelToEntry(
      installedModel({ name: "cv:8001", hf_repo: "", size_gb: 0, description: "" }),
    );
    expect(entry).toMatchObject({
      id: "cv:8001",
      name: "cv:8001",
      source: "civitai",
      source_id: "8001",
      size_bytes: null,
      page_url: null,
      description: null,
    });
  });

  it("titles Civitai installs with their human-readable name, keeping the id for logic", () => {
    const entry = installedModelToEntry(
      installedModel({
        name: "cv:1759168",
        hf_repo: "",
        display_name: "Juggernaut XL - Ragnarok",
      }),
    );
    expect(entry.id).toBe("cv:1759168");
    expect(entry.name).toBe("cv:1759168");
    expect(entry.display_name).toBe("Juggernaut XL - Ragnarok");
  });

  it("leaves display_name empty when the name is already readable", () => {
    expect(installedModelToEntry(installedModel()).display_name).toBeNull();
  });

  it("preserves additive installed metadata and derives built-in utility kinds", () => {
    expect(
      installedModelToEntry(
        installedModel({
          family: "flux",
          kind: "lora",
          modality: "image",
          nsfw: true,
        }),
      ),
    ).toMatchObject({ kind: "lora", modality: "image", nsfw: true });
    expect(installedModelToEntry(installedModel({ family: "real-esrgan" })).kind).toBe("upscaler");
    expect(installedModelToEntry(installedModel({ family: "qwen3-expand" })).kind).toBe(
      "prompt-expander",
    );
  });

  it("preserves unknown safety metadata instead of classifying it as safe", () => {
    expect(installedModelToEntry(installedModel({ nsfw: null })).nsfw).toBeNull();
    expect(installedModelToEntry(installedModel({ nsfw: true })).nsfw).toBe(true);
  });

  it("suppresses a description that merely repeats the derived display title", () => {
    const entry = installedModelToEntry(
      installedModel({
        name: "cv:1759168",
        hf_repo: "",
        display_name: "Juggernaut XL - Ragnarok",
        description: "  Juggernaut XL - Ragnarok  ",
      }),
    );
    expect(entry.display_name).toBe("Juggernaut XL - Ragnarok");
    expect(entry.description).toBeNull();
  });

  it("suppresses the legacy installed name-by-author placeholder", () => {
    const entry = installedModelToEntry(
      installedModel({
        name: "cv:1759168",
        hf_repo: "",
        display_name: "Juggernaut XL - Ragnarok",
        description: "Juggernaut XL - Ragnarok by KandooAI",
      }),
    );
    expect(entry.description).toBeNull();
  });

  it("keeps the repo id encoded in hf: catalog install names", () => {
    expect(
      installedModelToEntry(installedModel({ name: "hf:org/thing", hf_repo: "" })),
    ).toMatchObject({
      source: "hf",
      source_id: "org/thing",
      page_url: "https://huggingface.co/org/thing",
    });
  });

  it("classifies repo-less models as local files, not Hugging Face", () => {
    expect(
      installedModelToEntry(installedModel({ name: "my-finetune", hf_repo: "" })),
    ).toMatchObject({ source: "local", source_id: null, page_url: null });
  });
});

describe("mergeCatalogSummaryDetail", () => {
  it("keeps useful summary metadata when a detail response omits or defaults it", () => {
    const summaryEntry = entry({
      author: "Summary author",
      description: "A useful summary description.",
      license: "apache-2.0",
      tags: ["portrait", "cinematic"],
      trained_words: ["studio portrait"],
      page_url: "https://huggingface.co/author/model",
      thumbnail_url: "https://cdn.example/listing.webp",
      download_count: 12_300,
      likes: 456,
      rating: 4.7,
      installed: true,
      nsfw: true,
    });
    const detailEntry = entry({
      author: null,
      description: " ",
      license: null,
      tags: [],
      trained_words: [],
      page_url: null,
      thumbnail_url: null,
      download_count: 0,
      likes: 0,
      rating: null,
      installed: false,
      nsfw: false,
    });

    expect(mergeCatalogSummaryDetail(summaryEntry, detailEntry)).toMatchObject({
      author: "Summary author",
      description: "A useful summary description.",
      license: "apache-2.0",
      tags: ["portrait", "cinematic"],
      trained_words: ["studio portrait"],
      page_url: "https://huggingface.co/author/model",
      thumbnail_url: "https://cdn.example/listing.webp",
      download_count: 12_300,
      likes: 456,
      rating: 4.7,
      installed: true,
      nsfw: true,
    });
  });

  it("uses meaningful detail metadata while retaining the clicked summary thumbnail", () => {
    const summaryEntry = entry({
      author: "Summary author",
      description: "Summary copy.",
      tags: ["summary"],
      thumbnail_url: "https://cdn.example/listing.webp",
      download_count: 100,
    });
    const detailEntry = entry({
      author: "Detail author",
      description: "Detailed copy.",
      tags: ["detail"],
      thumbnail_url: "https://cdn.example/detail.webp",
      download_count: 200,
    });

    expect(mergeCatalogSummaryDetail(summaryEntry, detailEntry)).toMatchObject({
      author: "Detail author",
      description: "Detailed copy.",
      tags: ["detail"],
      thumbnail_url: "https://cdn.example/listing.webp",
      download_count: 200,
    });
  });
});

function entry(part: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: "hf:author/model",
    source: "hf",
    name: "model",
    family: "flux",
    kind: "checkpoint",
    nsfw: false,
    installed: false,
    size_bytes: null,
    ...part,
  };
}

describe("buildDownloadContents", () => {
  it("itemizes primary recipe files and companion details with sizes", () => {
    const items = buildDownloadContents(
      entry({
        download_recipe: {
          needs_token: "civitai",
          files: [
            {
              url: "https://civitai.example/model",
              dest: "flux2/civitai/2910912/moody.safetensors",
              sha256: "abc",
              size_bytes: 8_000_000_000,
            },
          ],
        },
        companion_details: [
          {
            name: "flux2-te-9b",
            kind: "text-encoder",
            repo: "black-forest-labs/FLUX.2-klein-9B",
            size_bytes: 16_000_000_000,
          },
          { name: "flux2-vae", kind: "vae", size_bytes: 168_000_000 },
        ],
      }),
    );

    expect(items).toEqual([
      {
        key: "primary:flux2/civitai/2910912/moody.safetensors",
        label: "moody.safetensors",
        kind: "primary",
        sizeBytes: 8_000_000_000,
      },
      {
        key: "companion:flux2-te-9b",
        label: "flux2-te-9b",
        kind: "text-encoder",
        sizeBytes: 16_000_000_000,
      },
      { key: "companion:flux2-vae", label: "flux2-vae", kind: "vae", sizeBytes: 168_000_000 },
    ]);
  });

  it("tolerates older-server entries with neither recipe nor companion details", () => {
    expect(buildDownloadContents(entry())).toEqual([]);
  });

  it("labels a dest with no basename as the primary model", () => {
    const items = buildDownloadContents(
      entry({
        download_recipe: {
          needs_token: null,
          files: [{ url: "u", dest: "", sha256: null, size_bytes: null }],
        },
      }),
    );
    expect(items[0]?.label).toBe("Primary model");
  });

  it("falls back to a generic component kind when a companion omits kind", () => {
    const items = buildDownloadContents(
      entry({ companion_details: [{ name: "mystery", size_bytes: null }] }),
    );
    expect(items).toEqual([
      { key: "companion:mystery", label: "mystery", kind: "component", sizeBytes: null },
    ]);
  });
});

describe("downloadContentsTotalBytes", () => {
  it("sums known sizes but flags the total incomplete when some are unknown", () => {
    const total = downloadContentsTotalBytes([
      { key: "a", label: "a", kind: "primary", sizeBytes: 8_000_000_000 },
      { key: "b", label: "b", kind: "vae", sizeBytes: null },
      { key: "c", label: "c", kind: "text-encoder", sizeBytes: 16_000_000_000 },
    ]);
    // Partial sum is a lower bound — the null-sized item is missing from it.
    expect(total).toEqual({ bytes: 24_000_000_000, complete: false });
  });

  it("reports the total complete when every item has a size", () => {
    const total = downloadContentsTotalBytes([
      { key: "a", label: "a", kind: "primary", sizeBytes: 8_000_000_000 },
      { key: "c", label: "c", kind: "text-encoder", sizeBytes: 16_000_000_000 },
    ]);
    expect(total).toEqual({ bytes: 24_000_000_000, complete: true });
  });

  it("has null bytes when nothing has a size", () => {
    expect(downloadContentsTotalBytes([])).toEqual({ bytes: null, complete: true });
    expect(
      downloadContentsTotalBytes([{ key: "a", label: "a", kind: "vae", sizeBytes: null }]),
    ).toEqual({ bytes: null, complete: false });
  });
});

describe("catalogActionLabel", () => {
  it("is Pull for available entries and Repair for installed ones", () => {
    expect(catalogActionLabel(entry())).toBe("Pull");
    expect(catalogActionLabel(entry({ installed: true }))).toBe("Repair");
  });
});

describe("canDownloadEntry", () => {
  it("allows supported entries", () => {
    expect(canDownloadEntry(entry({ supported: true }))).toBe(true);
  });

  it("blocks unsupported catalog packages", () => {
    expect(canDownloadEntry(entry({ supported: false }))).toBe(false);
  });

  it("allows entries from older servers that don't report support", () => {
    expect(canDownloadEntry(entry())).toBe(true);
  });
});

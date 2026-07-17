import { describe, expect, it } from "vitest";
import {
  buildDownloadContents,
  canDownloadEntry,
  catalogActionLabel,
  downloadContentsTotalBytes,
} from "./catalogDetail";
import type { CatalogEntry } from "./api/types";

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
  it("sums known sizes, skipping unknown ones", () => {
    const total = downloadContentsTotalBytes([
      { key: "a", label: "a", kind: "primary", sizeBytes: 8_000_000_000 },
      { key: "b", label: "b", kind: "vae", sizeBytes: null },
      { key: "c", label: "c", kind: "text-encoder", sizeBytes: 16_000_000_000 },
    ]);
    expect(total).toBe(24_000_000_000);
  });

  it("is null when nothing has a size", () => {
    expect(downloadContentsTotalBytes([])).toBeNull();
    expect(
      downloadContentsTotalBytes([{ key: "a", label: "a", kind: "vae", sizeBytes: null }]),
    ).toBeNull();
  });
});

describe("catalogActionLabel", () => {
  it("is Pull for available entries and Repair for installed ones", () => {
    expect(catalogActionLabel(entry())).toBe("Pull");
    expect(catalogActionLabel(entry({ installed: true }))).toBe("Repair");
  });
});

describe("canDownloadEntry", () => {
  it("allows entries whose engine phase is shipped (<= 5)", () => {
    expect(canDownloadEntry(entry({ engine_phase: 1 }))).toBe(true);
    expect(canDownloadEntry(entry({ engine_phase: 5 }))).toBe(true);
  });

  it("blocks unsupported catalog packages (phase >= 6)", () => {
    expect(canDownloadEntry(entry({ engine_phase: 6 }))).toBe(false);
  });

  it("allows entries from older servers that don't report a phase", () => {
    expect(canDownloadEntry(entry())).toBe(true);
  });
});

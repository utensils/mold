import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCatalog } from "./useCatalog";

const originalFetch = globalThis.fetch;

beforeEach(() => {
  globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
    if (url.startsWith("/api/catalog/families")) {
      return {
        ok: true,
        json: async () => ({
          families: [{ family: "flux", foundation: 1, finetune: 4 }],
        }),
      };
    }
    if (url.startsWith("/api/catalog?")) {
      return {
        ok: true,
        json: async () => ({
          entries: [
            {
              id: "hf:a",
              name: "Alpha",
              family: "flux",
              engine_phase: 1,
              source: "hf",
              source_id: "a",
              author: null,
              family_role: "foundation",
              sub_family: null,
              modality: "image",
              kind: "checkpoint",
              file_format: "safetensors",
              bundling: "separated",
              size_bytes: 1,
              download_count: 100,
              rating: null,
              likes: 0,
              nsfw: false,
              thumbnail_url: null,
              description: null,
              license: null,
              license_flags: null,
              tags: [],
              companions: [],
              download_recipe: { files: [], needs_token: null },
              created_at: null,
              updated_at: null,
              added_at: 0,
            },
          ],
          page: 1,
          page_size: 48,
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
});
afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("useCatalog", () => {
  it("loads families and entries on init", async () => {
    const cat = useCatalog();
    await cat.refresh();
    expect(cat.entries.value.length).toBe(1);
    expect(cat.families.value[0].family).toBe("flux");
  });

  it("setFilter triggers a re-fetch", async () => {
    const cat = useCatalog();
    await cat.refresh();
    (globalThis.fetch as any).mockClear();
    cat.setFilter({ family: "flux" });
    await new Promise((r) => setTimeout(r, 300)); // past the 250ms debounce
    expect(
      (globalThis.fetch as any).mock.calls.some((c: any[]) =>
        (c[0] as string).includes("family=flux"),
      ),
    ).toBe(true);
  });

  it("disables download for entries with engine_phase >= 2", async () => {
    const cat = useCatalog();
    expect(cat.canDownload({ engine_phase: 1 } as any)).toBe(true);
    expect(cat.canDownload({ engine_phase: 2 } as any)).toBe(false);
    expect(cat.canDownload({ engine_phase: 99 } as any)).toBe(false);
  });
});

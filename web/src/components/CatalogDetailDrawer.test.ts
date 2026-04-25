import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ref } from "vue";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";

const makeEntry = (phase: number) => ({
  id: "hf:a",
  name: "Alpha",
  family: "flux",
  engine_phase: phase,
  source: "hf" as const,
  source_id: "a",
  author: "alice",
  family_role: "finetune" as const,
  sub_family: null,
  modality: "image" as const,
  kind: "checkpoint" as const,
  file_format: "safetensors" as const,
  bundling: "separated" as const,
  size_bytes: 6_000_000_000,
  download_count: 1234,
  rating: 4.7,
  likes: 0,
  nsfw: false,
  thumbnail_url: null,
  description: "A test model",
  license: null,
  license_flags: null,
  tags: [],
  companions: [],
  download_recipe: { files: [], needs_token: null },
  created_at: null,
  updated_at: null,
  added_at: 0,
});

const mockCloseDetail = vi.fn();
const mockStartDownload = vi.fn();
const mockCanDownload = vi.fn(
  (e: { engine_phase: number }) => e.engine_phase === 1,
);
const mockDetail = ref<ReturnType<typeof makeEntry> | null>(null);

vi.mock("../composables/useCatalog", () => {
  return {
    useCatalog: () => ({
      detail: mockDetail,
      closeDetail: mockCloseDetail,
      startDownload: mockStartDownload,
      canDownload: mockCanDownload,
    }),
  };
});

describe("CatalogDetailDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders entry name and description when detail is set", () => {
    mockDetail.value = makeEntry(1);
    const w = mount(CatalogDetailDrawer);
    expect(w.text()).toContain("Alpha");
    expect(w.text()).toContain("A test model");
  });

  it("Download button is enabled for engine_phase === 1", () => {
    mockDetail.value = makeEntry(1);
    const w = mount(CatalogDetailDrawer);
    const btn = w.find("[data-test=download-btn]");
    expect(btn.exists()).toBe(true);
    expect((btn.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("Download button is disabled when engine_phase >= 2 and shows phase tooltip", () => {
    mockDetail.value = makeEntry(3);
    const w = mount(CatalogDetailDrawer);
    const btn = w.find("[data-test=download-btn]");
    expect(btn.exists()).toBe(true);
    expect((btn.element as HTMLButtonElement).disabled).toBe(true);
    const title = btn.attributes("title") ?? "";
    expect(title).toMatch(/phase 3|coming/i);
  });

  it("close button calls closeDetail", async () => {
    mockDetail.value = makeEntry(1);
    const w = mount(CatalogDetailDrawer);
    await w.find("[data-test=close-btn]").trigger("click");
    expect(mockCloseDetail).toHaveBeenCalledOnce();
  });
});

import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ref } from "vue";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";
import type { CatalogEntryWire } from "../types";

const makeEntry = (phase: number): CatalogEntryWire => ({
  id: "hf:a",
  name: "Alpha",
  family: "flux",
  engine_phase: phase,
  installed: false,
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
  companion_details: [],
  download_recipe: { files: [], needs_token: null },
  primary_path: null,
  created_at: null,
  updated_at: null,
  added_at: 0,
});

const mockCloseDetail = vi.fn();
const mockStartDownload = vi.fn();
const mockCanDownload = vi.fn(
  (e: { engine_phase: number }) => e.engine_phase <= 5,
);
const mockDetail = ref<CatalogEntryWire | null>(null);

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

  it("Download button is enabled for engine_phase === 2 (single-file SD1.5/SDXL)", () => {
    mockDetail.value = makeEntry(2);
    const w = mount(CatalogDetailDrawer);
    const btn = w.find("[data-test=download-btn]");
    expect(btn.exists()).toBe(true);
    expect((btn.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("Download button is enabled for engine_phase 5 (LTX single-file)", () => {
    mockDetail.value = makeEntry(5);
    const w = mount(CatalogDetailDrawer);
    const btn = w.find("[data-test=download-btn]");
    expect(btn.exists()).toBe(true);
    expect((btn.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("Download button is disabled when engine_phase >= 6 and shows unsupported tooltip", () => {
    mockDetail.value = makeEntry(6);
    const w = mount(CatalogDetailDrawer);
    const btn = w.find("[data-test=download-btn]");
    expect(btn.exists()).toBe(true);
    expect((btn.element as HTMLButtonElement).disabled).toBe(true);
    const title = btn.attributes("title") ?? "";
    expect(title).toMatch(/unsupported/i);
    expect(title).not.toMatch(/phase 6|coming/i);
  });

  it("does not render the Coming-in-phase badge for engine_phase 5", () => {
    mockDetail.value = makeEntry(5);
    const w = mount(CatalogDetailDrawer);
    expect(w.text()).not.toMatch(/coming in phase 5/i);
  });

  it("close button calls closeDetail", async () => {
    mockDetail.value = makeEntry(1);
    const w = mount(CatalogDetailDrawer);
    await w.find("[data-test=close-btn]").trigger("click");
    expect(mockCloseDetail).toHaveBeenCalledOnce();
  });

  it("hides Download and shows Repair when entry is installed", () => {
    mockDetail.value = { ...makeEntry(1), installed: true };
    const w = mount(CatalogDetailDrawer);
    expect(w.find("[data-test=download-btn]").exists()).toBe(false);
    const repair = w.find("[data-test=repair-btn]");
    expect(repair.exists()).toBe(true);
    expect((repair.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("Repair button calls startDownload with the entry id", async () => {
    mockDetail.value = { ...makeEntry(1), installed: true };
    const w = mount(CatalogDetailDrawer);
    await w.find("[data-test=repair-btn]").trigger("click");
    expect(mockStartDownload).toHaveBeenCalledWith("hf:a");
  });

  it("renders an Installed chip when entry is installed", () => {
    mockDetail.value = { ...makeEntry(1), installed: true };
    const w = mount(CatalogDetailDrawer);
    expect(w.text()).toMatch(/installed/i);
  });

  it("renders primary and required companion download contents", () => {
    mockDetail.value = {
      ...makeEntry(3),
      download_recipe: {
        needs_token: "civitai" as const,
        files: [
          {
            url: "https://civitai.example/model",
            dest: "flux2/civitai/2910912/moody.safetensors",
            sha256: "abc",
            size_bytes: 8_000_000_000,
          },
        ],
      },
      companions: ["flux2-te-9b", "flux2-vae"],
      companion_details: [
        {
          name: "flux2-te-9b",
          kind: "text-encoder" as const,
          repo: "black-forest-labs/FLUX.2-klein-9B",
          size_bytes: 16_000_000_000,
        },
        {
          name: "flux2-vae",
          kind: "vae" as const,
          repo: "black-forest-labs/FLUX.2-klein-4B",
          size_bytes: 168_000_000,
        },
      ],
    };
    const w = mount(CatalogDetailDrawer);

    expect(w.find("[data-test=download-contents]").exists()).toBe(true);
    expect(w.text()).toContain("moody.safetensors");
    expect(w.text()).toContain("flux2-te-9b");
    expect(w.text()).toContain("flux2-vae");
    expect(w.text()).toContain("24.2 GB");
  });
});

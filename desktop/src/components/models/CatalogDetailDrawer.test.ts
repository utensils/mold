import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import type { CatalogEntry } from "../../lib/api/types";

const { fetchCatalogDetail, startCatalogDownload } = vi.hoisted(() => ({
  fetchCatalogDetail: vi.fn(),
  startCatalogDownload: vi.fn(),
}));
vi.mock("../../lib/api/catalog", () => ({ fetchCatalogDetail, startCatalogDownload }));
const { fetchModelComponents } = vi.hoisted(() => ({ fetchModelComponents: vi.fn() }));
vi.mock("../../lib/api/models", () => ({ fetchModelComponents }));

import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";

function summary(part: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: "cv:8001",
    source: "civitai",
    source_id: "8001",
    name: "Moody Flux",
    author: "alice",
    family: "flux2",
    kind: "checkpoint",
    nsfw: false,
    installed: false,
    size_bytes: 8_000_000_000,
    download_count: 4_242,
    likes: 88,
    thumbnail_url: null,
    ...part,
  };
}

function detail(part: Partial<CatalogEntry> = {}): CatalogEntry {
  return summary({
    modality: "image",
    file_format: "safetensors",
    engine_phase: 3,
    description: "A moody cinematic finetune.",
    license: "CreativeML Open RAIL-M",
    tags: ["cinematic", "portrait"],
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
    ...part,
  });
}

async function mountDrawer(
  entry: CatalogEntry,
  extra: Record<string, unknown> = {},
): Promise<ReturnType<typeof mount>> {
  const wrapper = mount(CatalogDetailDrawer, {
    props: { entry, pulling: false, ...extra },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchCatalogDetail.mockResolvedValue(detail());
  fetchModelComponents.mockRejectedValue(new Error("not installed here"));
});

describe("CatalogDetailDrawer", () => {
  it("lists on-disk component state with a presence summary for installed entries", async () => {
    fetchModelComponents.mockResolvedValue({
      model: "cv:8001",
      components: [
        { kind: "vae", name: "vae", present: true },
        { kind: "text-encoder", name: "clip-l", present: false, repair_model: "cv:8001" },
        { kind: "clip", name: "orphan", present: false, repair_model: null },
      ],
    });
    const wrapper = await mountDrawer(summary({ installed: true }));

    expect(fetchModelComponents).toHaveBeenCalledWith("cv:8001", undefined);
    const section = wrapper.get("[data-test='component-list']");
    expect(section.text()).toContain("1/3 present");
    expect(section.findAll("[data-test='component-row']")).toHaveLength(3);
    // Repair only where the server names a repair_model for a missing part.
    expect(section.findAll("[data-test='component-repair']")).toHaveLength(1);
  });

  it("never queries component state for entries that are not installed", async () => {
    const wrapper = await mountDrawer(summary());
    expect(fetchModelComponents).not.toHaveBeenCalled();
    expect(wrapper.find("[data-test='component-list']").exists()).toBe(false);
  });

  it("fetches the detail for the entry and renders description, license, and tags", async () => {
    const wrapper = await mountDrawer(summary());
    expect(fetchCatalogDetail).toHaveBeenCalledWith("cv:8001", false, undefined);
    expect(wrapper.text()).toContain("Moody Flux");
    expect(wrapper.text()).toContain("A moody cinematic finetune.");
    expect(wrapper.text()).toContain("CreativeML Open RAIL-M");
    expect(wrapper.text()).toContain("cinematic");
    expect(wrapper.text()).toContain("portrait");
  });

  it("renders modality and file format from the detail", async () => {
    const wrapper = await mountDrawer(summary());
    expect(wrapper.text()).toContain("image");
    expect(wrapper.text()).toContain("safetensors");
  });

  it("itemizes download contents with per-file sizes and the computed total", async () => {
    const wrapper = await mountDrawer(summary());
    const contents = wrapper.get("[data-test='download-contents']");
    expect(contents.text()).toContain("moody.safetensors");
    expect(contents.text()).toContain("flux2-te-9b");
    expect(contents.text()).toContain("flux2-vae");
    expect(contents.text()).toContain("16.0 GB");
    // 8 GB weights + 16 GB TE + 0.168 GB VAE.
    expect(contents.text()).toContain("24.2 GB");
  });

  it("marks the download total a lower bound when a companion omits its size", async () => {
    fetchCatalogDetail.mockResolvedValue(
      detail({
        companion_details: [
          {
            name: "flux2-te-9b",
            kind: "text-encoder",
            repo: "black-forest-labs/FLUX.2-klein-9B",
            size_bytes: 16_000_000_000,
          },
          { name: "flux2-vae", kind: "vae", size_bytes: null },
        ],
      }),
    );
    const wrapper = await mountDrawer(summary());
    // 8 GB weights + 16 GB TE, VAE size unknown — the shown total is a floor.
    expect(wrapper.get("[data-test='download-contents']").text()).toContain("≥ 24.0 GB");
    expect(wrapper.get("[data-test='drawer-pull']").text()).toContain("Pull · ≥ 24.0 GB");
  });

  it("shows an exact download total when every item reports a size", async () => {
    const wrapper = await mountDrawer(summary());
    const pull = wrapper.get("[data-test='drawer-pull']");
    expect(pull.text()).toContain("Pull · 24.2 GB");
    expect(pull.text()).not.toContain("≥");
  });

  it("targets the given host and forwards credentials for the detail fetch", async () => {
    const target = { baseUrl: "http://hal9000:7680", apiKey: "hk" };
    await mountDrawer(summary(), { target, forwardCredentials: true });
    expect(fetchCatalogDetail).toHaveBeenCalledWith("cv:8001", true, target);
  });

  it("shows Pull for available entries and emits pull with the detailed entry", async () => {
    const wrapper = await mountDrawer(summary());
    expect(wrapper.find("[data-test='drawer-repair']").exists()).toBe(false);
    const pull = wrapper.get("[data-test='drawer-pull']");
    await pull.trigger("click");
    expect(wrapper.emitted("pull")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });
  });

  it("swaps Pull for Repair when the entry is installed", async () => {
    fetchCatalogDetail.mockResolvedValue(detail({ installed: true }));
    const wrapper = await mountDrawer(summary({ installed: true }));
    expect(wrapper.find("[data-test='drawer-pull']").exists()).toBe(false);
    const repair = wrapper.get("[data-test='drawer-repair']");
    expect(repair.text()).toContain("Repair");
    await repair.trigger("click");
    expect(wrapper.emitted("pull")?.[0]?.[0]).toMatchObject({ id: "cv:8001" });
  });

  it("disables the action for unsupported catalog packages (engine_phase >= 6)", async () => {
    fetchCatalogDetail.mockResolvedValue(detail({ engine_phase: 6 }));
    const wrapper = await mountDrawer(summary());
    const pull = wrapper.get("[data-test='drawer-pull']");
    expect((pull.element as HTMLButtonElement).disabled).toBe(true);
    expect(pull.attributes("title")).toMatch(/unsupported/i);
    expect(wrapper.text()).toContain("Unsupported catalog package");
  });

  it("keeps rendering the summary fields when the detail fetch fails", async () => {
    fetchCatalogDetail.mockRejectedValue(new Error("upstream 502"));
    const wrapper = await mountDrawer(summary());
    expect(wrapper.text()).toContain("Moody Flux");
    // No detail — descriptive sections stay absent, the action still works.
    expect(wrapper.find("[data-test='download-contents']").exists()).toBe(false);
    expect(wrapper.find("[data-test='drawer-pull']").exists()).toBe(true);
  });

  it("emits close from the close button and from Escape", async () => {
    const wrapper = await mountDrawer(summary());
    await wrapper.get("[data-test='drawer-close']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    expect(wrapper.emitted("close")).toHaveLength(2);
  });
});

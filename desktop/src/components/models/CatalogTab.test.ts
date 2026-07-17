import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { CatalogEntry } from "../../lib/api/types";
import type { CatalogSearchParams } from "../../lib/api/catalog";

const { searchCatalog, fetchCatalogFamilies, startCatalogDownload } = vi.hoisted(() => ({
  searchCatalog: vi.fn(),
  fetchCatalogFamilies: vi.fn(),
  startCatalogDownload: vi.fn(),
}));
vi.mock("../../lib/api/catalog", () => ({
  searchCatalog,
  fetchCatalogFamilies,
  startCatalogDownload,
}));
vi.mock("../../lib/catalogSizes", () => ({
  resolveEntrySize: vi.fn().mockResolvedValue(null),
}));
vi.mock("../../lib/openExternal", () => ({ openExternal: vi.fn() }));

import CatalogTab from "./CatalogTab.vue";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
import { useModelStore } from "../../stores/models";

const PAGE_SIZE = 24;

function entry(name: string, family: string): CatalogEntry {
  return {
    id: `hf:${name}`,
    source: "hf",
    source_id: name,
    name,
    family,
    kind: "checkpoint",
    nsfw: false,
    installed: false,
    size_bytes: 1_000,
    thumbnail_url: null,
    page_url: null,
  } as CatalogEntry;
}

/** A full page of image-family entries (no video families anywhere). */
function imagePage(page: number, size = PAGE_SIZE): CatalogEntry[] {
  return Array.from({ length: size }, (_, i) => entry(`image-${page}-${i}`, "flux"));
}

async function mountTab(mediaType: "all" | "image" | "video" = "all") {
  setActivePinia(createPinia());
  const wrapper = mount(CatalogTab, {
    props: { query: "", layout: "grid" as const, mediaType },
    global: { plugins: [] },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchCatalogFamilies.mockResolvedValue([]);
  searchCatalog.mockResolvedValue({ entries: [], page: 1, page_size: PAGE_SIZE, total: 0 });
});

describe("CatalogTab media filter under pagination", () => {
  it("excludes an installed model by name when its catalog id is source-prefixed", async () => {
    setActivePinia(createPinia());
    searchCatalog.mockResolvedValue({
      entries: [entry("flux-dev:q8", "flux")],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        excludeInstalled: true,
        installedIds: ["flux-dev:q8"],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    expect(wrapper.text()).not.toContain("flux-dev:q8");
    expect(wrapper.get("[data-test='catalog-empty']").text()).toContain("already installed");
  });

  it("offers safe manifest variants and hides aggregate HF rows for the same repo", async () => {
    setActivePinia(createPinia());
    useModelStore().all = [
      {
        name: "ltx-video-0.9.8-2b-distilled:bf16",
        family: "ltx-video",
        size_gb: 4.2,
        is_loaded: false,
        hf_repo: "Lightricks/LTX-Video",
        default_steps: 8,
        default_guidance: 1,
        default_width: 768,
        default_height: 512,
        description: "Small LTX-Video",
        downloaded: false,
        remaining_download_bytes: 16_200_000_000,
      },
    ];
    searchCatalog.mockResolvedValue({
      entries: [
        {
          ...entry("LTX-Video", "ltx-video"),
          source_id: "Lightricks/LTX-Video",
          bundling: "separated",
          size_bytes: 253_800_000_000,
        },
      ],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "grid" as const, mediaType: "video" },
      global: { plugins: [] },
    });
    await flushPromises();

    expect(wrapper.text()).toContain("ltx-video-0.9.8-2b-distilled:bf16");
    expect(wrapper.text()).toContain("SIZE 4.2 GB · FETCH 16.2 GB");
    expect(wrapper.text()).not.toContain("253.8 GB");
  });

  it("auto-fetches follow-up pages until the Video filter has content", async () => {
    // Page 1 is all image models (the common HF/Civitai shape); the first
    // video model only appears on page 2.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve(
        params.page === 1
          ? { entries: imagePage(1), page: 1, page_size: PAGE_SIZE, total: 48 }
          : {
              entries: [entry("LTX-2 Distilled", "ltx2"), ...imagePage(2, PAGE_SIZE - 1)],
              page: 2,
              page_size: PAGE_SIZE,
              total: 48,
            },
      ),
    );
    const wrapper = await mountTab("video");

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as CatalogSearchParams).page);
    expect(pages).toEqual([1, 2]);
    expect(wrapper.text()).toContain("LTX-2 Distilled");
    expect(wrapper.find("[data-test='catalog-empty']").exists()).toBe(false);
  });

  it("shows a filter-aware message instead of a blank grid when no video models exist", async () => {
    // One full image page, then a short page: results exhausted, zero video.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve(
        params.page === 1
          ? { entries: imagePage(1), page: 1, page_size: PAGE_SIZE, total: 27 }
          : { entries: imagePage(2, 3), page: 2, page_size: PAGE_SIZE, total: 27 },
      ),
    );
    const wrapper = await mountTab("video");

    const empty = wrapper.get("[data-test='catalog-empty']");
    expect(empty.text()).toContain("No video models");
    // Results are exhausted — no dangling Load more button.
    expect(wrapper.findAll("button").some((b) => b.text().includes("Load more"))).toBe(false);
  });

  it("offers a clear-filter affordance that emits clear-media-filter", async () => {
    searchCatalog.mockResolvedValue({
      entries: imagePage(1, 3),
      page: 1,
      page_size: PAGE_SIZE,
      total: 3,
    });
    const wrapper = await mountTab("video");

    const clear = wrapper.get("[data-test='clear-media-filter']");
    await clear.trigger("click");
    expect(wrapper.emitted("clear-media-filter")).toHaveLength(1);
  });

  it("bounds the auto-fetch and keeps Load more available for further pages", async () => {
    // Every page is a full page of image models — the loop must stop at its
    // bound instead of walking the whole catalog.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 10_000,
      }),
    );
    const wrapper = await mountTab("video");

    expect(searchCatalog.mock.calls.length).toBeLessThanOrEqual(5);
    expect(wrapper.get("[data-test='catalog-empty']").text()).toContain("No video models");
    // More pages exist — the user can keep digging manually.
    expect(wrapper.findAll("button").some((b) => b.text().includes("Load more"))).toBe(true);
  });

  it("keys the empty state on the filtered list, not the raw page", async () => {
    // A page with content that the Video chip filters out entirely must not
    // render a message-less blank grid (the original bug).
    searchCatalog.mockResolvedValue({
      entries: imagePage(1, 5),
      page: 1,
      page_size: PAGE_SIZE,
      total: 5,
    });
    const wrapper = await mountTab("video");

    expect(wrapper.find("[data-test='catalog-empty']").exists()).toBe(true);
  });
});

describe("CatalogTab host fallback when the local engine is down", () => {
  it("targets the first ready host with credential forwarding when the primary isn't ready", async () => {
    setActivePinia(createPinia());
    // Local engine failed to start: no primary connection info at all.
    useConnectionStore().info = null;
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: "hk",
      status: "ready",
      error: null,
      instanceId: null,
    });
    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "grid" as const },
      global: { plugins: [] },
    });
    await flushPromises();

    // Search runs against the ready remote host, forwarding credentials.
    const [, forward, target] = searchCatalog.mock.calls[0] as [
      CatalogSearchParams,
      boolean,
      { baseUrl: string; apiKey: string | null } | undefined,
    ];
    expect(target).toEqual({ baseUrl: "http://hal9000:7680", apiKey: "hk" });
    expect(forward).toBe(true);
    // Families load from the same fallback host.
    const familiesCall = fetchCatalogFamilies.mock.calls[0] as [boolean, unknown];
    expect(familiesCall[0]).toBe(true);
    expect(familiesCall[1]).toEqual({ baseUrl: "http://hal9000:7680", apiKey: "hk" });
    wrapper.unmount();
  });

  it("keeps targeting the primary (no explicit target) when it is ready", async () => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: null };
    conn.status = "ready";
    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "grid" as const },
      global: { plugins: [] },
    });
    await flushPromises();

    const [, forward, target] = searchCatalog.mock.calls[0] as [
      CatalogSearchParams,
      boolean,
      unknown,
    ];
    expect(forward).toBe(false);
    expect(target).toBeUndefined();
    wrapper.unmount();
  });
});

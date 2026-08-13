import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { CatalogEntry } from "../../lib/api/types";
import type { CatalogSearchParams } from "../../lib/api/catalog";

const { searchCatalog, fetchCatalogDetail, fetchCatalogFamilies, startCatalogDownload } =
  vi.hoisted(() => ({
    searchCatalog: vi.fn(),
    fetchCatalogDetail: vi.fn(),
    fetchCatalogFamilies: vi.fn(),
    startCatalogDownload: vi.fn(),
  }));
const { fetchModelComponents } = vi.hoisted(() => ({
  fetchModelComponents: vi.fn(),
}));
vi.mock("../../lib/api/catalog", () => ({
  searchCatalog,
  fetchCatalogDetail,
  fetchCatalogFamilies,
  startCatalogDownload,
}));
vi.mock("../../lib/api/models", () => ({ fetchModelComponents }));
vi.mock("../../lib/catalogSizes", () => ({
  resolveEntrySize: vi.fn().mockResolvedValue(null),
}));
vi.mock("../../lib/openExternal", () => ({ openExternal: vi.fn() }));

import CatalogTab from "./CatalogTab.vue";
import CatalogDetailDrawer from "./CatalogDetailDrawer.vue";
import { useConnectionStore } from "../../stores/connection";
import { useDownloadsStore } from "../../stores/downloads";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useModelStore } from "../../stores/models";

const PAGE_SIZE = 24;

/** Records instances so tests can walk the sentinel into view. */
class FakeIntersectionObserver {
  static instances: FakeIntersectionObserver[] = [];
  targets: Element[] = [];
  constructor(private callback: IntersectionObserverCallback) {
    FakeIntersectionObserver.instances.push(this);
  }
  observe(el: Element) {
    this.targets.push(el);
  }
  disconnect() {
    this.targets = [];
  }
  intersect(isIntersecting = true) {
    this.callback(
      this.targets.map((target) => ({ isIntersecting, target }) as IntersectionObserverEntry),
      this as unknown as IntersectionObserver,
    );
  }
}

/** Fires "sentinel scrolled into view" on every live observer. */
function scrollToSentinel() {
  for (const observer of FakeIntersectionObserver.instances) observer.intersect();
}

/** Fires "sentinel left the viewport" on every live observer. */
function scrollPastSentinel() {
  for (const observer of FakeIntersectionObserver.instances) observer.intersect(false);
}

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
  FakeIntersectionObserver.instances = [];
  (globalThis as { IntersectionObserver?: unknown }).IntersectionObserver =
    FakeIntersectionObserver;
  fetchCatalogFamilies.mockResolvedValue([]);
  fetchModelComponents.mockResolvedValue({ model: "", components: [] });
  fetchCatalogDetail.mockImplementation((id: string) =>
    Promise.resolve({ ...entry(id, "flux"), id, name: id, installed: true }),
  );
  searchCatalog.mockResolvedValue({ entries: [], page: 1, page_size: PAGE_SIZE, total: 0 });
});

afterEach(() => {
  delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  vi.useRealTimers();
});

describe("CatalogTab media filter under pagination", () => {
  it("shows an installed model ONCE, host-tagged, when a live catalog copy also matches", async () => {
    setActivePinia(createPinia());
    searchCatalog.mockResolvedValue({
      entries: [
        {
          ...entry("flux-dev:q8", "flux"),
          id: "hf:org/repo",
          source_id: "org/repo",
          kind: "lora",
          nsfw: true,
          description: "Rich live catalog metadata.",
        },
      ],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "flux-dev:q8",
            family: "flux",
            size_gb: 12,
            is_loaded: false,
            hf_repo: "org/repo",
            default_steps: 4,
            default_guidance: 1,
            default_width: 1024,
            default_height: 1024,
            description: "",
            downloaded: true,
            hostIds: ["local", "hal9000-7680"],
          },
        ],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    // One card, not the installed row plus the untagged live duplicate.
    const cards = wrapper.findAll("[data-test='catalog-card']");
    const occurrences = wrapper.text().split("flux-dev:q8").length - 1;
    expect(occurrences).toBe(1);
    expect(cards.length).toBeLessThanOrEqual(1);
    // Host chips are the "you have this" indicator.
    const chips = wrapper.findAll("[data-test='installed-host']").map((c) => c.text());
    expect(chips).toContain("This device");
    expect(wrapper.text()).toContain("● installed");
    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe("LoRA");
    expect(wrapper.get('[data-test="model-nsfw-badge"]').text()).toBe("18+ NSFW");
    expect(wrapper.get('[data-test="catalog-description"]').text()).toBe(
      "Rich live catalog metadata.",
    );
  });

  it("does not enrich or deduplicate unrelated models that only share a human name", async () => {
    setActivePinia(createPinia());
    searchCatalog.mockResolvedValue({
      entries: [
        {
          ...entry("Shared title", "flux"),
          id: "hf:other/repo",
          source_id: "other/repo",
          kind: "lora",
          nsfw: true,
          description: "Metadata that belongs only to the other repository.",
        },
      ],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "Shared title",
            family: "flux",
            size_gb: 12,
            is_loaded: false,
            hf_repo: "installed/repo",
            default_steps: 4,
            default_guidance: 1,
            default_width: 1024,
            default_height: 1024,
            description: "",
            downloaded: true,
            hostIds: ["local"],
          },
        ],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    const cards = wrapper.findAll("[data-test='catalog-card']");
    expect(cards).toHaveLength(2);
    expect(cards[0]!.get("[data-test='installed-host']").text()).toBe("This device");
    expect(cards[0]!.get("[data-test='model-kind-badge']").text()).toBe("Checkpoint");
    expect(cards[0]!.find("[data-test='model-nsfw-badge']").exists()).toBe(false);
    expect(cards[0]!.text()).not.toContain("Metadata that belongs only");
    expect(cards[1]!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(cards[1]!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
  });

  it("enriches an installed Civitai id from the exact live id before dedup", async () => {
    setActivePinia(createPinia());
    searchCatalog.mockResolvedValue({
      entries: [
        {
          ...entry("Legacy Adapter", "flux"),
          id: "cv:4242",
          source: "civitai",
          source_id: "4242",
          kind: "lora",
          nsfw: true,
          description: "Rich metadata from the exact live catalog version.",
        },
      ],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "cv:4242",
            display_name: "Legacy Adapter",
            family: "flux",
            size_gb: 0.2,
            is_loaded: false,
            hf_repo: "",
            default_steps: 20,
            default_guidance: 3.5,
            default_width: 1024,
            default_height: 1024,
            description: "",
            downloaded: true,
            kind: null,
            modality: null,
            nsfw: null,
            hostIds: ["local"],
          },
        ],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    const cards = wrapper.findAll("[data-test='catalog-card']");
    expect(cards).toHaveLength(1);
    expect(cards[0]!.text()).toContain("Legacy Adapter");
    expect(cards[0]!.get("[data-test='installed-host']").text()).toBe("This device");
    expect(cards[0]!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(cards[0]!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
    expect(cards[0]!.get("[data-test='catalog-description']").text()).toBe(
      "Rich metadata from the exact live catalog version.",
    );
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
    // Results are exhausted — no dangling infinite-scroll sentinel.
    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(false);
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

  it("bounds the auto-fetch and keeps the scroll sentinel for further pages", async () => {
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
    // More pages exist — scrolling keeps digging.
    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(true);
  });

  it("loads the next page when the end-of-list sentinel scrolls into view", async () => {
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 10_000,
      }),
    );
    const wrapper = await mountTab("all");
    expect(searchCatalog.mock.calls.length).toBe(1);
    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(true);

    scrollToSentinel();
    // Re-firing while the page is already loading must not double-fetch.
    scrollToSentinel();
    // The new rows pushed the sentinel out of view before the page settled.
    scrollPastSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as CatalogSearchParams).page);
    expect(pages).toEqual([1, 2]);
    expect(wrapper.text()).toContain("image-2-0");

    // No manual pagination button remains.
    expect(wrapper.findAll("button").some((b) => b.text().includes("Load more"))).toBe(false);
  });

  it("keeps paginating a merged All-source page that arrives short of PAGE_SIZE", async () => {
    // Under source=All the server splits the page budget across sources, so
    // a merged page is structurally short whenever one source has no rows
    // (e.g. ControlNet: HF contributes zero). The wire `total` is the only
    // honest exhaustion signal — a short page with total > fetched must keep
    // the infinite scroll alive.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1, 12),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 103,
      }),
    );
    const wrapper = await mountTab("all");
    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(true);

    scrollToSentinel();
    scrollPastSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as CatalogSearchParams).page);
    expect(pages).toEqual([1, 2]);
    expect(wrapper.text()).toContain("image-2-0");
  });

  it("falls back to the server-echoed page_size when an older server omits total", async () => {
    // No `total` on the wire and the server clamped the page to 12 rows: a
    // full clamped page still means more results, even though it is short of
    // the client's requested PAGE_SIZE.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1, 12),
        page: params.page ?? 1,
        page_size: 12,
      }),
    );
    const wrapper = await mountTab("all");

    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(true);
  });

  it("stops paginating when the accumulated rows reach the wire total", async () => {
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1, params.page === 1 ? 12 : 5),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 17,
      }),
    );
    const wrapper = await mountTab("all");

    scrollToSentinel();
    scrollPastSentinel();
    await flushPromises();

    // 12 + 5 = 17 = total — the feed is exhausted, the sentinel goes away.
    expect(wrapper.find("[data-test='catalog-scroll-sentinel']").exists()).toBe(false);
  });

  it("keeps fetching (bounded) while the sentinel stays visible after a page lands", async () => {
    // A page whose rows get swallowed by dedup/filtering adds no height, so
    // the observer never re-fires — the chain watcher must keep digging, but
    // only up to the auto-fetch bound. With no leave event the sentinel stays
    // "visible" forever here, so the bound is what terminates the chain.
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 10_000,
      }),
    );
    await mountTab("all");

    scrollToSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as CatalogSearchParams).page);
    // Initial load, the intersection's page, then at most MAX_AUTO_PAGES
    // chained follow-ups.
    expect(pages).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("never hides a different live model that shares an installed model's title", async () => {
    // Same human title, different catalog ids — the installed row's title
    // must not act as a dedup key that suppresses the unrelated model. The
    // exact same version is already collapsed by id.
    searchCatalog.mockResolvedValue({
      entries: [
        { ...entry("Juggernaut XL", "sdxl"), id: "cv:111" },
        { ...entry("Juggernaut XL", "sdxl"), id: "cv:222" },
      ],
      page: 1,
      page_size: PAGE_SIZE,
      total: 2,
    });
    setActivePinia(createPinia());
    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "cv:111",
            family: "sdxl",
            size_gb: 6.9,
            is_loaded: false,
            hf_repo: "",
            default_steps: 25,
            default_guidance: 7,
            default_width: 1024,
            default_height: 1024,
            description: "",
            downloaded: true,
            display_name: "Juggernaut XL",
            hostIds: ["local"],
          },
        ],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    const cards = wrapper.findAll("[data-test='catalog-card']");
    // The installed cv:111 (collapsed with its live copy by id) plus the
    // unrelated cv:222 that happens to share the title.
    expect(cards.length).toBe(2);
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

describe("CatalogTab installed repair routing", () => {
  it("confirms repair against owning hosts only", async () => {
    setActivePinia(createPinia());
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    vi.spyOn(useDownloadsStore(), "subscribe").mockResolvedValue(undefined);
    useHostsStore().extras.push(
      {
        id: "studio-7680",
        label: "Studio GPU",
        url: "http://studio:7680",
        apiKey: "studio-key",
        status: "ready",
        error: null,
        instanceId: null,
      },
      {
        id: "other-7680",
        label: "Other GPU",
        url: "http://other:7680",
        apiKey: "other-key",
        status: "ready",
        error: null,
        instanceId: null,
      },
    );
    const installed = {
      name: "flux-dev:q8",
      family: "flux",
      size_gb: 12,
      is_loaded: false,
      hf_repo: "org/flux-dev",
      default_steps: 4,
      default_guidance: 1,
      default_width: 1024,
      default_height: 1024,
      description: "",
      downloaded: true,
      hostIds: ["local", "studio-7680"],
    };
    const wrapper = mount(CatalogTab, {
      attachTo: document.body,
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [installed],
      },
    });
    await flushPromises();

    await wrapper.get("[data-test='catalog-card']").trigger("click");
    await flushPromises();
    await document.body.querySelector<HTMLButtonElement>("[data-test='drawer-repair']")!.click();
    await flushPromises();

    const dialog = document.body.querySelector<HTMLElement>(
      "[data-test='download-target-dialog']",
    )!;
    expect(dialog.textContent).toContain("Choose where to repair");
    expect(dialog.textContent).toContain("This device");
    expect(dialog.textContent).toContain("Studio GPU");
    expect(dialog.textContent).not.toContain("Other GPU");
    expect(startCatalogDownload).not.toHaveBeenCalled();

    dialog.querySelector<HTMLButtonElement>("[data-test='download-target-studio-7680']")!.click();
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "flux-dev:q8",
      { baseUrl: "http://studio:7680", apiKey: "studio-key" },
      true,
    );
    wrapper.unmount();
  });

  it("does not reroute an offline owner's repair to the local primary", async () => {
    setActivePinia(createPinia());
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    vi.spyOn(useDownloadsStore(), "subscribe").mockResolvedValue(undefined);
    useHostsStore().extras.push({
      id: "offline-7680",
      label: "Offline GPU",
      url: "http://offline:7680",
      apiKey: "offline-key",
      status: "error",
      error: "offline",
      instanceId: null,
    });
    const wrapper = mount(CatalogTab, {
      attachTo: document.body,
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "flux-dev:q8",
            family: "flux",
            size_gb: 12,
            is_loaded: false,
            hf_repo: "org/flux-dev",
            default_steps: 4,
            default_guidance: 1,
            default_width: 1024,
            default_height: 1024,
            description: "",
            downloaded: true,
            hostIds: ["offline-7680"],
          },
        ],
      },
    });
    await flushPromises();
    await wrapper.get("[data-test='catalog-card']").trigger("click");
    await flushPromises();
    document.body.querySelector<HTMLButtonElement>("[data-test='drawer-repair']")!.click();
    await flushPromises();

    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(document.body.textContent).not.toContain("Choose where to repair");
    wrapper.unmount();
  });
});

describe("CatalogTab install on a machine that is missing the model", () => {
  /** Local primary + two ready remotes, all with their inventory already read. */
  function threeReadyHosts() {
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    vi.spyOn(useDownloadsStore(), "subscribe").mockResolvedValue(undefined);
    useHostsStore().extras.push(
      {
        id: "studio-7680",
        label: "Studio GPU",
        url: "http://studio:7680",
        apiKey: "studio-key",
        status: "ready",
        error: null,
        instanceId: null,
      },
      {
        id: "other-7680",
        label: "Other GPU",
        url: "http://other:7680",
        apiKey: "other-key",
        status: "ready",
        error: null,
        instanceId: null,
      },
    );
    const hostModels = useHostModelsStore();
    for (const id of ["local", "studio-7680", "other-7680"]) {
      hostModels.byHost[id] = { entries: [], fetchedAt: Date.now(), error: null };
    }
  }

  const videoModel = {
    name: "ltx2-distilled",
    family: "ltx2",
    size_gb: 24,
    is_loaded: false,
    hf_repo: "Lightricks/LTX-2",
    default_steps: 8,
    default_guidance: 1,
    default_width: 1216,
    default_height: 704,
    description: "",
    downloaded: true,
  };

  it("offers an install for a model only the remote hosts have", async () => {
    setActivePinia(createPinia());
    threeReadyHosts();
    const wrapper = mount(CatalogTab, {
      attachTo: document.body,
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [{ ...videoModel, hostIds: ["studio-7680"] }],
      },
    });
    await flushPromises();

    // Installed elsewhere, so the row keeps its marker — and still offers the pull.
    expect(wrapper.text()).toContain("● installed");
    await wrapper.get("[data-test='pull']").trigger("click");
    await flushPromises();

    const dialog = document.body.querySelector<HTMLElement>(
      "[data-test='download-target-dialog']",
    )!;
    expect(dialog.textContent).toContain("Choose where to install");
    // The two machines that lack it are installs; the owner is only a repair.
    expect(dialog.querySelector("[data-test='download-target-local']")?.textContent).toContain(
      "Install",
    );
    expect(dialog.querySelector("[data-test='download-target-other-7680']")?.textContent).toContain(
      "Install",
    );
    expect(
      dialog.querySelector("[data-test='download-target-studio-7680']")?.textContent,
    ).toContain("Already installed");

    dialog.querySelector<HTMLButtonElement>("[data-test='download-target-local']")!.click();
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith(
      "ltx2-distilled",
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
      false,
    );
    wrapper.unmount();
  });

  it("stops offering an install once every machine has the model", async () => {
    setActivePinia(createPinia());
    threeReadyHosts();
    const wrapper = mount(CatalogTab, {
      attachTo: document.body,
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [{ ...videoModel, hostIds: ["local", "studio-7680", "other-7680"] }],
      },
    });
    await flushPromises();

    expect(wrapper.text()).toContain("● installed");
    expect(wrapper.find("[data-test='pull']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("never treats a machine whose inventory is unread as missing the model", async () => {
    setActivePinia(createPinia());
    threeReadyHosts();
    // Other GPU connected but its /api/models has not landed yet.
    delete useHostModelsStore().byHost["other-7680"];
    const wrapper = mount(CatalogTab, {
      attachTo: document.body,
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [{ ...videoModel, hostIds: ["local", "studio-7680"] }],
      },
    });
    await flushPromises();

    expect(wrapper.find("[data-test='pull']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("credits the browsing host for a live row it reports as installed", async () => {
    // A live catalog row carries no hostIds — its `installed` flag is only the
    // browsed host's answer (LoRAs and ControlNets never reach the merged
    // generation-model shelf). With one machine that is nothing left to pull.
    setActivePinia(createPinia());
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    useHostModelsStore().byHost["local"] = {
      entries: [],
      fetchedAt: Date.now(),
      error: null,
    };
    searchCatalog.mockResolvedValue({
      entries: [{ ...entry("portrait-lora", "flux"), kind: "lora", installed: true }],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });

    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "grid" as const },
    });
    await flushPromises();

    expect(wrapper.text()).toContain("● installed");
    expect(wrapper.find("[data-test='pull']").exists()).toBe(false);
    wrapper.unmount();
  });
});

describe("CatalogTab Discover source chips", () => {
  it("offers exactly All / HuggingFace / Civitai — Installed is now a segment", async () => {
    setActivePinia(createPinia());
    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "grid" as const },
      global: { plugins: [] },
    });
    await flushPromises();

    const chips = wrapper.get("[data-test='catalog-source-chips']").findAll("button");
    expect(chips.map((c) => c.text())).toEqual(["All", "HuggingFace", "Civitai"]);
    expect(chips[0]!.attributes("aria-pressed")).toBe("true");
  });
});

describe("CatalogTab kind and sort filters", () => {
  it("renders the shared kind chips and sort options", async () => {
    const wrapper = await mountTab();

    const chips = wrapper.get("[data-test='catalog-kind-chips']").findAll("button");
    expect(chips.map((c) => c.text())).toEqual([
      "All",
      "Models",
      "LoRAs",
      "CLIP",
      "Text encoders",
      "VAEs",
      "Tokenizers",
      "ControlNet",
    ]);
    expect(chips[0]!.attributes("aria-pressed")).toBe("true");

    const options = wrapper.get("[data-test='catalog-sort']").findAll("option");
    expect(options.map((o) => o.text())).toEqual(["Downloads", "Rating", "Recent"]);
    expect(wrapper.get('input[type="checkbox"]').element.closest("label")?.textContent).toContain(
      "Include NSFW",
    );
  });

  it("omits kind and sort from the search at their defaults", async () => {
    await mountTab();
    const params = searchCatalog.mock.calls[0]![0] as CatalogSearchParams & { sort?: string };
    expect(params.kind).toBeUndefined();
    expect(params.sort).toBeUndefined();
  });

  it("resets to page 1 with the kind on the wire when a kind chip is toggled", async () => {
    searchCatalog.mockImplementation((params: CatalogSearchParams) =>
      Promise.resolve({
        entries: imagePage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: PAGE_SIZE,
        total: 10_000,
      }),
    );
    const wrapper = await mountTab();
    scrollToSentinel();
    scrollPastSentinel();
    await flushPromises();
    expect((searchCatalog.mock.calls.at(-1)![0] as CatalogSearchParams).page).toBe(2);

    vi.useFakeTimers();
    const lora = wrapper
      .get("[data-test='catalog-kind-chips']")
      .findAll("button")
      .find((c) => c.text() === "LoRAs")!;
    await lora.trigger("click");
    await vi.advanceTimersByTimeAsync(400);

    const params = searchCatalog.mock.calls.at(-1)![0] as CatalogSearchParams & { sort?: string };
    expect(params.kind).toBe("lora");
    expect(params.page).toBe(1);
    expect(lora.attributes("aria-pressed")).toBe("true");
  });

  it("resets to page 1 with the sort on the wire when the sort changes", async () => {
    const wrapper = await mountTab();
    vi.useFakeTimers();
    await wrapper.get("[data-test='catalog-sort']").setValue("rating");
    await vi.advanceTimersByTimeAsync(400);

    const params = searchCatalog.mock.calls.at(-1)![0] as CatalogSearchParams & { sort?: string };
    expect(params.sort).toBe("rating");
    expect(params.page).toBe(1);
  });

  it("hides manifest and installed rows under a non-checkpoint kind", async () => {
    setActivePinia(createPinia());
    useModelStore().all = [
      {
        name: "flux-dev:q8",
        family: "flux",
        size_gb: 12,
        is_loaded: false,
        hf_repo: "org/flux-dev",
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "",
        downloaded: false,
      },
    ];
    searchCatalog.mockResolvedValue({
      entries: [{ ...entry("Detail LoRA", "flux"), kind: "lora" }],
      page: 1,
      page_size: PAGE_SIZE,
      total: 1,
    });
    const wrapper = mount(CatalogTab, {
      props: {
        query: "",
        layout: "grid" as const,
        installedEntries: [
          {
            name: "sd15:q8",
            family: "sd15",
            size_gb: 2,
            is_loaded: false,
            hf_repo: "org/sd15",
            default_steps: 25,
            default_guidance: 7,
            default_width: 512,
            default_height: 512,
            description: "",
            downloaded: true,
            hostIds: ["local"],
          },
        ],
      },
      global: { plugins: [] },
    });
    await flushPromises();
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).toContain("sd15:q8");

    vi.useFakeTimers();
    const lora = wrapper
      .get("[data-test='catalog-kind-chips']")
      .findAll("button")
      .find((c) => c.text() === "LoRAs")!;
    await lora.trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    vi.useRealTimers();
    await flushPromises();

    expect(wrapper.text()).toContain("Detail LoRA");
    expect(wrapper.text()).not.toContain("flux-dev:q8");
    expect(wrapper.text()).not.toContain("sd15:q8");
  });
});

describe("CatalogTab variant chips", () => {
  it("surfaces a manifest model's quant variants in the detail drawer", async () => {
    setActivePinia(createPinia());
    // Two undrawn manifest variants of the same base model → catalog cards.
    useModelStore().all = [
      {
        name: "flux-dev:q4",
        family: "flux",
        size_gb: 6,
        is_loaded: false,
        hf_repo: "org/flux-dev",
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "",
        downloaded: false,
      },
      {
        name: "flux-dev:q8",
        family: "flux",
        size_gb: 12,
        is_loaded: false,
        hf_repo: "org/flux-dev",
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "",
        downloaded: false,
      },
    ];
    const wrapper = mount(CatalogTab, {
      props: { query: "", layout: "table" as const },
      global: { plugins: [] },
    });
    await flushPromises();

    const rows = wrapper.findAll("[data-test='catalog-table-row']");
    const q4Row = rows.find((row) => row.text().includes("flux-dev:q4"))!;
    const q8Row = rows.find((row) => row.text().includes("flux-dev:q8"))!;
    await q4Row.trigger("click");
    await flushPromises();
    expect(q4Row.attributes("data-selected")).toBe("true");

    const chips = wrapper
      .get("[data-test='drawer-variants']")
      .findAll("[data-test='variant-chip']");
    const labels = chips.map((c) => c.text());
    expect(labels.some((t) => t.includes("q4"))).toBe(true);
    expect(labels.some((t) => t.includes("q8"))).toBe(true);
    // Each variant advertises its footprint so the choice is informed.
    expect(labels.some((t) => t.includes("6.0 GB"))).toBe(true);

    const q8 = chips.find((chip) => chip.text().includes("q8"))!;
    await q8.trigger("click");
    await flushPromises();

    // The chosen variant becomes the drawer's actual entry; every detail and
    // repair field now derives from q8 rather than changing only the Pull id,
    // and the selected-list highlight follows that same authority.
    expect(wrapper.getComponent(CatalogDetailDrawer).props("entry").id).toBe("flux-dev:q8");
    expect(q8.attributes("aria-pressed")).toBe("true");
    expect(q4Row.attributes("data-selected")).toBeUndefined();
    expect(q8Row.attributes("data-selected")).toBe("true");
  });

  it("keeps a filtered non-installed sibling as a full manifest Pull entry", async () => {
    setActivePinia(createPinia());
    useModelStore().all = [
      {
        name: "flux-dev:q4",
        family: "flux",
        size_gb: 6,
        is_loaded: false,
        hf_repo: "org/flux-dev",
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "",
        downloaded: false,
      },
      {
        name: "flux-dev:q8",
        family: "flux",
        size_gb: 12,
        remaining_download_bytes: 20_000_000_000,
        is_loaded: false,
        hf_repo: "org/flux-dev",
        default_steps: 4,
        default_guidance: 1,
        default_width: 1024,
        default_height: 1024,
        description: "",
        downloaded: false,
      },
    ];
    const wrapper = mount(CatalogTab, {
      props: { query: "q4", layout: "table" as const },
      global: { plugins: [] },
    });
    await flushPromises();

    await wrapper.get("[data-test='catalog-table-row']").trigger("click");
    await flushPromises();
    const q8 = wrapper
      .get("[data-test='drawer-variants']")
      .findAll("[data-test='variant-chip']")
      .find((chip) => chip.text().includes("q8"))!;
    await q8.trigger("click");
    await flushPromises();

    const chosen = wrapper.getComponent(CatalogDetailDrawer).props("entry") as CatalogEntry;
    expect(chosen.id).toBe("flux-dev:q8");
    expect(chosen.installed).toBe(false);
    expect(chosen.size_bytes).toBe(12_000_000_000);
    expect(chosen.companion_details).toEqual([
      { name: "shared runtime components", size_bytes: 8_000_000_000 },
    ]);
    expect(wrapper.getComponent(CatalogDetailDrawer).props("action")).toBe("Pull");
  });

  it("preserves remote ownership when a filtered installed sibling is selected", async () => {
    setActivePinia(createPinia());
    const q4 = {
      name: "flux-dev:q4",
      family: "flux",
      size_gb: 6,
      is_loaded: false,
      hf_repo: "org/flux-dev",
      default_steps: 4,
      default_guidance: 1,
      default_width: 1024,
      default_height: 1024,
      description: "",
      downloaded: false,
    };
    const q8 = {
      ...q4,
      name: "flux-dev:q8",
      size_gb: 12,
      downloaded: true,
      hostIds: ["render-box"],
    };
    useModelStore().all = [q4, q8];
    const wrapper = mount(CatalogTab, {
      props: {
        query: "q4",
        layout: "table" as const,
        installedEntries: [q8],
      },
      global: { plugins: [] },
    });
    await flushPromises();

    await wrapper.get("[data-test='catalog-table-row']").trigger("click");
    await flushPromises();
    const q8Chip = wrapper
      .get("[data-test='drawer-variants']")
      .findAll("[data-test='variant-chip']")
      .find((chip) => chip.text().includes("q8"))!;
    await q8Chip.trigger("click");
    await flushPromises();

    const chosen = wrapper.getComponent(CatalogDetailDrawer).props("entry") as CatalogEntry & {
      hostIds?: string[];
    };
    expect(chosen.id).toBe("flux-dev:q8");
    expect(chosen.installed).toBe(true);
    expect(chosen.hostIds).toEqual(["render-box"]);
  });
});

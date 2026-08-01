import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia } from "pinia";
import { defineComponent, h, KeepAlive, nextTick, ref } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  CatalogEntry,
  CatalogSearchResponse,
  DownloadEvent,
  ModelEntry,
} from "../lib/api/types";
import type { ApiTarget } from "../lib/api/client";
import type { MobileHost } from "./hosts";

const {
  apiFetchTo,
  fetchCatalogDetail,
  fetchCatalogFamilies,
  fetchModelComponents,
  loadModel,
  removeModel,
  searchCatalog,
  sseStream,
  startCatalogDownload,
  unloadModel,
  streams,
} = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  fetchCatalogDetail: vi.fn(),
  fetchCatalogFamilies: vi.fn(),
  fetchModelComponents: vi.fn(),
  loadModel: vi.fn(),
  removeModel: vi.fn(),
  searchCatalog: vi.fn(),
  sseStream: vi.fn(),
  startCatalogDownload: vi.fn(),
  unloadModel: vi.fn(),
  streams: [] as Array<{
    target: ApiTarget;
    signal: AbortSignal;
    onOpen?: () => void;
    onOpenError?: (error: Error) => void;
    onClose?: (error: Error | null) => void;
    onEvent: (event: string, data: string) => void;
  }>,
}));

vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
}));

vi.mock("../lib/api/catalog", () => ({
  fetchCatalogDetail,
  fetchCatalogFamilies,
  searchCatalog,
  startCatalogDownload,
}));

vi.mock("../lib/api/models", () => ({
  fetchModelComponents,
  loadModel,
  removeModel,
  unloadModel,
}));

vi.mock("../lib/api/sse", () => ({ sseStream }));

import MobileCatalogView from "./MobileCatalogView.vue";

const studio: MobileHost = {
  id: "studio-id",
  name: "Studio",
  baseUrl: "https://studio.tailnet.ts.net:7680",
  apiKey: "studio-key",
  hostname: "studio",
  version: "0.18.0",
  online: true,
};

const renderBox: MobileHost = {
  id: "render-id",
  name: "Render Box",
  baseUrl: "https://render.tailnet.ts.net:7680",
  apiKey: "render-key",
  hostname: "render",
  version: "0.18.0",
  online: true,
};

const targets = {
  studio: { baseUrl: studio.baseUrl, apiKey: studio.apiKey },
  render: { baseUrl: renderBox.baseUrl, apiKey: renderBox.apiKey },
};

function model(
  name: string,
  family: string,
  downloaded: boolean,
  overrides: Partial<ModelEntry> = {},
): ModelEntry {
  return {
    name,
    family,
    size_gb: 12,
    is_loaded: false,
    hf_repo: "mold/safe-model",
    default_steps: 30,
    default_guidance: 4,
    default_width: 1024,
    default_height: 1024,
    description: "A test model",
    downloaded,
    ...overrides,
  };
}

function entry(name: string, overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    id: `hf:${name}`,
    source: "hf",
    source_id: `example/${name}`,
    name,
    family: "flux",
    kind: "checkpoint",
    nsfw: false,
    installed: false,
    size_bytes: 8_000_000_000,
    thumbnail_url: null,
    ...overrides,
  };
}

function searchResponse(entries: CatalogEntry[]): CatalogSearchResponse {
  return { entries, page: 1, page_size: 24, total: entries.length };
}

/** Records instances so tests can walk the end-of-list sentinel into view. */
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

/** Values offered by the Family filter, starting with the "All families" row. */
function familyOptionValues(view: VueWrapper): string[] {
  return view
    .get("[data-test='mobile-catalog-family']")
    .findAll("option")
    .map((option) => option.attributes("value") ?? "");
}

function jsonResponse<T>(value: T): Response {
  return { json: () => Promise.resolve(value) } as Response;
}

function emptyDownloadSnapshot(): DownloadEvent {
  return {
    type: "snapshot",
    listing: { active_jobs: [], queued: [], history: [] },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

let wrapper: VueWrapper | null = null;

function mountCatalog(
  selectedHostId = studio.id,
  hosts: MobileHost[] = [studio, renderBox],
): VueWrapper {
  return mount(MobileCatalogView, {
    attachTo: document.body,
    props: { hosts, selectedHostId },
    global: { plugins: [createPinia()] },
  });
}

beforeEach(() => {
  streams.length = 0;
  vi.clearAllMocks();
  FakeIntersectionObserver.instances = [];
  (globalThis as { IntersectionObserver?: unknown }).IntersectionObserver =
    FakeIntersectionObserver;
  fetchCatalogFamilies.mockResolvedValue(["flux", "ltx2"]);
  fetchCatalogDetail.mockImplementation((id: string) =>
    Promise.resolve({ ...entry("detail"), id, name: "Catalog model", description: "Full detail" }),
  );
  fetchModelComponents.mockResolvedValue({ model: "installed:q8", components: [] });
  loadModel.mockResolvedValue(new Response(null, { status: 204 }));
  unloadModel.mockResolvedValue(new Response(null, { status: 204 }));
  removeModel.mockResolvedValue({ removed: [], kept: [], freed_bytes: 1_000_000_000 });
  startCatalogDownload.mockResolvedValue("download-1");
  searchCatalog.mockResolvedValue(searchResponse([entry("Catalog model")]));
  apiFetchTo.mockImplementation((target: ApiTarget, path: string) => {
    if (path === "/api/models") {
      return Promise.resolve(
        jsonResponse(
          target.baseUrl === studio.baseUrl
            ? [
                model("installed:q8", "flux", true),
                model("safe-variant:q4", "ltx2", false, {
                  size_gb: 4,
                  remaining_download_bytes: 6_000_000_000,
                }),
              ]
            : [model("installed:q8", "flux", true)],
        ),
      );
    }
    return Promise.resolve(new Response(null, { status: 204 }));
  });
  sseStream.mockImplementation(
    (
      _path: string,
      options: {
        target: ApiTarget;
        signal: AbortSignal;
        onOpen?: () => void;
        onOpenError?: (error: Error) => void;
        onClose?: (error: Error | null) => void;
        onEvent: (event: string, data: string) => void;
      },
    ) => {
      streams.push(options);
      options.onOpen?.();
      options.onEvent("download", JSON.stringify(emptyDownloadSnapshot()));
      return Promise.resolve();
    },
  );
});

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
  document.body.innerHTML = "";
  delete (globalThis as Partial<typeof globalThis>).IntersectionObserver;
  vi.useRealTimers();
});

describe("MobileCatalogView", () => {
  it("loads the next page when the end-of-list sentinel scrolls into view", async () => {
    const fullPage = (page: number) =>
      Array.from({ length: 24 }, (_, i) => entry(`catalog-${page}-${i}`));
    searchCatalog.mockImplementation((params: { page?: number }) =>
      Promise.resolve({
        entries: fullPage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: 24,
        total: 10_000,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    expect(searchCatalog.mock.calls.length).toBe(1);
    expect(wrapper.find("[data-test='mobile-catalog-sentinel']").exists()).toBe(true);
    // No manual pagination button remains.
    expect(wrapper.findAll("button").some((b) => b.text().includes("Load more"))).toBe(false);

    scrollToSentinel();
    // Re-firing while the page is already loading must not double-fetch.
    scrollToSentinel();
    // The new rows pushed the sentinel out of view before the page settled.
    scrollPastSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as { page?: number }).page);
    expect(pages).toEqual([1, 2]);
    expect(wrapper.text()).toContain("catalog-2-0");
  });

  it("keeps paginating a merged All-source page that arrives short of the page size", async () => {
    // Under source=All the server splits the page budget across sources, so
    // a merged page is structurally short whenever one source has no rows
    // (e.g. ControlNet: HF contributes zero). The wire `total` is the only
    // honest exhaustion signal — a short page with total > fetched must keep
    // the infinite scroll alive.
    const halfPage = (page: number) =>
      Array.from({ length: 12 }, (_, i) => entry(`catalog-${page}-${i}`));
    searchCatalog.mockImplementation((params: { page?: number }) =>
      Promise.resolve({
        entries: halfPage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: 24,
        total: 103,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-catalog-sentinel']").exists()).toBe(true);

    scrollToSentinel();
    scrollPastSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as { page?: number }).page);
    expect(pages).toEqual([1, 2]);
    expect(wrapper.text()).toContain("catalog-2-0");
  });

  it("falls back to the server-echoed page_size when an older server omits total", async () => {
    // No `total` on the wire and the server clamped the page to 12 rows: a
    // full clamped page still means more results, even though it is short of
    // the client's requested page size.
    searchCatalog.mockImplementation((params: { page?: number }) =>
      Promise.resolve({
        entries: Array.from({ length: 12 }, (_, i) => entry(`catalog-${params.page ?? 1}-${i}`)),
        page: params.page ?? 1,
        page_size: 12,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    expect(wrapper.find("[data-test='mobile-catalog-sentinel']").exists()).toBe(true);
  });

  it("stops paginating when the accumulated rows reach the wire total", async () => {
    searchCatalog.mockImplementation((params: { page?: number }) =>
      Promise.resolve({
        entries: Array.from({ length: (params.page ?? 1) === 1 ? 12 : 5 }, (_, i) =>
          entry(`catalog-${params.page ?? 1}-${i}`),
        ),
        page: params.page ?? 1,
        page_size: 24,
        total: 17,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    scrollToSentinel();
    scrollPastSentinel();
    await flushPromises();

    // 12 + 5 = 17 = total — the feed is exhausted, the sentinel goes away.
    expect(wrapper.find("[data-test='mobile-catalog-sentinel']").exists()).toBe(false);
  });

  it("keeps fetching (bounded) while the sentinel stays visible after a page lands", async () => {
    // Height-less pages (dedup/filter-swallowed) emit no new intersection —
    // the chain watcher keeps digging up to the auto-fetch bound.
    const fullPage = (page: number) =>
      Array.from({ length: 24 }, (_, i) => entry(`catalog-${page}-${i}`));
    searchCatalog.mockImplementation((params: { page?: number }) =>
      Promise.resolve({
        entries: fullPage(params.page ?? 1),
        page: params.page ?? 1,
        page_size: 24,
        total: 10_000,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    scrollToSentinel();
    await flushPromises();

    const pages = searchCatalog.mock.calls.map((c) => (c[0] as { page?: number }).page);
    // Initial load, the intersection's page, then at most MAX_AUTO_PAGES
    // chained follow-ups.
    expect(pages).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("repoints the detail pull to the selected manifest variant", async () => {
    searchCatalog.mockResolvedValue(searchResponse([]));
    apiFetchTo.mockImplementation((_target: ApiTarget, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve(
          jsonResponse([
            model("flux-dev:q4", "flux", false, { size_gb: 6 }),
            model("flux-dev:q8", "flux", false, { size_gb: 12 }),
          ]),
        );
      }
      return Promise.resolve(new Response(null, { status: 204 }));
    });
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const q4 = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("flux-dev:q4"))!;
    await q4.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const chips = [...document.querySelectorAll<HTMLButtonElement>("[data-test='variant-chip']")];
    expect(chips.map((chip) => chip.textContent)).toEqual([
      expect.stringContaining("q4"),
      expect.stringContaining("q8"),
    ]);
    chips[1]!.click();
    await flushPromises();
    document.querySelector<HTMLButtonElement>(".mobile-catalog-detail-action button")!.click();
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith("flux-dev:q8", targets.studio, false);
  });

  it("keeps installed siblings in the variant choices", async () => {
    searchCatalog.mockResolvedValue(searchResponse([]));
    apiFetchTo.mockImplementation((_target: ApiTarget, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve(
          jsonResponse([
            model("flux-dev:q4", "flux", true, { size_gb: 6 }),
            model("flux-dev:q8", "flux", false, { size_gb: 12 }),
          ]),
        );
      }
      return Promise.resolve(new Response(null, { status: 204 }));
    });
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const q8 = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("flux-dev:q8"))!;
    await q8.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const labels = [
      ...document.querySelectorAll<HTMLButtonElement>("[data-test='variant-chip']"),
    ].map((chip) => chip.textContent);
    expect(labels).toEqual([expect.stringContaining("q4"), expect.stringContaining("q8")]);
  });

  it("searches the selected remote host and merges installed models with host labels", async () => {
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Aggregate repo", {
          source_id: "mold/safe-model",
          bundling: "separated",
          size_bytes: 250_000_000_000,
        }),
        entry("Catalog model"),
      ]),
    );
    wrapper = mountCatalog();
    await flushPromises();

    expect(searchCatalog).toHaveBeenCalledWith(
      {
        q: undefined,
        family: undefined,
        source: undefined,
        include_nsfw: false,
        page: 1,
        page_size: 24,
      },
      false,
      targets.studio,
    );
    expect(fetchCatalogFamilies).toHaveBeenCalledWith(false, targets.studio);

    const text = wrapper.text();
    expect(text).toContain("installed:q8");
    expect(text).toContain("Studio");
    expect(text).toContain("Render Box");
    expect(text).toContain("safe-variant:q4");
    expect(text).toContain("Catalog model");
    expect(text).not.toContain("Aggregate repo");
    expect(wrapper.findAll("[data-test='mobile-catalog-card']")).toHaveLength(3);
  });

  it("debounces search and ignores a late response from the previous host", async () => {
    vi.useFakeTimers();
    const pending: Array<(response: CatalogSearchResponse) => void> = [];
    searchCatalog.mockImplementation(
      () =>
        new Promise<CatalogSearchResponse>((resolve) => {
          pending.push(resolve);
        }),
    );
    wrapper = mountCatalog();
    await vi.waitFor(() => expect(searchCatalog).toHaveBeenCalledTimes(1));

    await wrapper.get("[data-test='mobile-catalog-search']").setValue("portrait");
    await vi.advanceTimersByTimeAsync(399);
    expect(searchCatalog).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(searchCatalog).toHaveBeenCalledTimes(2);

    await wrapper.setProps({ selectedHostId: renderBox.id });
    await vi.waitFor(() => expect(searchCatalog).toHaveBeenCalledTimes(3));
    pending[2]!(searchResponse([entry("Render result")]));
    await flushPromises();
    pending[0]!(searchResponse([entry("Old Studio result")]));
    pending[1]!(searchResponse([entry("Old query result")]));
    await flushPromises();

    expect(wrapper.text()).toContain("Render result");
    expect(wrapper.text()).not.toContain("Old Studio result");
    expect(wrapper.text()).not.toContain("Old query result");
  });

  it("chooses a remote target for a pull and opens detail on the selected host", async () => {
    wrapper = mountCatalog();
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"));
    expect(catalogCard).toBeDefined();
    (catalogCard!.get(".mobile-catalog-card-open").element as HTMLButtonElement).focus();
    await catalogCard!.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    expect(fetchCatalogDetail).toHaveBeenCalledWith("hf:Catalog model", false, targets.studio);
    expect(document.querySelector("[data-test='mobile-catalog-detail']")?.textContent).toContain(
      "Full detail",
    );
    expect(wrapper.get(".mobile-catalog").attributes()).toHaveProperty("inert");

    const detailClose = document.querySelector<HTMLButtonElement>(
      "[aria-label='Close model details']",
    )!;
    const detailAction = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-detail-action button",
    )!;
    expect(document.activeElement).toBe(detailClose);
    window.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Tab",
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(detailAction);

    detailAction.click();
    await flushPromises();
    const detailDialog = document.querySelector<HTMLElement>(
      "[data-test='mobile-catalog-detail']",
    )!;
    expect(document.querySelector("[data-test='mobile-catalog-target-sheet']")).not.toBeNull();
    expect(detailDialog.hasAttribute("inert")).toBe(true);

    const targetClose = document.querySelector<HTMLButtonElement>(
      "[aria-label='Close host picker']",
    )!;
    const targetButtons = document.querySelectorAll<HTMLButtonElement>(
      ".mobile-catalog-target-list button",
    );
    expect(targetButtons).toHaveLength(2);
    expect(document.activeElement).toBe(targetClose);
    window.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Tab",
        shiftKey: true,
        bubbles: true,
        cancelable: true,
      }),
    );
    expect(document.activeElement).toBe(targetButtons[1]);
    window.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }),
    );
    expect(document.activeElement).toBe(targetClose);
    targetButtons[1]!.click();
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith("hf:Catalog model", targets.render, false);
    expect(document.body.textContent).toContain("Pulling Catalog model on Render Box");
    expect(detailDialog.hasAttribute("inert")).toBe(false);
    expect(wrapper.get(".mobile-catalog").attributes()).toHaveProperty("inert");

    detailClose.click();
    await flushPromises();
    expect(wrapper.get(".mobile-catalog").attributes()).not.toHaveProperty("inert");
    expect(document.activeElement).toBe(catalogCard!.get(".mobile-catalog-card-open").element);
  });

  it("suspends global dialog handling while the kept-alive catalog is off-tab", async () => {
    const active = ref(true);
    const OtherView = defineComponent({
      setup: () => () => h("section", { "data-test": "other-view" }, "Other view"),
    });
    const Harness = defineComponent({
      setup: () => () =>
        h(KeepAlive, null, {
          default: () =>
            active.value
              ? h(MobileCatalogView, { hosts: [studio, renderBox], selectedHostId: studio.id })
              : h(OtherView),
        }),
    });

    wrapper = mount(Harness, { attachTo: document.body });
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"));
    expect(catalogCard).toBeDefined();
    await catalogCard!.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const catalog = wrapper.get(".mobile-catalog").element as HTMLElement;
    expect(catalog.hasAttribute("inert")).toBe(true);

    active.value = false;
    await nextTick();
    expect(catalog.hasAttribute("inert")).toBe(false);
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));

    active.value = true;
    await nextTick();
    await nextTick();
    expect(document.querySelector("[data-test='mobile-catalog-detail']")).not.toBeNull();
    expect(catalog.hasAttribute("inert")).toBe(true);
  });

  it("waits for the opening download snapshot before posting a pull", async () => {
    sseStream.mockImplementation(
      (
        _path: string,
        options: {
          target: ApiTarget;
          signal: AbortSignal;
          onOpen?: () => void;
          onOpenError?: (error: Error) => void;
          onClose?: (error: Error | null) => void;
          onEvent: (event: string, data: string) => void;
        },
      ) => {
        streams.push(options);
        return Promise.resolve();
      },
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await catalogCard.get(".mobile-catalog-pull").trigger("click");
    await flushPromises();

    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='mobile-catalog-action-status']").text()).toContain(
      "Connecting to downloads on Studio",
    );

    streams[0]!.onOpen?.();
    await flushPromises();
    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(wrapper.get("[data-test='mobile-catalog-action-status']").text()).toContain(
      "Connecting to downloads on Studio",
    );

    streams[0]!.onEvent("download", JSON.stringify(emptyDownloadSnapshot()));
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith("hf:Catalog model", targets.studio, false);
    expect(wrapper.get("[data-test='mobile-catalog-action-status']").text()).toContain(
      "Pulling Catalog model on Studio",
    );
  });

  it("keeps a pull button busy until SSE adopts the job and then shows live progress", async () => {
    const start = deferred<string | null>();
    startCatalogDownload.mockReturnValue(start.promise);
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("repo", {
          id: "hf:owner/repo",
          source_id: "owner/repo",
        }),
      ]),
    );
    sseStream.mockImplementation(
      (
        _path: string,
        options: {
          target: ApiTarget;
          signal: AbortSignal;
          onOpen?: () => void;
          onOpenError?: (error: Error) => void;
          onClose?: (error: Error | null) => void;
          onEvent: (event: string, data: string) => void;
        },
      ) => {
        streams.push(options);
        return Promise.resolve();
      },
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("repo"))!;
    const pullButton = catalogCard.get(".mobile-catalog-pull");
    await pullButton.trigger("click");
    expect(pullButton.text()).toBe("Connecting…");
    expect((pullButton.element as HTMLButtonElement).disabled).toBe(true);

    streams[0]!.onOpen?.();
    await flushPromises();
    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(pullButton.text()).toBe("Connecting…");

    streams[0]!.onEvent("download", JSON.stringify(emptyDownloadSnapshot()));
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
    expect(pullButton.text()).toBe("Starting…");

    // Even a programmatic second action is ignored while the POST is in flight.
    (pullButton.element as HTMLButtonElement).click();
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);

    start.resolve("download-1");
    await flushPromises();
    expect(pullButton.text()).toBe("Starting…");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "download-1",
        model: "owner/repo",
        position: 0,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(pullButton.text()).toBe("Queued");
    expect(wrapper.findAll("[data-test='mobile-catalog-download']")).toHaveLength(1);

    // Duplicate stream deltas must not add another download row or action.
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "download-1",
        model: "owner/repo",
        position: 0,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(wrapper.findAll("[data-test='mobile-catalog-download']")).toHaveLength(1);

    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "started", id: "download-1", files_total: 2, bytes_total: 3 }),
    );
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "progress",
        id: "download-1",
        files_done: 0,
        bytes_done: 1,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(pullButton.text()).toBe("Pulling 33%");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "progress",
        id: "download-1",
        files_done: 2,
        bytes_done: 4,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(pullButton.text()).toBe("Pulling 100%");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "job_failed", id: "download-1", error: "disk full" }),
    );
    await flushPromises();
    expect(pullButton.text()).toContain("Pull");
    expect((pullButton.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("clears a pending pull when the host returns no job id despite an HF model mismatch", async () => {
    startCatalogDownload.mockResolvedValue(null);
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Friendly repo name", {
          id: "hf:owner/repo",
          source_id: "owner/repo",
        }),
      ]),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const pullButton = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Friendly repo name"))!
      .get(".mobile-catalog-pull");
    await pullButton.trigger("click");
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith("hf:owner/repo", targets.studio, false);
    expect(pullButton.text()).not.toBe("Starting…");
    expect((pullButton.element as HTMLButtonElement).disabled).toBe(false);
    expect(wrapper.get("[data-test='mobile-catalog-action-status']").text()).toContain(
      "Studio did not return a download job",
    );

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "canonical-job",
        model: "canonical-manifest-name",
        position: 0,
      } satisfies DownloadEvent),
    );
    await flushPromises();

    expect(pullButton.text()).not.toBe("Starting…");
    expect((pullButton.element as HTMLButtonElement).disabled).toBe(false);

    await pullButton.trigger("click");
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledTimes(2);
  });

  it("does not POST when an asynchronous opening snapshot adopts an existing pull", async () => {
    sseStream.mockImplementation(
      (
        _path: string,
        options: {
          target: ApiTarget;
          signal: AbortSignal;
          onOpen?: () => void;
          onOpenError?: (error: Error) => void;
          onClose?: (error: Error | null) => void;
          onEvent: (event: string, data: string) => void;
        },
      ) => {
        streams.push(options);
        return Promise.resolve();
      },
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const pullButton = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!
      .get(".mobile-catalog-pull");
    await pullButton.trigger("click");
    streams[0]!.onOpen?.();
    // Drain the transport-open continuation before delivering the opening
    // snapshot. Resolving readiness from `onOpen` would POST at this point.
    await flushPromises();
    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(pullButton.text()).toBe("Connecting…");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "snapshot",
        listing: {
          active_jobs: [
            {
              id: "existing-pull",
              model: "Catalog model",
              catalog_id: "hf:Catalog model",
              status: "active",
              files_done: 0,
              files_total: 1,
              bytes_done: 20,
              bytes_total: 100,
            },
          ],
          queued: [],
          history: [],
        },
      } satisfies DownloadEvent),
    );
    await flushPromises();

    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(pullButton.text()).toBe("Pulling 20%");
    expect(wrapper.findAll("[data-test='mobile-catalog-download']")).toHaveLength(1);
  });

  it("shows per-host pull state in the card, detail, and host picker", async () => {
    const start = deferred<string | null>();
    startCatalogDownload.mockReturnValue(start.promise);
    wrapper = mountCatalog();
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await catalogCard.get(".mobile-catalog-pull").trigger("click");
    await flushPromises();
    let targetButtons = document.querySelectorAll<HTMLButtonElement>(
      ".mobile-catalog-target-list button",
    );
    targetButtons[1]!.click();
    await flushPromises();

    expect(catalogCard.get(".mobile-catalog-pull").text()).toBe("Starting…");
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);

    // Reopening the picker makes the chosen host's duplicate-safe state explicit.
    await catalogCard.get(".mobile-catalog-pull").trigger("click");
    await flushPromises();
    targetButtons = document.querySelectorAll<HTMLButtonElement>(
      ".mobile-catalog-target-list button",
    );
    expect(targetButtons[0]!.disabled).toBe(false);
    expect(targetButtons[1]!.disabled).toBe(true);
    expect(targetButtons[1]!.textContent).toContain("Starting…");
    targetButtons[1]!.click();
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);

    start.resolve("download-render");
    await flushPromises();
    streams[1]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "download-render",
        model: "Catalog model",
        position: 0,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(targetButtons[1]!.textContent).toContain("Queued");

    document.querySelector<HTMLButtonElement>("[aria-label='Close host picker']")!.click();
    await catalogCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    const detailAction = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-detail-action button",
    )!;
    expect(detailAction.textContent).toContain("Queued");

    streams[1]!.onEvent(
      "download",
      JSON.stringify({
        type: "started",
        id: "download-render",
        files_total: 2,
        bytes_total: 100,
      } satisfies DownloadEvent),
    );
    streams[1]!.onEvent(
      "download",
      JSON.stringify({
        type: "progress",
        id: "download-render",
        files_done: 1,
        bytes_done: 60,
      } satisfies DownloadEvent),
    );
    await flushPromises();
    expect(detailAction.textContent).toContain("Pulling 60%");
    expect(catalogCard.get(".mobile-catalog-pull").text()).toBe("Pulling 60%");
  });

  it("shows a visible error and does not POST when the download stream cannot open", async () => {
    sseStream.mockImplementation(
      (
        _path: string,
        options: {
          target: ApiTarget;
          signal: AbortSignal;
          onOpen?: () => void;
          onOpenError?: (error: Error) => void;
          onClose?: (error: Error | null) => void;
          onEvent: (event: string, data: string) => void;
        },
      ) => {
        streams.push(options);
        return Promise.resolve();
      },
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await catalogCard.get(".mobile-catalog-pull").trigger("click");
    streams[0]!.onOpenError?.(new Error("HTTP 401"));
    await flushPromises();

    expect(startCatalogDownload).not.toHaveBeenCalled();
    const status = wrapper.get("[data-test='mobile-catalog-action-status']");
    expect(status.attributes("role")).toBe("alert");
    expect(status.text()).toContain(
      "Could not pull Catalog model on Studio: Studio didn’t accept the API key.",
    );
  });

  it("repairs an installed model on its owning host and labels component presence in text", async () => {
    fetchModelComponents.mockResolvedValue({
      model: "installed:q8",
      components: [
        { name: "weights.safetensors", kind: "weights", present: true },
        { name: "text_encoder", kind: "companion", present: false },
      ],
    });
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const detail = document.querySelector<HTMLElement>("[data-test='mobile-catalog-detail']")!;
    expect(detail.textContent).toContain("Present · weights");
    expect(detail.textContent).toContain("Missing · companion");
    expect(detail.querySelector("[aria-label='Present: weights']")).not.toBeNull();
    expect(detail.querySelector("[aria-label='Missing: companion']")).not.toBeNull();

    detail.querySelector<HTMLButtonElement>(".mobile-catalog-detail-action button")!.click();
    await flushPromises();

    expect(document.querySelector("[data-test='mobile-catalog-target-sheet']")).toBeNull();
    expect(startCatalogDownload).toHaveBeenCalledWith("installed:q8", targets.studio, false);
  });

  it("asks which owning host to repair when an installed model exists on multiple hosts", async () => {
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    expect(installedCard.text()).toContain("Studio");
    expect(installedCard.text()).toContain("Render Box");

    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    document.querySelector<HTMLButtonElement>(".mobile-catalog-detail-action button")!.click();
    await flushPromises();

    const picker = document.querySelector<HTMLElement>(
      "[data-test='mobile-catalog-target-sheet']",
    )!;
    expect(picker).not.toBeNull();
    expect(picker.textContent).toContain("Choose where to repair");
    expect(picker.textContent).toContain("Studio");
    expect(picker.textContent).toContain("Render Box");
    expect(startCatalogDownload).not.toHaveBeenCalled();

    picker.querySelectorAll<HTMLButtonElement>(".mobile-catalog-target-list button")[1]!.click();
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith("installed:q8", targets.render, false);
  });

  it("still offers an install for a model installed only on another machine", async () => {
    // The merged row is "installed" because Render Box has it — Studio does not,
    // so an install must stay reachable from this row.
    apiFetchTo.mockImplementation((target: ApiTarget, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve(
          jsonResponse(
            target.baseUrl === renderBox.baseUrl ? [model("installed:q8", "flux", true)] : [],
          ),
        );
      }
      return Promise.resolve(new Response(null, { status: 204 }));
    });
    searchCatalog.mockResolvedValue(searchResponse([]));
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    // Where it already lives stays visible.
    expect(installedCard.get(".mobile-catalog-installed").text()).toBe("Installed");
    expect(installedCard.text()).toContain("Render Box");

    const pullButton = installedCard.get(".mobile-catalog-pull");
    expect(pullButton.text()).toContain("Pull");
    expect(pullButton.text()).not.toContain("Repair");

    await pullButton.trigger("click");
    await flushPromises();

    const picker = document.querySelector<HTMLElement>(
      "[data-test='mobile-catalog-target-sheet']",
    )!;
    expect(picker.textContent).toContain("Choose where to install");
    // A mixed list must not promise a fresh install for the machine that can
    // only be repaired.
    expect(picker.textContent).toContain("machines that already have it are repaired instead");
    const options = [
      ...picker.querySelectorAll<HTMLButtonElement>("[data-test='mobile-catalog-target-option']"),
    ];
    // Install targets lead; the owner is offered as a repair.
    expect(options.map((option) => option.dataset.action)).toEqual(["install", "repair"]);
    expect(options[0]!.textContent).toContain("Studio");
    expect(options[0]!.textContent).toContain("Install");
    expect(options[1]!.textContent).toContain("Render Box");
    expect(options[1]!.textContent).toContain("Repair");

    options[0]!.click();
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith("installed:q8", targets.studio, false);
    expect(document.body.textContent).toContain("on Studio");
    expect(document.body.textContent).toContain("Pulling");
  });

  it("offers the install from the detail sheet of a model missing on this machine", async () => {
    apiFetchTo.mockImplementation((target: ApiTarget, path: string) => {
      if (path === "/api/models") {
        return Promise.resolve(
          jsonResponse(
            target.baseUrl === renderBox.baseUrl ? [model("installed:q8", "flux", true)] : [],
          ),
        );
      }
      return Promise.resolve(new Response(null, { status: 204 }));
    });
    searchCatalog.mockResolvedValue(searchResponse([]));
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const detailAction = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-detail-action button",
    )!;
    expect(detailAction.textContent).toContain("Pull");
    expect(detailAction.textContent).not.toContain("Repair");

    detailAction.click();
    await flushPromises();
    const options = document.querySelectorAll<HTMLButtonElement>(
      "[data-test='mobile-catalog-target-option']",
    );
    expect([...options].map((option) => option.dataset.action)).toEqual(["install", "repair"]);
    options[0]!.click();
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith("installed:q8", targets.studio, false);
  });

  it("degrades to Repair only when every reachable machine already owns the model", async () => {
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    // Nothing left to install anywhere, so the row keeps its plain chip.
    expect(installedCard.find(".mobile-catalog-pull").exists()).toBe(false);

    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    const detailAction = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-detail-action button",
    )!;
    expect(detailAction.textContent).toContain("Repair");

    detailAction.click();
    await flushPromises();
    const picker = document.querySelector<HTMLElement>(
      "[data-test='mobile-catalog-target-sheet']",
    )!;
    expect(picker.textContent).toContain("Choose where to repair");
    const options = [
      ...picker.querySelectorAll<HTMLButtonElement>("[data-test='mobile-catalog-target-option']"),
    ];
    expect(options.map((option) => option.dataset.action)).toEqual(["repair", "repair"]);
  });

  it("segmented control swaps shelves and remembers the Discover sub-source", async () => {
    wrapper = mountCatalog();
    await flushPromises();

    expect(
      wrapper.get("[data-test='mobile-catalog-segment-discover']").attributes("aria-pressed"),
    ).toBe("true");

    const huggingFaceChip = () =>
      wrapper!.findAll(".mobile-catalog-sources button").find((b) => b.text() === "HuggingFace")!;
    await huggingFaceChip().trigger("click");
    await flushPromises();
    expect(huggingFaceChip().attributes("aria-pressed")).toBe("true");

    // Installed shelf hides the Discover filters and lists installed models.
    await wrapper.get("[data-test='mobile-catalog-segment-installed']").trigger("click");
    await flushPromises();
    expect(
      wrapper.get("[data-test='mobile-catalog-segment-installed']").attributes("aria-pressed"),
    ).toBe("true");
    expect(wrapper.find(".mobile-catalog-sources").exists()).toBe(false);
    expect(wrapper.text()).toContain("installed:q8");

    // Returning to Discover restores HuggingFace, not All.
    await wrapper.get("[data-test='mobile-catalog-segment-discover']").trigger("click");
    await flushPromises();
    expect(huggingFaceChip().attributes("aria-pressed")).toBe("true");
  });

  it("labels every card with a friendly model kind and explicitly marks NSFW entries", async () => {
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Portrait Base", { kind: "checkpoint" }),
        entry("Spicy Adapter", { kind: "lora", nsfw: true }),
      ]),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const cards = wrapper.findAll("[data-test='mobile-catalog-card']");
    const safeCard = cards.find((candidate) => candidate.text().includes("Portrait Base"))!;
    const nsfwCard = cards.find((candidate) => candidate.text().includes("Spicy Adapter"))!;

    expect(safeCard.get("[data-test='model-kind-badge']").text()).toBe("Checkpoint");
    expect(safeCard.find("[data-test='model-nsfw-badge']").exists()).toBe(false);
    expect(safeCard.get(".mobile-catalog-card-open").attributes("aria-label")).toContain(
      "Checkpoint",
    );
    expect(safeCard.get(".mobile-catalog-card-open").attributes("aria-label")).not.toContain(
      "NSFW",
    );

    expect(nsfwCard.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(nsfwCard.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
    expect(nsfwCard.get(".mobile-catalog-card-open").attributes("aria-label")).toContain("LoRA");
    expect(nsfwCard.get(".mobile-catalog-card-open").attributes("aria-label")).toContain(
      "18+ NSFW",
    );
  });

  it("preserves a mature summary classification when fetched detail reports false", async () => {
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Mature Adapter", {
          kind: "lora",
          nsfw: true,
          author: "Summary author",
          description: "Useful summary copy.",
          license: "apache-2.0",
          tags: ["portrait", "cinematic"],
          download_count: 12_300,
          likes: 456,
        }),
      ]),
    );
    fetchCatalogDetail.mockImplementation((id: string) =>
      Promise.resolve({
        ...entry("Mature Adapter"),
        id,
        kind: "lora",
        nsfw: false,
        author: null,
        description: null,
        license: null,
        tags: [],
        download_count: 0,
        likes: 0,
      }),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const card = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Mature Adapter"))!;
    await card.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const detail = document.querySelector<HTMLElement>("[data-test='mobile-catalog-detail']")!;
    expect(detail.querySelector("[data-test='model-nsfw-badge']")?.textContent?.trim()).toBe(
      "18+ NSFW",
    );
    expect(detail.textContent).toContain("Summary author");
    expect(detail.textContent).toContain("Useful summary copy.");
    expect(detail.textContent).toContain("apache-2.0");
    expect(detail.textContent).toContain("portrait");
    expect(detail.textContent).toContain("12.3k");
    expect(detail.textContent).toContain("456");
  });

  it("conservatively merges presentation metadata from duplicate installed hosts", async () => {
    searchCatalog.mockResolvedValue(searchResponse([]));
    fetchCatalogDetail.mockRejectedValue(new Error("older host has no detail endpoint"));
    apiFetchTo.mockImplementation((target: ApiTarget, path: string) => {
      if (path !== "/api/models") {
        return Promise.resolve(new Response(null, { status: 204 }));
      }
      return Promise.resolve(
        jsonResponse(
          target.baseUrl === studio.baseUrl
            ? [
                model("shared-adapter", "flux", true, {
                  description: "",
                  kind: null,
                  modality: null,
                  nsfw: null,
                }),
              ]
            : [
                model("shared-adapter", "flux", true, {
                  description: "A mature portrait adapter.",
                  kind: "lora",
                  modality: "image",
                  nsfw: true,
                }),
              ],
        ),
      );
    });
    wrapper = mountCatalog();
    await flushPromises();

    const cards = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .filter((candidate) => candidate.text().includes("shared-adapter"));
    expect(cards).toHaveLength(1);
    expect(cards[0]!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(cards[0]!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");

    await cards[0]!.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    const detail = document.querySelector<HTMLElement>("[data-test='mobile-catalog-detail']")!;
    expect(detail.getAttribute("aria-labelledby")).toBe(
      "mobile-catalog-detail-heading mobile-catalog-detail-title",
    );
    expect(document.getElementById("mobile-catalog-detail-heading")?.textContent).toContain(
      "Model details",
    );
    expect(document.getElementById("mobile-catalog-detail-title")?.textContent).toContain(
      "shared-adapter",
    );
    expect(detail.querySelector("[data-test='model-modality-badge']")?.textContent).toBe("Image");
    expect(detail.textContent).toContain("A mature portrait adapter.");
  });

  it("enriches a legacy installed catalog row before kind filtering and id dedup", async () => {
    vi.useFakeTimers();
    const live = entry("Legacy Adapter", {
      id: "cv:4242",
      source: "civitai",
      source_id: "4242",
      kind: "lora",
      modality: "image",
      nsfw: true,
      description: "Rich metadata from the live catalog.",
    });
    searchCatalog.mockResolvedValue(searchResponse([live]));
    apiFetchTo.mockImplementation((_target: ApiTarget, path: string) =>
      path === "/api/models"
        ? Promise.resolve(
            jsonResponse([
              model("cv:4242", "flux", true, {
                hf_repo: "",
                display_name: "Legacy Adapter",
                description: "",
                kind: null,
                modality: null,
                nsfw: null,
              }),
            ]),
          )
        : Promise.resolve(new Response(null, { status: 204 })),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const loraChip = wrapper
      .get("[data-test='mobile-catalog-kind-chips']")
      .findAll("button")
      .find((candidate) => candidate.text() === "LoRAs")!;
    await loraChip.trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    await flushPromises();

    const cards = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .filter((candidate) => candidate.text().includes("Legacy Adapter"));
    expect(cards).toHaveLength(1);
    expect(cards[0]!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(cards[0]!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
    expect(cards[0]!.text()).toContain("Installed");
    expect(cards[0]!.text()).toContain("Studio");
  });

  it("does not enrich or hide unrelated models that only share a human name", async () => {
    const live = entry("Shared title", {
      id: "hf:other/repo",
      source: "hf",
      source_id: "other/repo",
      kind: "lora",
      modality: "image",
      nsfw: true,
      description: "Metadata that belongs only to the other repository.",
    });
    searchCatalog.mockResolvedValue(searchResponse([live]));
    apiFetchTo.mockImplementation((_target: ApiTarget, path: string) =>
      path === "/api/models"
        ? Promise.resolve(
            jsonResponse([
              model("Shared title", "flux", true, {
                hf_repo: "installed/repo",
                description: "",
                kind: null,
                modality: null,
                nsfw: null,
              }),
            ]),
          )
        : Promise.resolve(new Response(null, { status: 204 })),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const cards = wrapper.findAll("[data-test='mobile-catalog-card']");
    expect(cards).toHaveLength(2);
    expect(cards[0]!.text()).toContain("Installed");
    expect(cards[0]!.get("[data-test='model-kind-badge']").text()).toBe("Checkpoint");
    expect(cards[0]!.find("[data-test='model-nsfw-badge']").exists()).toBe(false);
    expect(cards[0]!.text()).not.toContain("Metadata that belongs only");
    expect(cards[1]!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(cards[1]!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
  });

  it("classifies model details and labels weights, source, counts, and description accurately", async () => {
    fetchCatalogDetail.mockImplementation((id: string) =>
      Promise.resolve({
        ...entry("detail"),
        id,
        name: "Spicy Adapter",
        source: "civitai",
        kind: "lora",
        modality: "image",
        nsfw: true,
        size_bytes: 220_000_000,
        download_count: 12_300,
        likes: 456,
        description: "A portrait adapter with controlled studio lighting.",
      }),
    );
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Spicy Adapter", {
          source: "civitai",
          kind: "lora",
          modality: "image",
          nsfw: true,
        }),
      ]),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    const card = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Spicy Adapter"))!;
    await card.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const detail = document.querySelector<HTMLElement>("[data-test='mobile-catalog-detail']")!;
    expect(detail.querySelector("[data-test='model-modality-badge']")?.textContent).toBe("Image");
    expect(detail.querySelector("[data-test='model-kind-badge']")?.textContent).toBe("LoRA");
    expect(detail.querySelector("[data-test='model-nsfw-badge']")?.textContent?.trim()).toBe(
      "18+ NSFW",
    );

    const tiles = detail.querySelector("[data-test='mobile-catalog-detail-tiles']")!;
    expect(tiles.textContent).toContain("LoRA weights");
    expect(tiles.textContent).not.toContain("Checkpoint weights");

    const metadata = detail.querySelector(".mobile-catalog-detail-meta")!;
    expect(metadata.textContent).toContain("Source");
    expect(metadata.textContent).toContain("Civitai");
    expect(metadata.textContent).toContain("Downloads");
    expect(metadata.textContent).toContain("12.3k");
    expect(metadata.textContent).toContain("Likes");
    expect(metadata.textContent).toContain("456");
    expect(detail.textContent).toContain("A portrait adapter with controlled studio lighting.");
  });

  it("renders classification badges and Checkpoint weights/Footprint tiles in the detail sheet", async () => {
    fetchCatalogDetail.mockImplementation((id: string) =>
      Promise.resolve({
        ...entry("detail"),
        id,
        name: "Catalog model",
        description: "Full detail",
        modality: "image",
        size_bytes: 8_000_000_000,
      }),
    );
    wrapper = mountCatalog();
    await flushPromises();

    const card = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await card.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();

    const detail = document.querySelector<HTMLElement>("[data-test='mobile-catalog-detail']")!;
    expect(detail.querySelector("[data-test='model-modality-badge']")?.textContent).toBe("Image");
    const tiles = detail.querySelector("[data-test='mobile-catalog-detail-tiles']")!;
    expect(tiles.textContent).toContain("Checkpoint weights");
    expect(tiles.textContent).toContain("Footprint");
    expect(tiles.textContent).toContain("8.0 GB");
  });

  it("sends kind and sort from the Discover filters and resets to page 1", async () => {
    vi.useFakeTimers();
    wrapper = mountCatalog();
    await vi.waitFor(() => expect(searchCatalog).toHaveBeenCalledTimes(1));
    const initial = searchCatalog.mock.calls[0]![0] as { kind?: string; sort?: string };
    expect(initial.kind).toBeUndefined();
    expect(initial.sort).toBeUndefined();

    const chips = wrapper.get("[data-test='mobile-catalog-kind-chips']").findAll("button");
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
    await chips.find((c) => c.text() === "LoRAs")!.trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    expect(searchCatalog).toHaveBeenCalledTimes(2);
    expect(searchCatalog.mock.calls[1]![0]).toMatchObject({ kind: "lora", page: 1 });

    await wrapper.get("[data-test='mobile-catalog-sort']").setValue("rating");
    await vi.advanceTimersByTimeAsync(400);
    expect(searchCatalog).toHaveBeenCalledTimes(3);
    expect(searchCatalog.mock.calls[2]![0]).toMatchObject({
      kind: "lora",
      sort: "rating",
      page: 1,
    });
  });

  it("hides installed rows under a non-checkpoint kind and clears kind entering Installed", async () => {
    vi.useFakeTimers();
    wrapper = mountCatalog(studio.id, [studio]);
    await vi.waitFor(() => expect(wrapper!.text()).toContain("installed:q8"));
    expect(wrapper.text()).toContain("safe-variant:q4");

    const loraChip = () =>
      wrapper!
        .get("[data-test='mobile-catalog-kind-chips']")
        .findAll("button")
        .find((c) => c.text() === "LoRAs")!;
    await loraChip().trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    expect(wrapper.text()).not.toContain("installed:q8");
    expect(wrapper.text()).not.toContain("safe-variant:q4");
    expect(loraChip().attributes("aria-pressed")).toBe("true");

    // Installed hides the chips and still lists everything installed.
    await wrapper.get("[data-test='mobile-catalog-segment-installed']").trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    expect(wrapper.find("[data-test='mobile-catalog-kind-chips']").exists()).toBe(false);
    expect(wrapper.text()).toContain("installed:q8");

    // Returning to Discover starts from All, like the family filter.
    await wrapper.get("[data-test='mobile-catalog-segment-discover']").trigger("click");
    await vi.advanceTimersByTimeAsync(400);
    const all = wrapper
      .get("[data-test='mobile-catalog-kind-chips']")
      .findAll("button")
      .find((c) => c.text() === "All")!;
    expect(all.attributes("aria-pressed")).toBe("true");
  });

  it("clears a hidden family filter when switching to installed models", async () => {
    wrapper = mountCatalog();
    await flushPromises();

    await wrapper.get(".mobile-catalog-filters select").setValue("ltx2");
    await wrapper.get("[data-test='mobile-catalog-segment-installed']").trigger("click");
    await flushPromises();

    expect(wrapper.find(".mobile-catalog-filters select").exists()).toBe(false);
    expect(wrapper.text()).toContain("installed:q8");
  });

  it("retries the family taxonomy once the host's keychain key arrives", async () => {
    const offlineStudio: MobileHost = { ...studio, apiKey: "", online: false };
    fetchCatalogFamilies.mockRejectedValueOnce(new Error("unauthorized"));
    fetchCatalogFamilies.mockResolvedValue(["flux", "ltx2", "z-image"]);
    wrapper = mountCatalog(studio.id, [offlineStudio]);
    await flushPromises();
    expect(fetchCatalogFamilies).toHaveBeenCalledWith(false, {
      baseUrl: studio.baseUrl,
      apiKey: null,
    });
    expect(familyOptionValues(wrapper)).not.toContain("z-image");

    // The Keychain lookup resolves after mount and the probe finds the host.
    await wrapper.setProps({ hosts: [{ ...studio, online: true }] });
    await flushPromises();

    expect(fetchCatalogFamilies).toHaveBeenLastCalledWith(false, targets.studio);
    expect(familyOptionValues(wrapper)).toEqual(["", "flux", "ltx2", "z-image"]);
  });

  it("re-runs a failed search when the browsed host comes back online", async () => {
    const offlineStudio: MobileHost = { ...studio, online: false };
    searchCatalog.mockRejectedValueOnce(new Error("fetch failed"));
    wrapper = mountCatalog(studio.id, [offlineStudio]);
    await flushPromises();
    expect(wrapper.text()).toContain("Couldn’t reach Studio");

    await wrapper.setProps({ hosts: [{ ...studio, online: true }] });
    await flushPromises();

    expect(searchCatalog).toHaveBeenCalledTimes(2);
    expect(wrapper.text()).toContain("Catalog model");
    expect(wrapper.text()).not.toContain("Couldn’t reach Studio");
  });

  it("falls back to the families seen in results when the taxonomy is unavailable", async () => {
    fetchCatalogFamilies.mockRejectedValue(new Error("fetch failed"));
    searchCatalog.mockResolvedValue(
      searchResponse([
        entry("Video model", { id: "hf:video", family: "ltx2" }),
        entry("Image model", { id: "hf:image", family: "z-image" }),
      ]),
    );
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();

    // Installed inventory contributes its families too (flux from installed:q8).
    expect(familyOptionValues(wrapper)).toEqual(["", "flux", "ltx2", "z-image"]);

    // Filtering must not collapse the list to the one family still on screen.
    await wrapper.get(".mobile-catalog-filters select").setValue("ltx2");
    await flushPromises();
    expect(familyOptionValues(wrapper)).toEqual(["", "flux", "ltx2", "z-image"]);
  });

  it("keeps the loaded family taxonomy when a later reload fails", async () => {
    wrapper = mountCatalog(studio.id, [studio]);
    await flushPromises();
    expect(familyOptionValues(wrapper)).toEqual(["", "flux", "ltx2"]);

    fetchCatalogFamilies.mockRejectedValue(new Error("fetch failed"));
    await wrapper.setProps({ hosts: [{ ...studio, online: false }] });
    await flushPromises();

    expect(fetchCatalogFamilies).toHaveBeenCalledTimes(2);
    expect(familyOptionValues(wrapper)).toEqual(["", "flux", "ltx2"]);
  });

  it("resets detail loading state when a pending installed detail is replaced", async () => {
    const components = deferred<{ model: string; components: [] }>();
    fetchModelComponents.mockReturnValue(components.promise);
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    expect(document.body.textContent).toContain("Checking files");

    document.querySelector<HTMLButtonElement>("[aria-label='Close model details']")!.click();
    await flushPromises();
    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await catalogCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    expect(
      document.querySelector("[data-test='mobile-catalog-detail']")?.textContent,
    ).not.toContain("Checking files");

    components.resolve({ model: "installed:q8", components: [] });
    await flushPromises();
  });

  it("does not let a completed removal close a newly opened model detail", async () => {
    const removal = deferred<{ removed: string[]; kept: string[]; freed_bytes: number }>();
    removeModel.mockReturnValue(removal.promise);
    fetchCatalogDetail.mockImplementation((id: string) =>
      Promise.resolve(
        entry(id === "hf:Catalog model" ? "Fresh catalog" : "Installed model", {
          id,
          installed: false,
        }),
      ),
    );
    wrapper = mountCatalog();
    await flushPromises();

    const installedCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("installed:q8"))!;
    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    let removeButton = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-installed-actions .danger-button",
    )!;
    removeButton.click();
    await flushPromises();
    expect(removeButton.textContent).toContain("Tap again to remove");

    document.querySelector<HTMLButtonElement>("[aria-label='Close model details']")!.click();
    await flushPromises();
    await installedCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    removeButton = document.querySelector<HTMLButtonElement>(
      ".mobile-catalog-installed-actions .danger-button",
    )!;
    expect(removeButton.textContent).toContain("Remove from host");
    removeButton.click();
    removeButton.click();
    await flushPromises();
    expect(removeModel).toHaveBeenCalledWith("installed:q8", targets.studio);

    document.querySelector<HTMLButtonElement>("[aria-label='Close model details']")!.click();
    await flushPromises();
    const catalogCard = wrapper
      .findAll("[data-test='mobile-catalog-card']")
      .find((candidate) => candidate.text().includes("Catalog model"))!;
    await catalogCard.get(".mobile-catalog-card-open").trigger("click");
    await flushPromises();
    expect(document.querySelector("[data-test='mobile-catalog-detail']")?.textContent).toContain(
      "Fresh catalog",
    );
    expect(
      document.querySelector<HTMLButtonElement>(".mobile-catalog-detail-action button")!.disabled,
    ).toBe(false);

    removal.resolve({ removed: ["installed:q8"], kept: [], freed_bytes: 1_000_000_000 });
    await flushPromises();
    expect(document.querySelector("[data-test='mobile-catalog-detail']")?.textContent).toContain(
      "Fresh catalog",
    );
  });

  it("reduces download SSE snapshots, exposes progress, cancels explicitly, and aborts streams", async () => {
    wrapper = mountCatalog();
    await flushPromises();
    expect(streams).toHaveLength(2);

    const event: DownloadEvent = {
      type: "snapshot",
      listing: {
        active_jobs: [
          {
            id: "job-1",
            model: "flux-dev:q8",
            status: "active",
            files_done: 0,
            files_total: 0,
            bytes_done: 0,
            bytes_total: 0,
            current_file: "Verifying file [1/4] tokenizer.json...",
          },
        ],
        queued: [],
        history: [
          {
            id: "finished-while-suspended",
            model: "offline-finish:q8",
            status: "completed",
            files_done: 1,
            files_total: 1,
            bytes_done: 100,
            bytes_total: 100,
          },
        ],
      },
    };
    streams[0]!.onEvent("download", JSON.stringify(event));
    await flushPromises();

    const row = wrapper.get("[data-test='mobile-catalog-download']");
    expect(row.text()).toContain("flux-dev:q8");
    expect(row.text()).toContain("Preparing");
    expect(row.text()).toContain("Verifying file [1/4]");
    expect(row.text()).not.toContain("Waiting");
    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "started", id: "job-1", files_total: 4, bytes_total: 100 }),
    );
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "progress",
        id: "job-1",
        files_done: 1,
        bytes_done: 25,
        current_file: "transformer.gguf",
      }),
    );
    await flushPromises();
    expect(row.get("[role='progressbar']").attributes("aria-valuenow")).toBe("25");
    expect(wrapper.emitted("models-changed")).toContainEqual([studio.id]);

    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "job_done", id: "job-1", model: "flux-dev:q8" }),
    );
    await flushPromises();
    expect(wrapper.emitted("models-changed")).toContainEqual([studio.id]);
    expect(wrapper.get("[data-test='mobile-catalog-action-status']").text()).toContain(
      "flux-dev:q8 is ready on Studio",
    );

    // Put the row back into the snapshot so cancellation remains independently
    // covered after the terminal-event reconciliation above.
    streams[0]!.onEvent("download", JSON.stringify(event));
    await flushPromises();
    const activeRow = wrapper.get("[data-test='mobile-catalog-download']");
    await activeRow.get("button").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(targets.studio, "/api/downloads/job-1", {
      method: "DELETE",
    });

    const signals = streams.map((stream) => stream.signal);
    wrapper.unmount();
    wrapper = null;
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });
});

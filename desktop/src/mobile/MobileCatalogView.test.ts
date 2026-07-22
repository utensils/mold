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
  vi.useRealTimers();
});

describe("MobileCatalogView", () => {
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
    expect(status.text()).toContain("Could not pull Catalog model on Studio: HTTP 401");
  });

  it("repairs an installed model on its owning host and labels component presence in text", async () => {
    fetchModelComponents.mockResolvedValue({
      model: "installed:q8",
      components: [
        { name: "weights.safetensors", kind: "weights", present: true },
        { name: "text_encoder", kind: "companion", present: false },
      ],
    });
    wrapper = mountCatalog();
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

  it("renders a media badge and Checkpoint/Footprint size tiles in the detail sheet", async () => {
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
    expect(detail.querySelector("[data-test='mobile-catalog-detail-badge']")?.textContent).toBe(
      "image",
    );
    const tiles = detail.querySelector("[data-test='mobile-catalog-detail-tiles']")!;
    expect(tiles.textContent).toContain("Checkpoint");
    expect(tiles.textContent).toContain("Footprint");
    expect(tiles.textContent).toContain("8.0 GB");
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
            files_done: 1,
            files_total: 4,
            bytes_done: 25,
            bytes_total: 100,
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

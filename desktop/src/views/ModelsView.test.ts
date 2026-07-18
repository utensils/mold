import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import type { CatalogEntry, ModelEntry } from "../lib/api/types";

const { searchCatalog, fetchCatalogFamilies } = vi.hoisted(() => ({
  searchCatalog: vi.fn(),
  fetchCatalogFamilies: vi.fn().mockResolvedValue([]),
}));
vi.mock("../lib/api/catalog", () => ({
  searchCatalog,
  fetchCatalogFamilies,
  fetchCatalogDetail: vi.fn().mockRejectedValue(new Error("no detail in tests")),
  startCatalogDownload: vi.fn(),
}));
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiFetch: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiJson: vi.fn().mockResolvedValue([]),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../lib/api/sse", () => ({
  sseStream: vi.fn().mockImplementation((_path: string, options: { onOpen?: () => void }) => {
    options.onOpen?.();
    return Promise.resolve();
  }),
}));
vi.mock("../lib/catalogSizes", () => ({
  resolveEntrySize: vi.fn().mockResolvedValue(null),
}));

import ModelsView from "./ModelsView.vue";
import { getActivePinia } from "pinia";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useUiStore } from "../stores/ui";
import type { DownloadJob } from "../lib/api/types";

const stub = { template: "<div />" };

function model(name: string, family: string): ModelEntry {
  return {
    name,
    family,
    size_gb: 10,
    is_loaded: false,
    hf_repo: "org/repo",
    default_steps: 4,
    default_guidance: 1,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    disk_usage_bytes: 10_000_000_000,
  };
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
  };
}

function job(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "j1",
    model: "flux-dev:q4",
    status: "active",
    files_done: 1,
    files_total: 4,
    bytes_done: 25,
    bytes_total: 100,
    ...overrides,
  };
}

let router: Router;

async function mountView(path = "/models") {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: "/models", component: stub }],
  });
  await router.push(path);
  const pinia = createPinia();
  setActivePinia(pinia);
  const models = useModelStore();
  models.all = [model("flux-dev:q8", "flux"), model("ltx-2:q8", "ltx-2")];
  const wrapper = mount(ModelsView, { global: { plugins: [pinia, router] } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchCatalogFamilies.mockResolvedValue([]);
  searchCatalog.mockResolvedValue({
    entries: [entry("FLUX.2 Klein", "flux2"), entry("LTX-2 Distilled", "ltx2")],
    page: 1,
    page_size: 24,
    total: 2,
  });
});

describe("ModelsView unified catalog", () => {
  it("always stacks Installed and Catalog with no availability toggle", async () => {
    const wrapper = await mountView();
    expect(wrapper.find('[aria-label="Model availability"]').exists()).toBe(false);
    expect(wrapper.find("#installed-models-heading").exists()).toBe(true);
    expect(wrapper.find("#available-models-heading").exists()).toBe(true);
  });

  it("pins the downloads tray above the installed and catalog sections", async () => {
    const wrapper = await mountView();
    useDownloadsStore().activeJobs = [job()];
    await flushPromises();
    const html = wrapper.html();
    const downloadsAt = html.indexOf("Downloads");
    expect(downloadsAt).toBeGreaterThan(-1);
    expect(downloadsAt).toBeLessThan(html.indexOf("installed-models-heading"));
    expect(downloadsAt).toBeLessThan(html.indexOf("available-models-heading"));
  });

  it("shows installed models and active downloads from a connected remote host", async () => {
    const wrapper = await mountView();
    useHostsStore().extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    useHostModelsStore().byHost["hal9000-7680"] = {
      entries: [model("qwen-image:q4", "qwen-image")],
      fetchedAt: Date.now(),
      error: null,
    };
    useDownloadsStore().hostStates["hal9000-7680"] = {
      label: "hal9000",
      target: { baseUrl: "http://hal9000:7680", apiKey: null },
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
      activeJobs: [job({ model: "qwen-image:q4" })],
      queued: [],
      history: [],
    };
    await flushPromises();

    expect(wrapper.text()).toContain("qwen-image:q4");
    expect(wrapper.text()).toContain("hal9000");
    expect(
      wrapper.findAll("[data-test='installed-host']").some((chip) => chip.text() === "hal9000"),
    ).toBe(true);
    expect(wrapper.find("[aria-label='Downloading qwen-image:q4 on hal9000']").exists()).toBe(true);
  });

  it("resubscribes a ready host's download stream when its API key changes", async () => {
    await mountView();
    const hosts = useHostsStore();
    hosts.extras.push({
      id: "hal9000-7680",
      label: "hal9000",
      url: "http://hal9000:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    await flushPromises();
    expect(useDownloadsStore().hostStates["hal9000-7680"]?.target.apiKey).toBeNull();

    hosts.extras[0]!.apiKey = "rotated-key";
    await flushPromises();

    expect(useDownloadsStore().hostStates["hal9000-7680"]?.target.apiKey).toBe("rotated-key");
  });

  it("filters installed and catalog entries to video families for ?type=video", async () => {
    const wrapper = await mountView("/models?type=video");
    expect(wrapper.text()).toContain("ltx-2:q8");
    expect(wrapper.text()).not.toContain("flux-dev:q8");
    expect(wrapper.text()).toContain("LTX-2 Distilled");
    expect(wrapper.text()).not.toContain("FLUX.2 Klein");
  });

  it("filters to image families for ?type=image", async () => {
    const wrapper = await mountView("/models?type=image");
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).not.toContain("ltx-2:q8");
    expect(wrapper.text()).toContain("FLUX.2 Klein");
    expect(wrapper.text()).not.toContain("LTX-2 Distilled");
  });

  it("updates the route query when a media-type chip is selected", async () => {
    const wrapper = await mountView();
    const chips = wrapper.get('[aria-label="Media type"]').findAll("button");
    expect(chips.map((c) => c.text())).toEqual(["All", "Images", "Video"]);

    await chips[2]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.type).toBe("video");
    expect(wrapper.text()).not.toContain("flux-dev:q8");

    await chips[0]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.type).toBeUndefined();
    expect(wrapper.text()).toContain("flux-dev:q8");
  });

  it("treats the legacy catalog deep link as the unfiltered view", async () => {
    const wrapper = await mountView("/models?tab=catalog");
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).toContain("LTX-2 Distilled");
    const all = wrapper.get('[aria-label="Media type"]').findAll("button")[0]!;
    expect(all.attributes("aria-pressed")).toBe("true");
  });

  it("scrolls to the catalog section from the empty-shelf CTA", async () => {
    const scrollIntoView = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoView;
    const wrapper = await mountView();
    useModelStore().all = [];
    await flushPromises();

    const cta = wrapper.findAll("button").find((b) => b.text() === "Browse the catalog");
    expect(cta).toBeDefined();
    await cta!.trigger("click");
    expect(scrollIntoView).toHaveBeenCalled();
  });
});

describe("ModelsView catalog layout", () => {
  it("defaults to the table layout", async () => {
    const wrapper = await mountView();
    expect(wrapper.get("[data-test='layout-table']").attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='layout-grid']").attributes("aria-checked")).toBe("false");
  });

  it("persists the chosen layout for the app session, not just the mounted view", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='layout-grid']").trigger("click");
    expect(useUiStore().catalogLayout).toBe("grid");
    wrapper.unmount();

    // Same pinia = same app session: leaving Catalog and coming back keeps
    // the choice, while a fresh session (new pinia) resets to table.
    const again = mount(ModelsView, { global: { plugins: [getActivePinia()!, router] } });
    await flushPromises();
    expect(again.get("[data-test='layout-grid']").attributes("aria-checked")).toBe("true");
    expect(again.get("[data-test='layout-table']").attributes("aria-checked")).toBe("false");
    again.unmount();
  });
});

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
import { authenticatedMiniMaxH3Capabilities } from "@studio/lib/minimaxH3Inventory.testFixtures";
import { getActivePinia } from "pinia";
import { useConnectionStore } from "../stores/connection";
import type { ServerCapabilities } from "../lib/api/types";
import { useDownloadsStore } from "../stores/downloads";
import { useHostModelsStore } from "../stores/hostModels";
import { useHostsStore } from "../stores/hosts";
import { useModelStore } from "../stores/models";
import { useUiStore } from "../stores/ui";
import type { DownloadJob } from "../lib/api/types";

function model(name: string, family: string, overrides: Partial<ModelEntry> = {}): ModelEntry {
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
    ...overrides,
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
    routes: [{ path: "/models", component: { template: "<div />" } }],
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

/** Click one of the Ready to use | Browse more segment buttons. */
async function selectSegment(
  wrapper: Awaited<ReturnType<typeof mountView>>,
  label: "Ready to use" | "Browse more",
) {
  const seg = wrapper.get('[aria-label="Styles view"]');
  const button = seg.findAll("button").find((b) => b.text().startsWith(label));
  await button!.trigger("click");
  await flushPromises();
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

describe("ModelsView H3 runtime placement", () => {
  it("does not render the fleet-wide H3 runtime panel — it lives in host detail", async () => {
    const wrapper = await mountView();
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: null };
    conn.status = "ready";
    const hosts = useHostsStore();
    hosts.capabilities.local =
      authenticatedMiniMaxH3Capabilities() as unknown as ServerCapabilities;
    await flushPromises();
    expect(wrapper.find('[data-test="h3-inventory"]').exists()).toBe(false);
  });
});

describe("ModelsView segments", () => {
  it("defaults to the Installed segment showing the full-featured inventory", async () => {
    const wrapper = await mountView();
    // Installed models render; the live catalog is not fetched into view yet.
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).not.toContain("FLUX.2 Klein");
    // No Discover source chips while on Installed.
    expect(wrapper.find("[data-test='catalog-source-chips']").exists()).toBe(false);
  });

  it("Discover shows the unified merged list — installed models and live catalog together", async () => {
    const wrapper = await mountView();
    await selectSegment(wrapper, "Browse more");
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).toContain("FLUX.2 Klein");
  });

  it("Discover source chips are All / HuggingFace / Civitai with All the default", async () => {
    const wrapper = await mountView();
    await selectSegment(wrapper, "Browse more");
    const chips = wrapper.get("[data-test='catalog-source-chips']").findAll("button");
    expect(chips.map((c) => c.text())).toEqual(["All", "HuggingFace", "Civitai"]);
    expect(chips[0]!.attributes("aria-pressed")).toBe("true");
  });

  it("tags NSFW catalog entries while keeping the include checkbox in Discover", async () => {
    searchCatalog.mockResolvedValue({
      entries: [{ ...entry("Spicy Model", "sdxl"), nsfw: true }],
      page: 1,
      page_size: 24,
      total: 1,
    });
    const wrapper = await mountView();
    await selectSegment(wrapper, "Browse more");
    expect(wrapper.find("[data-test='nsfw-tag']").exists()).toBe(true);
    expect(wrapper.find("input[type='checkbox']").exists()).toBe(true);
  });

  it("routes the empty Installed CTA to the Discover segment", async () => {
    const wrapper = await mountView();
    useModelStore().all = [];
    await flushPromises();
    const cta = wrapper.findAll("button").find((b) => b.text() === "Browse more styles");
    expect(cta).toBeDefined();
    await cta!.trigger("click");
    await flushPromises();
    expect(wrapper.find("[data-test='catalog-source-chips']").exists()).toBe(true);
  });

  it("treats the legacy ?tab=catalog deep link as the Discover view", async () => {
    const wrapper = await mountView("/models?tab=catalog");
    expect(wrapper.find("[data-test='catalog-source-chips']").exists()).toBe(true);
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).toContain("LTX-2 Distilled");
    const all = wrapper.get('[aria-label="What they make"]').findAll("button")[0]!;
    expect(all.attributes("aria-checked")).toBe("true");
  });
});

describe("ModelsView downloads and remote hosts", () => {
  it("pins the downloads tray above the content in both segments, tagged with its host", async () => {
    const wrapper = await mountView();
    useDownloadsStore().activeJobs = [job()];
    await flushPromises();
    // Installed segment: the tray sits above the installed inventory.
    let html = wrapper.html();
    expect(html.indexOf("downloads-tray")).toBeGreaterThan(-1);
    expect(html.indexOf("downloads-tray")).toBeLessThan(html.indexOf("flux-dev:q8"));

    await selectSegment(wrapper, "Browse more");
    html = wrapper.html();
    expect(html.indexOf("downloads-tray")).toBeLessThan(html.indexOf("catalog-source-chips"));
  });

  it("pins the banner outside the scrolling list, not inside it", async () => {
    const wrapper = await mountView();
    useDownloadsStore().activeJobs = [job()];
    await flushPromises();
    const tray = wrapper.get("[data-test='downloads-tray']").element;
    const scroll = wrapper.get("[data-test='models-scroll']").element;
    expect(scroll.contains(tray)).toBe(false);
    expect(scroll.compareDocumentPosition(tray) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
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

  it("keeps local defaults while merging richer metadata from a duplicate remote model", async () => {
    const wrapper = await mountView();
    const local = useModelStore().all.find((entry) => entry.name === "flux-dev:q8")!;
    local.description = "";
    local.kind = null;
    local.modality = null;
    local.nsfw = null;
    local.default_steps = 4;

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
      entries: [
        model("flux-dev:q8", "flux", {
          description: "A mature portrait adapter.",
          kind: "lora",
          modality: "image",
          nsfw: true,
          default_steps: 40,
        }),
      ],
      fetchedAt: Date.now(),
      error: null,
    };
    await flushPromises();

    const row = wrapper
      .findAll("[data-test='model-table-row']")
      .find((candidate) => candidate.text().includes("flux-dev:q8"));
    expect(row).toBeDefined();
    expect(row!.get("[data-test='model-kind-badge']").text()).toBe("LoRA");
    expect(row!.get("[data-test='model-nsfw-badge']").text()).toBe("18+ NSFW");
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
});

describe("ModelsView media-type filter", () => {
  it("filters the Installed inventory to video families for ?type=video", async () => {
    const wrapper = await mountView("/models?type=video");
    expect(wrapper.text()).toContain("ltx-2:q8");
    expect(wrapper.text()).not.toContain("flux-dev:q8");
  });

  it("filters the Installed inventory to image families for ?type=image", async () => {
    const wrapper = await mountView("/models?type=image");
    expect(wrapper.text()).toContain("flux-dev:q8");
    expect(wrapper.text()).not.toContain("ltx-2:q8");
  });

  it("filters the Discover merged list by media type too", async () => {
    const wrapper = await mountView("/models?type=video");
    await selectSegment(wrapper, "Browse more");
    expect(wrapper.text()).toContain("ltx-2:q8");
    expect(wrapper.text()).toContain("LTX-2 Distilled");
    expect(wrapper.text()).not.toContain("flux-dev:q8");
    expect(wrapper.text()).not.toContain("FLUX.2 Klein");
  });

  it("updates the route query when a media-type chip is selected", async () => {
    const wrapper = await mountView();
    const chips = wrapper.get('[aria-label="What they make"]').findAll("button");
    // The Create toolbar's words, plus the 3-D kind Styles used to lack.
    expect(chips.map((c) => c.text())).toEqual([
      "All",
      "Still picture",
      "Short clip",
      "3-D object",
    ]);

    await chips[2]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.type).toBe("video");
    expect(wrapper.text()).not.toContain("flux-dev:q8");

    await chips[3]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.type).toBe("mesh");

    await chips[0]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.type).toBeUndefined();
    expect(wrapper.text()).toContain("flux-dev:q8");
  });
});

describe("ModelsView Discover layout toggle", () => {
  it("defaults to the table layout", async () => {
    const wrapper = await mountView();
    await selectSegment(wrapper, "Browse more");
    expect(wrapper.get("[data-test='layout-table']").attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='layout-grid']").attributes("aria-checked")).toBe("false");
  });

  it("persists the chosen layout for the app session", async () => {
    const wrapper = await mountView();
    await selectSegment(wrapper, "Browse more");
    await wrapper.get("[data-test='layout-grid']").trigger("click");
    expect(useUiStore().catalogLayout).toBe("grid");
    wrapper.unmount();

    // Same pinia = same app session: a fresh mount keeps the choice.
    const again = mount(ModelsView, { global: { plugins: [getActivePinia()!, router] } });
    await flushPromises();
    await selectSegment(again, "Browse more");
    expect(again.get("[data-test='layout-grid']").attributes("aria-checked")).toBe("true");
    expect(again.get("[data-test='layout-table']").attributes("aria-checked")).toBe("false");
    again.unmount();
  });
});

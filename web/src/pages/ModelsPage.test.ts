import { flushPromises, mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import ModelsPage from "./ModelsPage.vue";
import type { ModelInfoExtended, ServerCapabilities } from "../types";
import type { RoutableHost } from "../lib/hostRouting";
import InstalledModelRow from "../components/models/InstalledModelRow.vue";
import { addHost } from "../lib/hostRegistry";
import { useModelInstallTargets } from "../composables/useModelInstallTargets";

function makeModel(over: Partial<ModelInfoExtended> = {}): ModelInfoExtended {
  return {
    name: "flux-schnell:q8",
    family: "flux",
    size_gb: 12,
    is_loaded: false,
    last_used: null,
    hf_repo: "bfl/flux",
    downloaded: true,
    default_steps: 4,
    default_guidance: 0,
    default_width: 1024,
    default_height: 1024,
    description: "",
    ...over,
  };
}

let mock: {
  tab: ReturnType<typeof ref<"installed" | "discover">>;
  installed: ReturnType<typeof ref<ModelInfoExtended[]>>;
  installedLoading: ReturnType<typeof ref<boolean>>;
  installedError: ReturnType<typeof ref<string | null>>;
  setTab: ReturnType<typeof vi.fn>;
  refreshInstalled: ReturnType<typeof vi.fn>;
  refresh: ReturnType<typeof vi.fn>;
  openInstalledDetail: ReturnType<typeof vi.fn>;
  setFilter: ReturnType<typeof vi.fn>;
};

const routeQuery = ref<Record<string, unknown>>({});

vi.mock("../composables/useCatalog", () => ({
  useCatalog: () => mock,
}));
vi.mock("vue-router", () => ({
  useRoute: () => ({ query: routeQuery.value }),
}));

/* Multi-host install targeting; single-host by default so nothing changes. */
const mockHosts = ref<RoutableHost[]>([]);
const mockOwners = ref<Record<string, string[]>>({});
const mockCapabilities = ref<Record<string, ServerCapabilities>>({});
vi.mock("../composables/useHostRouting", () => ({
  useHostRouting: () => ({
    hosts: mockHosts,
    capabilitiesByHost: mockCapabilities,
    modelOwnerIds: (name: string) => mockOwners.value[name] ?? [],
    inventoryKnown: () => true,
  }),
}));

const mockHostCatalogDownload = vi.fn().mockResolvedValue({ queued: 1 });
vi.mock("../components/machines/hostClient", () => ({
  hostModelDownload: (...args: unknown[]) => mockHostCatalogDownload(...args),
}));

const mockToast = vi.fn();
vi.mock("../lib/toasts", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

function host(id: string, over: Partial<RoutableHost> = {}): RoutableHost {
  return {
    id,
    label: id === "origin" ? "this server" : id,
    url: id === "origin" ? "" : `http://${id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
    ...over,
  };
}

const mountPage = () =>
  mount(ModelsPage, {
    global: {
      stubs: {
        CatalogTopbar: true,
        CatalogSidebar: true,
        CatalogCardGrid: true,
        ModelDetailDrawer: true,
      },
    },
  });

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  routeQuery.value = {};
  mockHosts.value = [host("origin")];
  mockOwners.value = {};
  mockCapabilities.value = {};
  useModelInstallTargets().cancel();
  mock = {
    tab: ref("installed"),
    installed: ref<ModelInfoExtended[]>([]),
    installedLoading: ref(false),
    installedError: ref<string | null>(null),
    setTab: vi.fn((t: "installed" | "discover") => {
      mock.tab.value = t;
    }),
    refreshInstalled: vi.fn(),
    refresh: vi.fn(),
    openInstalledDetail: vi.fn(),
    setFilter: vi.fn(),
  };
});

describe("ModelsPage — Models workspace", () => {
  it("lets the workspace shrink to the mobile viewport", () => {
    const w = mountPage();
    expect(w.get(".models").classes()).toContain("min-w-0");
    expect(w.get(".models").classes()).toContain("w-full");
  });

  it("refreshes installed models and the catalog on mount", () => {
    mountPage();
    expect(mock.refreshInstalled).toHaveBeenCalled();
    expect(mock.refresh).toHaveBeenCalled();
  });

  it("shows the installed tab with model rows when models are installed", () => {
    mock.installed.value = [makeModel(), makeModel({ name: "sdxl-base" })];
    const w = mountPage();
    expect(w.find("[data-test=installed-tab]").exists()).toBe(true);
    expect(w.findAllComponents(InstalledModelRow)).toHaveLength(2);
  });

  it("passes loaded state through to the row badge", () => {
    mock.installed.value = [makeModel({ is_loaded: true })];
    const w = mountPage();
    expect(w.find("[data-test=loaded-badge]").exists()).toBe(true);
  });

  it("opens the drawer for an installed model via openInstalledDetail", async () => {
    const model = makeModel();
    mock.installed.value = [model];
    const w = mountPage();
    await w.find("[data-test=installed-row]").trigger("click");
    expect(mock.openInstalledDetail).toHaveBeenCalledWith(model);
  });

  it("keeps the installed shelf free of install controls on one machine", () => {
    mock.installed.value = [makeModel()];
    const w = mountPage();
    expect(w.find("[data-test=install-elsewhere-btn]").exists()).toBe(false);
    expect(w.find("[data-test=install-target-host]").exists()).toBe(false);
  });

  it("installs an already-installed model on the machine that lacks it", async () => {
    const remote = addHost({ url: "http://studio.local:7680", name: "Studio" });
    mockHosts.value = [host("origin"), host(remote.id, { label: remote.name })];
    mockOwners.value = { "flux-schnell:q8": ["origin"] };
    mock.installed.value = [makeModel()];
    const w = mountPage();

    await w.find("[data-test=install-elsewhere-btn]").trigger("click");
    await flushPromises();

    // Two machines, two outcomes — the picker names both.
    const options = w.findAll("[data-test=install-target-option]");
    expect(options).toHaveLength(2);
    expect(options[0].attributes("data-host")).toBe(remote.id);
    expect(options[0].text()).toContain("Install");

    await options[0].trigger("click");
    await flushPromises();

    expect(mockHostCatalogDownload).toHaveBeenCalledWith(
      remote,
      "flux-schnell:q8",
    );
    expect(mockToast).toHaveBeenCalledWith(
      "success",
      "Download queued on Studio",
    );
    expect(w.find("[data-test=install-target-host]").exists()).toBe(false);
  });

  it("switches to Discover via the segmented control", async () => {
    mock.installed.value = [makeModel()];
    const w = mountPage();
    const discoverBtn = w
      .find("[data-test=models-tabs]")
      .findAll("button")
      .find((b) => b.text() === "Discover");
    expect(discoverBtn).toBeDefined();
    await discoverBtn!.trigger("click");
    expect(mock.setTab).toHaveBeenCalledWith("discover");
    expect(w.find("[data-test=discover-tab]").exists()).toBe(true);
  });

  it("filters installed rows by a local search over name and family", async () => {
    mock.installed.value = [
      makeModel({ name: "flux-schnell:q8", family: "flux" }),
      makeModel({ name: "sdxl-base", family: "sdxl" }),
    ];
    const w = mountPage();
    await w.find("[data-test=installed-search]").setValue("sdxl");
    expect(w.findAllComponents(InstalledModelRow)).toHaveLength(1);
    expect(w.text()).toContain("sdxl-base");
  });

  it("shows the no-match empty state and a clear-search action", async () => {
    mock.installed.value = [makeModel()];
    const w = mountPage();
    await w.find("[data-test=installed-search]").setValue("zzz");
    expect(w.find("[data-test=installed-no-match]").exists()).toBe(true);
    await w.find("[data-test=clear-search]").trigger("click");
    expect(w.findAllComponents(InstalledModelRow)).toHaveLength(1);
  });

  it("shows the nothing-installed empty state with a Discover CTA", async () => {
    mock.installed.value = [];
    const w = mountPage();
    expect(w.find("[data-test=installed-empty]").exists()).toBe(true);
    await w.find("[data-test=discover-cta]").trigger("click");
    expect(mock.setTab).toHaveBeenCalledWith("discover");
  });

  it("renders the discover tab when the active tab is discover", () => {
    mock.tab.value = "discover";
    const w = mountPage();
    expect(w.find("[data-test=discover-tab]").exists()).toBe(true);
    expect(w.find("[data-test=installed-tab]").exists()).toBe(false);
  });

  it("opens sequence discovery on the Discover tab with video checkpoints filtered", () => {
    routeQuery.value = {
      tab: "discover",
      type: "video",
      kind: "checkpoint",
      intent: "sequence",
    };
    const w = mountPage();

    expect(mock.setTab).toHaveBeenCalledWith("discover");
    expect(mock.setFilter).toHaveBeenCalledWith({
      modality: "video",
      kind: "checkpoint",
    });
    expect(w.get("[data-test='sequence-model-guide']").text()).toContain(
      "sequence",
    );
  });
});

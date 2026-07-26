import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import ModelsPage from "./ModelsPage.vue";
import type { ModelInfoExtended } from "../types";
import InstalledModelRow from "../components/models/InstalledModelRow.vue";

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
  routeQuery.value = {};
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

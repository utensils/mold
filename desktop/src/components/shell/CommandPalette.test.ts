import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const routerPush = vi.hoisted(() => vi.fn());
const searchCatalogMock = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ entries: [], page: 1, page_size: 12, total: 0 }),
);
const startCatalogDownloadMock = vi.hoisted(() => vi.fn().mockResolvedValue("job-1"));
vi.mock("vue-router", () => ({ useRouter: () => ({ push: routerPush }) }));
vi.mock("../../lib/api/history", () => ({ fetchHistory: vi.fn().mockResolvedValue([]) }));
vi.mock("../../lib/api/models", () => ({ loadModel: vi.fn(), unloadModel: vi.fn() }));
vi.mock("../../lib/api/catalog", () => ({
  searchCatalog: searchCatalogMock,
  startCatalogDownload: startCatalogDownloadMock,
}));

import CommandPalette from "./CommandPalette.vue";
import { useGalleryStore } from "../../stores/gallery";
import { useUiStore } from "../../stores/ui";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useConnectionStore } from "../../stores/connection";
import { useDownloadsStore } from "../../stores/downloads";
import { useComposerStore } from "../../stores/composer";
import type { GalleryImage, ModelEntry } from "../../lib/api/types";

beforeEach(() => {
  setActivePinia(createPinia());
  routerPush.mockClear();
  searchCatalogMock.mockClear();
  searchCatalogMock.mockResolvedValue({ entries: [], page: 1, page_size: 12, total: 0 });
  startCatalogDownloadMock.mockClear();
  startCatalogDownloadMock.mockResolvedValue("job-1");
});

async function openPalette() {
  const wrapper = mount(CommandPalette, { attachTo: document.body });
  const ui = useUiStore();
  ui.paletteOpen = true;
  await wrapper.vm.$nextTick();
  await wrapper.vm.$nextTick();
  return wrapper;
}

describe("CommandPalette command registry", () => {
  it("navigates to Models for both 'models' and 'catalog' queries", async () => {
    const wrapper = await openPalette();
    const input = wrapper.get("input");

    await input.setValue("models");
    let texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Go to Models"))).toBe(true);

    // Muscle memory: the old "catalog" name still finds the Models entry.
    await input.setValue("catalog");
    texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Go to Models"))).toBe(true);
    expect(texts.some((t) => t.includes("Go to Catalog"))).toBe(false);
    wrapper.unmount();
  });

  it("offers Create sequence and Open history alongside the five workspaces", async () => {
    const wrapper = await openPalette();
    const input = wrapper.get("input");

    // "clips" and "video" both find the sequence entry (mode is a setting of
    // Create now, so the entry deep-links rather than opening a chains page).
    await input.setValue("clips");
    let texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Create sequence"))).toBe(true);

    await input.setValue("history");
    texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Open history"))).toBe(true);

    // Picking the entry deep-links into Create's sequence output (and closes).
    await input.setValue("sequence");
    const option = wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Create sequence"));
    expect(option).toBeTruthy();
    await option!.trigger("click");
    expect(routerPush).toHaveBeenCalledWith("/create?output=sequence");
    wrapper.unmount();
  });

  it("offers theme + appearance commands wired to the shared prefs plumbing", async () => {
    const wrapper = await openPalette();
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue();

    await wrapper.get("input").setValue("theme");
    const options = wrapper.findAll("[role='option']");
    const texts = options.map((o) => o.text());
    expect(texts.some((t) => t.includes("Theme: Mold"))).toBe(true);
    expect(texts.some((t) => t.includes("Theme: Safelight"))).toBe(true);
    expect(texts.some((t) => t.includes("Appearance: Dark"))).toBe(true);

    await options.find((o) => o.text().includes("Theme: Safelight"))!.trigger("click");
    expect(update).toHaveBeenCalledWith({ themeFamily: "safelight" });
    wrapper.unmount();
  });

  it("no longer offers the retired 'Switch to built-in engine' command", async () => {
    // The built-in engine is always the primary now — there is no remote
    // primary to switch away from; recovery is "Restart engine".
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("engine");
    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Switch to built-in engine"))).toBe(false);
    expect(texts.some((t) => t.includes("Restart engine"))).toBe(true);
    wrapper.unmount();
  });
});

describe("CommandPalette gallery results", () => {
  const print = (filename: string, prompt: string, model: string): GalleryImage =>
    ({ filename, timestamp: 1, metadata: { prompt, model, seed: 1 } }) as never;

  it("surfaces matching prints and deep-links to their lightbox", async () => {
    const wrapper = await openPalette();
    useGalleryStore().buckets["local"] = {
      items: [
        print("mold-flux-1.png", "a paper plane at dawn", "flux-dev:q8"),
        print("other.png", "a cat", "sd15:fp16"),
      ],
      loading: false,
      error: null,
      loaded: true,
    };
    await wrapper.get("input").setValue("plane");
    await wrapper.vm.$nextTick();

    const options = wrapper.findAll("[role='option']");
    const match = options.find((o) => o.text().includes("a paper plane at dawn"));
    expect(match).toBeDefined();
    expect(match!.text()).toContain("library");
    expect(options.some((o) => o.text().includes("a cat"))).toBe(false);

    await match!.trigger("click");
    expect(routerPush).toHaveBeenCalledWith("/library?print=mold-flux-1.png");
    wrapper.unmount();
  });

  it("offers no print rows for a blank query", async () => {
    const wrapper = await openPalette();
    useGalleryStore().buckets["local"] = {
      items: [print("mold-flux-1.png", "a paper plane at dawn", "flux-dev:q8")],
      loading: false,
      error: null,
      loaded: true,
    };
    await wrapper.vm.$nextTick();
    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("a paper plane at dawn"))).toBe(false);
    wrapper.unmount();
  });
});

describe("CommandPalette model search", () => {
  const model = (name: string, family: string): ModelEntry =>
    ({
      name,
      family,
      downloaded: true,
      is_loaded: false,
      description: "",
      hf_repo: "",
      size_gb: 1,
      default_steps: 20,
      default_guidance: 3.5,
      default_width: 1024,
      default_height: 1024,
    }) as never;

  /** A ready local primary plus one ready remote machine. */
  function seedFleet() {
    useConnectionStore().info = { baseUrl: "http://127.0.0.1:7680", apiKey: null } as never;
    useConnectionStore().status = "ready";
    useHostsStore().extras.push({
      id: "bender-7680",
      label: "bender",
      url: "http://bender:7680",
      apiKey: "bk",
      status: "ready",
      error: null,
      instanceId: null,
    });
  }

  it("offers a Use row for a model only another machine has", async () => {
    seedFleet();
    // Pinned to this Mac: Auto is already model-aware and implies no move.
    useAppPrefsStore().settings = { generateTargetHost: "local" } as never;
    useHostModelsStore().byHost["bender-7680"] = {
      entries: [model("ltx2-distilled", "ltx2")],
      fetchedAt: 1,
      error: null,
    };
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("ltx2");

    const row = wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Use ltx2-distilled"));
    expect(row).toBeDefined();
    expect(row!.text()).toContain("on bender");
    wrapper.unmount();
  });

  it("repins the generation target when using a model from another machine", async () => {
    seedFleet();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "local" } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue();
    useHostModelsStore().byHost["bender-7680"] = {
      entries: [model("ltx2-distilled", "ltx2")],
      fetchedAt: 1,
      error: null,
    };
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("ltx2");
    await wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Use ltx2-distilled"))!
      .trigger("click");

    expect(update).toHaveBeenCalledWith({ generateTargetHost: "bender-7680" });
    // The remote model's own defaults travel with it into the composer — a
    // model the primary has never seen must not fall back to a default.
    expect(useComposerStore().prefill).toMatchObject({ model: "ltx2-distilled" });
    expect(routerPush).toHaveBeenCalledWith("/create");
    wrapper.unmount();
  });

  it("leaves the target alone when the pinned machine already has the model", async () => {
    seedFleet();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "local" } as never;
    const update = vi.spyOn(prefs, "update").mockResolvedValue();
    useModelStore().all = [model("flux-dev:q4", "flux")];
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("flux");
    await wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Use flux-dev:q4"))!
      .trigger("click");

    expect(update).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("searches the catalog only once the query is long enough", async () => {
    vi.useFakeTimers();
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("q");
    await vi.advanceTimersByTimeAsync(300);
    expect(searchCatalogMock).not.toHaveBeenCalled();

    await wrapper.get("input").setValue("qwen");
    await vi.advanceTimersByTimeAsync(300);
    expect(searchCatalogMock).toHaveBeenCalledWith({
      q: "qwen",
      kind: "checkpoint",
      page_size: 12,
    });
    vi.useRealTimers();
    wrapper.unmount();
  });

  it("commits the repin before navigating to Create", async () => {
    seedFleet();
    const prefs = useAppPrefsStore();
    prefs.settings = { generateTargetHost: "local" } as never;
    // `appPrefs.update` re-reads and rewrites the whole settings file; Create's
    // own last-route write must not interleave with it and restore the old pin.
    const order: string[] = [];
    let releaseUpdate: () => void = () => {};
    vi.spyOn(prefs, "update").mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          releaseUpdate = () => {
            order.push("update");
            resolve();
          };
        }),
    );
    routerPush.mockImplementation(() => {
      order.push("navigate");
    });
    useHostModelsStore().byHost["bender-7680"] = {
      entries: [model("ltx2-distilled", "ltx2")],
      fetchedAt: 1,
      error: null,
    };

    const wrapper = await openPalette();
    await wrapper.get("input").setValue("ltx2");
    await wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Use ltx2-distilled"))!
      .trigger("click");

    expect(order).toEqual([]);
    releaseUpdate();
    await flushPromises();
    expect(order).toEqual(["update", "navigate"]);
    routerPush.mockImplementation(() => {});
    wrapper.unmount();
  });

  it("ignores an unreachable machine's cached inventory", async () => {
    seedFleet();
    useHostsStore().telemetry["bender-7680"] = { status: "error" } as never;
    const bender = useHostsStore().extras.find((h) => h.id === "bender-7680")!;
    bender.status = "error";
    // The host dropped offline but its last model list is still cached.
    useHostModelsStore().byHost["bender-7680"] = {
      entries: [model("ltx2-distilled", "ltx2")],
      fetchedAt: 1,
      error: "offline",
    };
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("ltx2");

    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Use ltx2-distilled"))).toBe(false);
    wrapper.unmount();
  });

  it("drops the previous query's install rows the moment a new query starts", async () => {
    vi.useFakeTimers();
    seedFleet();
    searchCatalogMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/qwen",
          name: "Qwen Image",
          family: "qwen-image",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("qwen");
    await vi.advanceTimersByTimeAsync(300);
    expect(
      wrapper.findAll("[role='option']").some((o) => o.text().includes("Install Qwen Image")),
    ).toBe(true);

    // Mid-flight for the NEW query the stale row must already be gone —
    // otherwise Enter here queues a model the user is no longer looking at.
    await wrapper.get("input").setValue("wuerstchen");
    expect(
      wrapper.findAll("[role='option']").some((o) => o.text().includes("Install Qwen Image")),
    ).toBe(false);
    vi.useRealTimers();
    wrapper.unmount();
  });

  it("disarms a pending search when the palette unmounts mid-keystroke", async () => {
    vi.useFakeTimers();
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("qwen");
    wrapper.unmount();
    await vi.advanceTimersByTimeAsync(500);

    // A request whose response nothing can consume must never be issued.
    expect(searchCatalogMock).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("offers an install row for a catalog model nobody has, and pulls it", async () => {
    vi.useFakeTimers();
    seedFleet();
    // Both machines' inventories are known and neither holds the model.
    useModelStore().all = [model("flux-dev:q4", "flux")];
    useHostModelsStore().byHost["bender-7680"] = { entries: [], fetchedAt: 1, error: null };
    searchCatalogMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/qwen",
          name: "Qwen Image",
          family: "qwen-image",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const downloads = useDownloadsStore();
    const subscribe = vi.spyOn(downloads, "subscribe").mockResolvedValue();

    const wrapper = await openPalette();
    await wrapper.get("input").setValue("qwen");
    await vi.advanceTimersByTimeAsync(300);

    const row = wrapper
      .findAll("[role='option']")
      .find((o) => o.text().includes("Install Qwen Image"));
    expect(row).toBeDefined();
    expect(row!.text()).toContain("not installed · hf");

    await row!.trigger("click");
    await vi.advanceTimersByTimeAsync(0);

    // Stream attached before the POST so a cached pull still shows a terminal event.
    expect(subscribe).toHaveBeenCalled();
    // This Mac is the plan's first install target, addressed explicitly (same
    // shape the Models workspace sends) with no credential forwarding.
    expect(startCatalogDownloadMock).toHaveBeenCalledWith(
      "hf:org/qwen",
      { baseUrl: "http://127.0.0.1:7680", apiKey: null },
      false,
    );
    vi.useRealTimers();
    wrapper.unmount();
  });

  it("never offers to install a model the fleet already has", async () => {
    vi.useFakeTimers();
    seedFleet();
    useModelStore().all = [model("flux-dev:q4", "flux")];
    searchCatalogMock.mockResolvedValue({
      entries: [
        {
          id: "flux-dev:q4",
          name: "FLUX Dev",
          family: "flux",
          source: "hf",
          installed: false,
          supported: true,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("flux");
    await vi.advanceTimersByTimeAsync(300);

    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Install"))).toBe(false);
    vi.useRealTimers();
    wrapper.unmount();
  });

  it("drops unsupported catalog entries", async () => {
    vi.useFakeTimers();
    seedFleet();
    searchCatalogMock.mockResolvedValue({
      entries: [
        {
          id: "hf:org/broken",
          name: "Broken",
          family: "x",
          source: "hf",
          installed: false,
          supported: false,
        },
      ],
      page: 1,
      page_size: 12,
      total: 1,
    });
    const wrapper = await openPalette();
    await wrapper.get("input").setValue("broken");
    await vi.advanceTimersByTimeAsync(300);

    const texts = wrapper.findAll("[role='option']").map((o) => o.text());
    expect(texts.some((t) => t.includes("Install Broken"))).toBe(false);
    vi.useRealTimers();
    wrapper.unmount();
  });
});

describe("CommandPalette a11y semantics", () => {
  it("is a modal dialog wrapping a combobox and listbox", async () => {
    const wrapper = await openPalette();

    const dialog = wrapper.get("[role='dialog']");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Command palette");

    const combobox = wrapper.get("[role='combobox']");
    expect(combobox.attributes("aria-controls")).toBe("cmd-palette-listbox");
    expect(combobox.attributes("aria-expanded")).toBe("true");

    const listbox = wrapper.get("#cmd-palette-listbox");
    expect(listbox.attributes("role")).toBe("listbox");
    wrapper.unmount();
  });

  it("marks options with role=option and points aria-activedescendant at the selection", async () => {
    const wrapper = await openPalette();

    const options = wrapper.findAll("[role='option']");
    expect(options.length).toBeGreaterThan(0);
    expect(options[0]!.attributes("aria-selected")).toBe("true");
    expect(options[0]!.attributes("id")).toBe("cmd-palette-option-0");

    // The combobox's active descendant tracks the highlighted option.
    expect(wrapper.get("[role='combobox']").attributes("aria-activedescendant")).toBe(
      "cmd-palette-option-0",
    );
    wrapper.unmount();
  });
});

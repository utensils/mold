import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const fetchHistoryAll = vi
  .fn()
  .mockResolvedValue({ entries: [], supportedHostIds: [], unreachableHostIds: [] });
const clearHistoryOn = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/api/history", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api/history")>();
  return {
    ...actual,
    fetchHistory: vi.fn().mockResolvedValue([]),
    fetchHistoryAll: (...a: unknown[]) => fetchHistoryAll(...a),
    clearHistory: vi.fn().mockResolvedValue(undefined),
    clearHistoryOn: (...a: unknown[]) => clearHistoryOn(...a),
  };
});
// The run rows read their bytes through the same origin-aware helper the
// Library tiles do, so "Use as source" needs a real Response here.
const apiFetchTo = vi.hoisted(() => vi.fn().mockResolvedValue(new Response()));
vi.mock("../../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiFetch: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiJson: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiFetchTo,
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { localGalleryList: vi.fn().mockResolvedValue({ images: [], target: null }) },
}));
// "Use these settings" asks the producing host what source media it kept.
const retainedInventory = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ availability: "available", members: [] }),
);
vi.mock("@studio/api/gallerySourceMedia", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@studio/api/gallerySourceMedia")>();
  return { ...actual, retainedSourceMediaInventory: retainedInventory };
});

import HistoryDrawer from "./HistoryDrawer.vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";
import { useConnectionStore } from "../../stores/connection";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore } from "../../stores/hosts";
import { useComposerStore } from "../../stores/composer";
import { useContextMenuStore } from "../../stores/contextMenu";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useAppPrefsStore } from "../../stores/appPrefs";
import type { GalleryImage } from "../../lib/api/types";
import { HISTORY_JOBS_RENDER_CAP } from "@studio/lib/activity";

const stub = { template: "<div />" };

function run(filename: string, prompt: string, timestamp: number): GalleryImage {
  return {
    filename,
    timestamp,
    metadata: {
      prompt,
      model: "flux2-klein",
      seed: 42,
      steps: 4,
      guidance: 1,
      width: 1024,
      height: 768,
    },
  };
}

let router: Router;

async function mountDrawer({ extra = false, at = "/library" } = {}) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/create", component: stub },
      { path: "/library", component: stub },
    ],
  });
  router.push(at);
  await router.isReady();
  const pinia = createPinia();
  setActivePinia(pinia);
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://x", apiKey: null };
  conn.status = "ready";
  const gallery = useGalleryStore();
  // Local primary → one bucket, key "local" (the built-in engine's host id).
  gallery.buckets["local"] = {
    items: [
      run("a.png", "a lighthouse at dusk", 1_700_000_100),
      run("b.png", "a stoic owl", 1_700_000_000),
    ],
    loading: false,
    error: null,
    loaded: true,
  };
  if (extra) {
    useHostsStore().extras.push({
      id: "okra-7680",
      label: "okra",
      url: "http://okra:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    gallery.buckets["okra-7680"] = {
      items: [run("o.png", "a remote heron", 1_700_000_050)],
      loading: false,
      error: null,
      loaded: true,
    };
  }
  const wrapper = mount(HistoryDrawer, {
    props: { open: true },
    attachTo: document.body,
    global: { plugins: [pinia, router], stubs: { AuthedMedia: stub } },
  });
  await flushPromises();
  return wrapper;
}

const promptEntry = (prompt: string, hostId: string, hostLabel: string, used_at: number) => ({
  prompt,
  model: "flux2-klein",
  used_at,
  hostId,
  hostLabel,
});

beforeEach(() => {
  vi.clearAllMocks();
  fetchHistoryAll.mockResolvedValue({ entries: [], supportedHostIds: [], unreachableHostIds: [] });
  clearHistoryOn.mockResolvedValue(undefined);
});

describe("HistoryDrawer runs", () => {
  it("is an inline column, never a modal drawer over the grid", async () => {
    const wrapper = await mountDrawer();
    // A scrim with `aria-modal` made every tile behind it unclickable; the
    // panel is a plain flex sibling of the grid instead.
    expect(wrapper.findComponent(DrawerPanel).exists()).toBe(false);
    const panel = wrapper.get("[data-test='history-panel']");
    expect(panel.attributes("aria-modal")).toBeUndefined();
    expect(panel.attributes("role")).toBeUndefined();
    expect(panel.attributes("aria-label")).toBe("History");
    expect(wrapper.find(".ms-drawer").exists()).toBe(false);
  });

  it("opens at the mock's column width and persists left-edge resizing or a reset", async () => {
    const wrapper = await mountDrawer();
    const prefs = useAppPrefsStore();
    const update = vi.spyOn(prefs, "update").mockResolvedValue(undefined);
    const width = () =>
      (wrapper.get("[data-test='history-panel']").element as HTMLElement).style.width;
    expect(width()).toBe("290px");

    const handle = wrapper.getComponent(PanelResizeHandle);
    expect(handle.props("label")).toBe("Resize history");
    handle.vm.$emit("resize", -80);
    await flushPromises();
    expect(width()).toBe("370px");
    handle.vm.$emit("commit");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ historyDrawerWidth: 370 });

    handle.vm.$emit("reset");
    await flushPromises();
    expect(update).toHaveBeenCalledWith({ historyDrawerWidth: null });
  });

  it("shows past runs with thumbnails and full metadata", async () => {
    const wrapper = await mountDrawer();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("a lighthouse at dusk");
    expect(rows[0]!.text()).toContain("flux2-klein");
    expect(rows[0]!.text()).toContain("1024×768");
    expect(rows[0]!.text()).toContain("seed 42");
    // The mock's meta line folds the clock in and drops the pass count.
    expect(rows[0]!.get("[data-test='run-meta']").text()).toMatch(
      /^flux2-klein · 1024×768 · seed 42 · \d{1,2}:\d{2}/,
    );
    expect(rows[0]!.text()).toContain("Use these settings");
  });

  it("reuses a run down the retained-source road, print and all", async () => {
    const wrapper = await mountDrawer();
    await wrapper.get("[data-test='run-row']").trigger("click");
    await flushPromises();
    const composer = useComposerStore();
    // A bare `composer.set` restored the numbers and dropped the photo the
    // print was made from — this is the same road the Lightbox takes.
    expect(composer.prefill).toEqual({
      metadata: expect.objectContaining({
        prompt: "a lighthouse at dusk",
        seed: 42,
        width: 1024,
        height: 768,
      }),
      print: expect.objectContaining({ filename: "a.png", hostId: null }),
    });
    expect(retainedInventory).toHaveBeenCalledWith("a.png", expect.anything());
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("caps the Runs list at the History render cap and says so", async () => {
    const wrapper = await mountDrawer();
    const gallery = useGalleryStore();
    const many: GalleryImage[] = [];
    for (let i = 0; i < HISTORY_JOBS_RENDER_CAP + 50; i++) {
      many.push(run(`bulk-${i}.png`, `bulk print ${i}`, 1_700_100_000 - i));
    }
    gallery.buckets["local"]!.items = many;
    await flushPromises();
    expect(wrapper.findAll('[data-test="run-row"]').length).toBe(HISTORY_JOBS_RENDER_CAP);
    expect(wrapper.get('[data-test="runs-cap-note"]').text()).toContain(
      `showing ${HISTORY_JOBS_RENDER_CAP} of ${HISTORY_JOBS_RENDER_CAP + 50}`,
    );
  });

  it("filters runs by prompt text", async () => {
    const wrapper = await mountDrawer();
    await wrapper.get("input[type='search']").setValue("owl");
    await flushPromises();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("a stoic owl");
  });

  // Every desktop surface that shows a print offers the same right-click
  // actions on it. A History run is a print, so its menu carries "Use as
  // source" and attaches through the one shared rule.
  it("offers a past run as the next render's source", async () => {
    apiFetchTo.mockResolvedValue(new Response(new Uint8Array([65, 66, 67])));
    const wrapper = await mountDrawer();
    const menu = useContextMenuStore();

    await wrapper.get("[data-test='run-row']").trigger("contextmenu");
    const entries = menu.entries;
    expect(entries.map((e) => ("separator" in e ? "—" : e.label))).toContain("Use as source");
    const source = entries.find((e) => !("separator" in e) && e.label === "Use as source")!;
    expect(source).toMatchObject({ disabled: false });
    menu.activate(source);
    await flushPromises();

    const form = useGenerateFormStore().form;
    expect(form.sourceImage).toBe("QUJD");
    expect(form.sourceImageName).toBe("a.png");
    expect(router.currentRoute.value.path).toBe("/create");
    wrapper.unmount();
  });

  it("attaches a run's video as source video and refuses an audio run", async () => {
    apiFetchTo.mockResolvedValue(
      new Response(new Uint8Array([65, 66, 67]), {
        headers: { "Content-Type": "video/mp4" },
      }),
    );
    const wrapper = await mountDrawer();
    const gallery = useGalleryStore();
    gallery.buckets["local"]!.items = [
      { ...run("clip.mp4", "a drifting clip", 1_700_000_200), format: "mp4" },
      { ...run("score.wav", "a slow score", 1_700_000_150), format: "wav" },
    ];
    await flushPromises();
    const menu = useContextMenuStore();

    const rows = wrapper.findAll("[data-test='run-row']");
    await rows[0]!.trigger("contextmenu");
    const video = menu.entries.find((e) => !("separator" in e) && e.label === "Use as source")!;
    menu.activate(video);
    await flushPromises();
    expect(useGenerateFormStore().form.sourceVideo).toMatchObject({ filename: "clip.mp4" });

    await rows[1]!.trigger("contextmenu");
    const audio = menu.entries.find((e) => !("separator" in e) && e.label === "Use as source")!;
    expect(audio).toMatchObject({ disabled: true });
    wrapper.unmount();
  });

  // A mesh is geometry: reading it and handing it to the image attach would
  // stage binary glTF as conditioning.
  it("refuses a mesh run as a source", async () => {
    const wrapper = await mountDrawer();
    const gallery = useGalleryStore();
    gallery.buckets["local"]!.items = [
      { ...run("bust.glb", "a marble bust", 1_700_000_300), format: "glb" },
    ];
    await flushPromises();
    const menu = useContextMenuStore();

    await wrapper.get("[data-test='run-row']").trigger("contextmenu");
    const source = menu.entries.find((e) => !("separator" in e) && e.label === "Use as source")!;
    expect(source).toMatchObject({ disabled: true });
    expect(apiFetchTo).not.toHaveBeenCalled();
    wrapper.unmount();
  });

  it("loads the prompt log only when that tab is opened", async () => {
    const wrapper = await mountDrawer();
    expect(fetchHistoryAll).not.toHaveBeenCalled();
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(fetchHistoryAll).toHaveBeenCalled();
  });
});

describe("HistoryDrawer multi-host", () => {
  it("shows no host chips with a single source", async () => {
    const wrapper = await mountDrawer();
    expect(wrapper.find("[role='tablist']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='host-badge']")).toHaveLength(0);
  });

  it("shows filter chips and per-row host chips when several hosts are live", async () => {
    const wrapper = await mountDrawer({ extra: true });
    expect(wrapper.find("[role='tablist']").exists()).toBe(true);
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(3);
    const badges = wrapper.findAll("[data-test='host-badge']");
    expect(badges.map((b) => b.text())).toEqual(["This device", "okra", "This device"]);
  });

  it("the chip filter narrows the runs list and hides row chips", async () => {
    const wrapper = await mountDrawer({ extra: true });
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("a remote heron");
    expect(wrapper.findAll("[data-test='host-badge']")).toHaveLength(0);
  });

  it("fans the prompt log out over every ready host and tags rows", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    const targets = fetchHistoryAll.mock.calls[0]![0] as Array<{ hostId: string }>;
    expect(targets.map((t) => t.hostId)).toEqual(["local", "okra-7680"]);
    const rows = wrapper.findAll("[data-test='prompt-row']");
    expect(rows).toHaveLength(2);
    const badges = wrapper.findAll("[data-test='host-badge']");
    expect(badges.map((b) => b.text())).toEqual(["This Mac", "okra"]);
  });

  it("the chip filter narrows the prompt log too", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    const rows = wrapper.findAll("[data-test='prompt-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("remote prompt");
  });

  it("shows the unavailable state only when no host supports history", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [],
      supportedHostIds: [],
      unreachableHostIds: [],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Prompt history isn't available");

    fetchHistoryAll.mockResolvedValue({
      entries: [promptEntry("kept", "local", "This Mac", 1_000)],
      supportedHostIds: ["local"],
      unreachableHostIds: [],
    });
    const other = await mountDrawer({ extra: true });
    await other.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(other.text()).not.toContain("Prompt history isn't available");
  });

  it("clears only the filtered host", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    useGalleryStore().filter = "okra-7680";
    await flushPromises();
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await flushPromises();
    expect(clearHistoryOn).toHaveBeenCalledTimes(1);
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://okra:7680", apiKey: null });
  });

  it("clearing All names every history-capable host and hits them all", async () => {
    fetchHistoryAll.mockResolvedValue({
      entries: [
        promptEntry("local prompt", "local", "This Mac", 2_000),
        promptEntry("remote prompt", "okra-7680", "okra", 1_000),
      ],
      supportedHostIds: ["local", "okra-7680"],
    });
    const wrapper = await mountDrawer({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='clear-history']").trigger("click");
    const label = wrapper.get("[data-test='clear-history']").text();
    expect(label).toContain("This device");
    expect(label).toContain("okra");
    await wrapper.get("[data-test='clear-history']").trigger("click");
    await flushPromises();
    expect(clearHistoryOn).toHaveBeenCalledTimes(2);
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://x", apiKey: null });
    expect(clearHistoryOn).toHaveBeenCalledWith({ baseUrl: "http://okra:7680", apiKey: null });
  });
});

// ── Two lenses, and no third ───────────────────────────────────────────────
// Scene-by-scene authoring is retired, so the drawer holds Runs and Prompts
// and nothing else. A bookmarked `?tab=sequences` from the retired log must
// land on Runs rather than render an empty column.

describe("HistoryDrawer tabs", () => {
  it("offers exactly two tabs", async () => {
    const wrapper = await mountDrawer();
    const tabs = wrapper.findAll("[data-test^='tab-']");
    expect(tabs.map((t) => t.text())).toEqual(["Runs", "Prompts"]);
    expect(wrapper.find("[data-test='tab-sequences']").exists()).toBe(false);
  });

  it("normalizes a retired sequence tab to Runs", async () => {
    const wrapper = await mountDrawer({ at: "/library?panel=history&tab=sequences" });
    expect(wrapper.get("[data-test='tab-runs']").attributes("aria-pressed")).toBe("true");
    expect(wrapper.findAll('[data-test="run-row"]').length).toBeGreaterThan(0);
  });

  it("writes the tab back on switch", async () => {
    const wrapper = await mountDrawer({ at: "/library?panel=history&tab=prompts" });
    expect(wrapper.get("[data-test='tab-prompts']").attributes("aria-pressed")).toBe("true");
    await wrapper.get("[data-test='tab-runs']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.tab).toBe("runs");
  });
});

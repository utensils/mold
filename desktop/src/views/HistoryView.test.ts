import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const fetchHistoryAll = vi
  .fn()
  .mockResolvedValue({ entries: [], supportedHostIds: [], unreachableHostIds: [] });
const clearHistoryOn = vi.fn().mockResolvedValue(undefined);
vi.mock("../lib/api/history", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api/history")>();
  return {
    ...actual,
    fetchHistory: vi.fn().mockResolvedValue([]),
    fetchHistoryAll: (...a: unknown[]) => fetchHistoryAll(...a),
    clearHistory: vi.fn().mockResolvedValue(undefined),
    clearHistoryOn: (...a: unknown[]) => clearHistoryOn(...a),
  };
});
vi.mock("../lib/api/client", () => ({
  ApiError: class ApiError extends Error {},
  apiFetch: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiJson: vi.fn().mockRejectedValue(new Error("no network in tests")),
  apiFetchTo: vi.fn(),
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../lib/api/sse", () => ({ sseStream: vi.fn() }));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { localGalleryList: vi.fn().mockResolvedValue([]) },
}));

import HistoryView from "./HistoryView.vue";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useComposerStore } from "../stores/composer";
import type { GalleryImage } from "../lib/api/types";

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

async function mountView({ extra = false } = {}) {
  router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", component: stub },
      { path: "/generate", component: stub },
      { path: "/gallery", component: stub },
    ],
  });
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
    });
    gallery.buckets["okra-7680"] = {
      items: [run("o.png", "a remote heron", 1_700_000_050)],
      loading: false,
      error: null,
      loaded: true,
    };
  }
  const wrapper = mount(HistoryView, {
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

describe("HistoryView runs", () => {
  it("shows past runs with thumbnails and full metadata", async () => {
    const wrapper = await mountView();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(2);
    expect(rows[0]!.text()).toContain("a lighthouse at dusk");
    expect(rows[0]!.text()).toContain("flux2-klein");
    expect(rows[0]!.text()).toContain("1024×768");
    expect(rows[0]!.text()).toContain("S 42");
    expect(rows[0]!.text()).toContain("4 steps");
  });

  it("reuses a run's full settings including the seed", async () => {
    const wrapper = await mountView();
    await wrapper.get("[data-test='run-row']").trigger("click");
    await flushPromises();
    const composer = useComposerStore();
    expect(composer.prefill).toMatchObject({
      prompt: "a lighthouse at dusk",
      seed: 42,
      width: 1024,
      height: 768,
    });
    expect(router.currentRoute.value.path).toBe("/generate");
  });

  it("filters runs by prompt text", async () => {
    const wrapper = await mountView();
    await wrapper.get("input[type='search']").setValue("owl");
    await flushPromises();
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("a stoic owl");
  });

  it("loads the prompt log only when that tab is opened", async () => {
    const wrapper = await mountView();
    expect(fetchHistoryAll).not.toHaveBeenCalled();
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(fetchHistoryAll).toHaveBeenCalled();
  });
});

describe("HistoryView multi-host", () => {
  it("shows no host chips with a single source", async () => {
    const wrapper = await mountView();
    expect(wrapper.find("[role='tablist']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='host-badge']")).toHaveLength(0);
  });

  it("shows filter chips and per-row host chips when several hosts are live", async () => {
    const wrapper = await mountView({ extra: true });
    expect(wrapper.find("[role='tablist']").exists()).toBe(true);
    const rows = wrapper.findAll("[data-test='run-row']");
    expect(rows).toHaveLength(3);
    const badges = wrapper.findAll("[data-test='host-badge']");
    expect(badges.map((b) => b.text())).toEqual(["This device", "okra", "This device"]);
  });

  it("the chip filter narrows the runs list and hides row chips", async () => {
    const wrapper = await mountView({ extra: true });
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
    const wrapper = await mountView({ extra: true });
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
    const wrapper = await mountView({ extra: true });
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
    const wrapper = await mountView({ extra: true });
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(wrapper.text()).toContain("Prompt history isn't available");

    fetchHistoryAll.mockResolvedValue({
      entries: [promptEntry("kept", "local", "This Mac", 1_000)],
      supportedHostIds: ["local"],
      unreachableHostIds: [],
    });
    const other = await mountView({ extra: true });
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
    const wrapper = await mountView({ extra: true });
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
    const wrapper = await mountView({ extra: true });
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

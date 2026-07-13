import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";

const fetchHistory = vi.fn().mockResolvedValue([]);
vi.mock("../lib/api/history", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api/history")>();
  return {
    ...actual,
    fetchHistory: (...a: unknown[]) => fetchHistory(...a),
    clearHistory: vi.fn().mockResolvedValue(undefined),
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

async function mountView() {
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
  const wrapper = mount(HistoryView, {
    global: { plugins: [pinia, router], stubs: { AuthedMedia: stub } },
  });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  fetchHistory.mockResolvedValue([]);
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
    expect(fetchHistory).not.toHaveBeenCalled();
    await wrapper.get("[data-test='tab-prompts']").trigger("click");
    await flushPromises();
    expect(fetchHistory).toHaveBeenCalled();
  });
});

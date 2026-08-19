import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const { apiFetchTo, localGalleryList } = vi.hoisted(() => ({
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  localGalleryList: vi.fn(),
}));

vi.mock("@tanstack/vue-virtual", () => ({
  useVirtualizer: (options: { value: { count: number } }) => ({
    getTotalSize: () => options.value.count * 188,
    getVirtualItems: () =>
      Array.from({ length: options.value.count }, (_, index) => ({
        index,
        key: index,
        start: index * 188,
      })),
  }),
}));
vi.mock("../lib/gallery/layout", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/gallery/layout")>();
  return {
    ...actual,
    layoutJustifiedRows: (items: GalleryImage[], _containerWidth: number, targetHeight: number) =>
      items.map((item) => ({
        height: targetHeight,
        items: [{ item, width: targetHeight, height: targetHeight }],
      })),
  };
});
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo,
  apiJsonTo: vi.fn(),
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => true,
  ipc: {
    localGalleryDelete: vi.fn(),
    localGalleryList,
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));

import LibraryView from "./LibraryView.vue";
import { useConnectionStore } from "../stores/connection";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import type { GalleryImage } from "../lib/api/types";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

const localPrint: GalleryImage = {
  filename: "local.png",
  timestamp: 2,
  metadata: {
    prompt: "local",
    model: "flux-dev:q8",
    seed: 1,
    steps: 4,
    guidance: 1,
    width: 1024,
    height: 1024,
  },
};
const remotePrint: GalleryImage = {
  filename: "remote.png",
  timestamp: 1,
  metadata: {
    prompt: "remote",
    model: "flux-dev:q8",
    seed: 2,
    steps: 4,
    guidance: 1,
    width: 1024,
    height: 1024,
  },
};

const stub = { template: "<div />" };

async function mountView() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
    ],
  });
  await router.push("/library");
  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = null;
  connection.status = "error";
  const gallery = useGalleryStore();
  gallery.buckets.local = { items: [localPrint], loading: false, error: null, loaded: true };
  useHostsStore().extras.push({
    id: "hal9000-7680",
    label: "hal9000",
    url: "http://hal9000:7680",
    apiKey: null,
    status: "ready",
    error: null,
    instanceId: null,
  });
  gallery.buckets["hal9000-7680"] = {
    items: [remotePrint],
    loading: false,
    error: null,
    loaded: true,
  };
  localGalleryList.mockResolvedValue({ images: [localPrint], target: null });
  const wrapper = mount(LibraryView, {
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: stub, HostFilterChips: stub, HistoryDrawer: stub },
    },
  });
  await flushPromises();
  return { wrapper, gallery };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("LibraryView remote-only prints", () => {
  it("opens the lightbox for a remote-only print on double-click", async () => {
    const { wrapper } = await mountView();
    const tiles = wrapper.findAll(".ms-lib-tile");
    expect(tiles.length).toBe(2);
    const remoteTile = tiles.find((t) => t.text().includes("hal9000"));
    expect(remoteTile).toBeTruthy();
    await remoteTile!.trigger("dblclick");
    await flushPromises();
    const lightbox = wrapper.findComponent({ name: "Lightbox" });
    expect(lightbox.exists()).toBe(true);
    expect(lightbox.props("item").filename).toBe("remote.png");
  });
});

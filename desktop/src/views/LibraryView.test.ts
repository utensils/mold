import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const { apiFetchTo, localGalleryDelete, localGalleryList } = vi.hoisted(() => ({
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  localGalleryDelete: vi.fn().mockResolvedValue(undefined),
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
    layoutJustifiedRows: (items: GalleryImage[]) =>
      items.map((item) => ({
        height: 180,
        items: [{ item, width: 180, height: 180 }],
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
  inTauri: () => false,
  ipc: {
    localGalleryDelete,
    localGalleryList,
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));

import LibraryView from "./LibraryView.vue";
import { useConnectionStore } from "../stores/connection";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";
import type { GalleryImage } from "../lib/api/types";

const prints: GalleryImage[] = [
  {
    filename: "first.png",
    timestamp: 2,
    metadata: {
      prompt: "first print",
      model: "flux-dev:q8",
      seed: 1,
      steps: 4,
      guidance: 1,
      width: 1024,
      height: 1024,
    },
  },
  {
    filename: "second.png",
    timestamp: 1,
    metadata: {
      prompt: "second print",
      model: "flux-dev:q8",
      seed: 2,
      steps: 4,
      guidance: 1,
      width: 1024,
      height: 1024,
    },
  },
];

const stub = { template: "<div />" };
// Named + prop-declaring stub so `getComponent({ name })` resolves it and the
// `open` prop is forwarded.
const historyDrawerStub = { name: "HistoryDrawer", props: ["open"], template: "<div />" };

async function mountView(
  remotePrint?: GalleryImage,
  seed?: (gallery: ReturnType<typeof useGalleryStore>) => void,
) {
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
  gallery.buckets.local = {
    items: [...prints],
    loading: false,
    error: null,
    loaded: true,
  };
  if (remotePrint) {
    useHostsStore().extras.push({
      id: "plato-7680",
      label: "plato",
      url: "http://plato:7680",
      apiKey: "secret",
      status: "ready",
      error: null,
      instanceId: null,
    });
    gallery.buckets["plato-7680"] = {
      items: [remotePrint],
      loading: false,
      error: null,
      loaded: true,
    };
  }
  localGalleryList.mockResolvedValue([...prints]);
  seed?.(gallery);

  const wrapper = mount(LibraryView, {
    global: {
      plugins: [pinia, router],
      stubs: {
        AuthedMedia: stub,
        HostFilterChips: stub,
        HistoryDrawer: historyDrawerStub,
      },
    },
  });
  await flushPromises();
  return { wrapper, gallery, router };
}

beforeEach(() => {
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response());
});

describe("LibraryView delete keyboard handling", () => {
  it.each(["Delete", "Backspace"])(
    "prevents %s navigation and requires a second press before deleting",
    async (key) => {
      const { wrapper, gallery, router } = await mountView();
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));

      const first = new KeyboardEvent("keydown", { key, cancelable: true });
      expect(window.dispatchEvent(first)).toBe(false);
      await flushPromises();

      expect(localGalleryDelete).not.toHaveBeenCalled();
      expect(wrapper.get('[data-test="single-delete-confirm"]').text()).toContain(
        "Delete first.png?",
      );
      expect(router.currentRoute.value.path).toBe("/library");

      const second = new KeyboardEvent("keydown", { key, cancelable: true });
      expect(window.dispatchEvent(second)).toBe(false);
      await flushPromises();

      expect(localGalleryDelete).toHaveBeenCalledWith("first.png");
      expect(gallery.filtered.map((entry) => entry.item.filename)).toEqual(["second.png"]);
      expect(router.currentRoute.value.path).toBe("/library");
      wrapper.unmount();
    },
  );

  it("routes context-menu confirmation to the selected remote host", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "remote.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    const { wrapper, gallery } = await mountView(remotePrint);
    const tile = wrapper.findAll("button").find((button) => button.text().includes("S 9"));
    expect(tile).toBeDefined();
    await tile!.trigger("contextmenu");

    const menu = useContextMenuStore();
    const deleteEntry = menu.entries.find(
      (entry): entry is Exclude<MenuEntry, { separator: true }> =>
        !("separator" in entry) && entry.label === "Delete",
    );
    expect(deleteEntry).toBeDefined();
    menu.activate(deleteEntry!);
    await flushPromises();

    expect(apiFetchTo).not.toHaveBeenCalled();
    await wrapper.get('[data-test="single-delete-confirm"] button').trigger("click");
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://plato:7680", apiKey: "secret" },
      "/api/gallery/image/remote.png",
      { method: "DELETE" },
    );
    expect(gallery.filtered.some((entry) => entry.item.filename === "remote.png")).toBe(false);
    wrapper.unmount();
  });

  it("surfaces a remote context-menu delete failure and keeps the print", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "remote.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    apiFetchTo.mockRejectedValueOnce(new Error("Forbidden"));
    const { wrapper, gallery } = await mountView(remotePrint);
    const tile = wrapper.findAll("button").find((button) => button.text().includes("S 9"));
    await tile!.trigger("contextmenu");
    const menu = useContextMenuStore();
    const deleteEntry = menu.entries.find(
      (entry) => !("separator" in entry) && entry.label === "Delete",
    )!;
    menu.activate(deleteEntry);
    await flushPromises();
    await wrapper.get('[data-test="single-delete-confirm"] button').trigger("click");
    await flushPromises();

    expect(useToastStore().items.at(-1)).toMatchObject({ message: "Forbidden", kind: "error" });
    expect(gallery.filtered.some((entry) => entry.item.filename === "remote.png")).toBe(true);
    wrapper.unmount();
  });

  it("leaves Backspace native while the library search field is being edited", async () => {
    const { wrapper } = await mountView();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));
    const search = wrapper.get("input[type='search']").element as HTMLInputElement;
    search.focus();

    const event = new KeyboardEvent("keydown", { key: "Backspace", cancelable: true });
    expect(search.dispatchEvent(event)).toBe(true);
    expect(localGalleryDelete).not.toHaveBeenCalled();
    expect(wrapper.find('[data-test="single-delete-confirm"]').exists()).toBe(false);
    wrapper.unmount();
  });
});

describe("LibraryView header + NEW badges", () => {
  it("titles the workspace Library and counts prints across all hosts", async () => {
    const { wrapper } = await mountView();
    const header = wrapper.get("header");
    expect(header.text()).toContain("Library");
    expect(header.text()).toContain("2 prints · all hosts");
    wrapper.unmount();
  });

  it("badges prints unseen since the last visit, then marks them seen", async () => {
    const { wrapper, gallery } = await mountView(undefined, (g) => {
      // A prior visit that had only seen second.png.
      g.libraryVisited = true;
      g.seenFilenames = new Set(["second.png"]);
    });

    const tiles = wrapper.findAll("button").filter((b) => b.text().includes("· S "));
    const fresh = tiles.find((b) => b.text().includes("S 1"));
    const stale = tiles.find((b) => b.text().includes("S 2"));
    expect(fresh!.find('[data-test="new-badge"]').exists()).toBe(true);
    expect(stale!.find('[data-test="new-badge"]').exists()).toBe(false);

    // Opening the Library marks everything currently shown as seen.
    expect(gallery.seenFilenames.has("first.png")).toBe(true);
    expect(gallery.seenFilenames.has("second.png")).toBe(true);
    wrapper.unmount();
  });

  it("shows nothing as new on the very first visit (baseline only)", async () => {
    const { wrapper, gallery } = await mountView();
    expect(wrapper.findAll('[data-test="new-badge"]')).toHaveLength(0);
    expect(gallery.libraryVisited).toBe(true);
    wrapper.unmount();
  });
});

describe("LibraryView history drawer", () => {
  it("opens the history drawer when ?panel=history is present", async () => {
    const { wrapper, router } = await mountView();
    expect(wrapper.getComponent({ name: "HistoryDrawer" }).props("open")).toBe(false);
    await router.push({ path: "/library", query: { panel: "history" } });
    await flushPromises();
    expect(wrapper.getComponent({ name: "HistoryDrawer" }).props("open")).toBe(true);
    wrapper.unmount();
  });

  it("the header History button deep-links to ?panel=history", async () => {
    const { wrapper, router } = await mountView();
    await wrapper.get('[aria-label="Open history"]').trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.panel).toBe("history");
    wrapper.unmount();
  });
});

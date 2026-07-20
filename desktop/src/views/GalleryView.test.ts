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

import GalleryView from "./GalleryView.vue";
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

async function mountView(remotePrint?: GalleryImage) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/gallery", component: { template: "<div />" } },
      { path: "/generate", component: { template: "<div />" } },
    ],
  });
  await router.push("/gallery");

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

  const wrapper = mount(GalleryView, {
    global: {
      plugins: [pinia, router],
      stubs: {
        AuthedMedia: { template: "<div />" },
        HostFilterChips: { template: "<div />" },
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

describe("GalleryView delete keyboard handling", () => {
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
      expect(router.currentRoute.value.path).toBe("/gallery");

      const second = new KeyboardEvent("keydown", { key, cancelable: true });
      expect(window.dispatchEvent(second)).toBe(false);
      await flushPromises();

      expect(localGalleryDelete).toHaveBeenCalledWith("first.png");
      expect(gallery.filtered.map((entry) => entry.item.filename)).toEqual(["second.png"]);
      expect(router.currentRoute.value.path).toBe("/gallery");
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

  it("leaves Backspace native while the gallery search field is being edited", async () => {
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

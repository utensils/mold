/**
 * My images: the Everything | Albums | Trash scopes, the filter
 * chip row, the collections shelf + drill-in, the trash grid, the bulk bar's
 * organization actions, the tile menu, the keyboard map, and URL sync. One
 * remote host (plato) advertises organize + trash; the local engine is off.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const { apiFetchTo, apiJsonTo, localGalleryList, localGalleryTrashList, org } = vi.hoisted(() => ({
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  apiJsonTo: vi.fn(),
  localGalleryList: vi.fn(),
  localGalleryTrashList: vi.fn(),
  org: {
    patchGalleryImage: vi.fn().mockResolvedValue(null),
    organizeGallery: vi.fn().mockResolvedValue(undefined),
    listCollections: vi.fn(),
    createCollection: vi.fn(),
    updateCollection: vi.fn(),
    updateCollectionHidden: vi.fn(),
    deleteCollection: vi.fn().mockResolvedValue(undefined),
    setCollectionItems: vi.fn(),
    listTags: vi.fn(),
    renameTag: vi.fn(),
    deleteTag: vi.fn(),
    trashGalleryImage: vi.fn(),
    deleteGalleryImageForever: vi.fn(),
    trashMany: vi.fn(),
    restoreTrashed: vi.fn().mockResolvedValue({ restored: 1 }),
    emptyTrash: vi.fn().mockResolvedValue({ purged: 1 }),
    sweepTrash: vi.fn(),
    listTrash: vi.fn(),
  },
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
    layoutJustifiedRows: (items: GalleryImage[], _w: number, targetHeight: number) =>
      items.map((item) => ({
        height: targetHeight,
        items: [{ item, width: targetHeight, height: targetHeight }],
      })),
  };
});
vi.mock("../lib/api/client", () => ({
  apiFetch: vi.fn(),
  apiFetchTo,
  apiJsonTo,
  currentTarget: () => ({ baseUrl: "http://x", apiKey: null }),
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public readonly status: number,
    ) {
      super(message);
      this.name = "ApiError";
    }
  },
}));
vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    localGalleryDelete: vi.fn(),
    localGalleryList,
    localGalleryTrashList,
    localGalleryRestore: vi.fn(),
    localGalleryDeleteForever: vi.fn(),
    revealOutputFile: vi.fn(),
    saveOutputBytes: vi.fn(),
  },
}));
vi.mock("@studio/api/galleryOrganization", () => org);

import LibraryView from "./LibraryView.vue";
import { useConnectionStore } from "../stores/connection";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useGalleryStore } from "../stores/gallery";
import { useHostsStore } from "../stores/hosts";
import { useToastStore } from "../stores/toasts";
import type { Collection, GalleryImage } from "../lib/api/types";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";

installMemoryLocalStorage();

const PLATO = { baseUrl: "http://plato:7680", apiKey: "secret" };
const NOW = Math.floor(Date.now() / 1000);

const base = (filename: string, timestamp: number, seed: number): GalleryImage => ({
  filename,
  timestamp,
  size_bytes: 1_000_000,
  metadata: {
    prompt: `prompt ${seed}`,
    model: "flux-dev:q8",
    seed,
    steps: 4,
    guidance: 1,
    width: 1024,
    height: 1024,
  },
});

const smurf04: GalleryImage = {
  ...base("mold-ltx-1~smurf-04.mp4", 40, 4),
  title: "smurf 04",
  favorite: true,
  tags: ["smurf", "blue"],
  collections: ["col-smurfs"],
};
const smurf03: GalleryImage = {
  ...base("mold-flux-2.png", 30, 3),
  tags: ["smurf"],
  collections: ["col-smurfs"],
};
const river: GalleryImage = { ...base("mold-flux-3.png", 20, 2), tags: ["outdoor"] };
const plain: GalleryImage = base("mold-flux-4.png", 10, 1);
const live = [smurf04, smurf03, river, plain];

const trashed: GalleryImage = {
  ...base("mold-flux-9.png", 5, 9),
  title: "Grain test 01",
  trashed_at: NOW - 60,
  purge_at: NOW + 3 * 86_400 - 60,
};

const smurfs: Collection = {
  id: "col-smurfs",
  name: "Smurfs",
  slug: "smurfs",
  description: null,
  cover_filename: null,
  count: 2,
  created_at: 1,
  updated_at: NOW - 7200,
};
const riverStudies: Collection = {
  ...smurfs,
  id: "col-river",
  name: "River studies",
  slug: "river-studies",
  count: 0,
};

const stub = { template: "<div />" };
const authedMediaStub = {
  name: "AuthedMedia",
  props: ["path"],
  template: "<div data-test='authed-media' :data-path='path' />",
};
const historyDrawerStub = { name: "HistoryDrawer", props: ["open"], template: "<div />" };

async function mountView(
  route = "/library",
  options: {
    organize?: boolean;
    trash?: boolean;
    trashItems?: GalleryImage[];
    hiddenCollection?: boolean;
    deferCapabilities?: boolean;
  } = {},
) {
  const {
    organize = true,
    trash = true,
    trashItems = [trashed],
    hiddenCollection = false,
    deferCapabilities = false,
  } = options;
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
      { path: "/settings", component: stub },
      { path: "/machines", component: stub },
      { path: "/machines/:id", component: stub },
    ],
  });
  await router.push(route);

  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = null;
  connection.status = "error";

  const hosts = useHostsStore();
  hosts.extras.push({
    id: "plato-7680",
    label: "plato",
    url: PLATO.baseUrl,
    apiKey: PLATO.apiKey,
    status: "ready",
    error: null,
    instanceId: null,
  });
  if (!deferCapabilities) {
    hosts.capabilities["plato-7680"] = {
      gallery: {
        can_delete: true,
        organize,
        trash: trash ? { enabled: true, retention_days: 30 } : null,
      },
    };
  }

  const gallery = useGalleryStore();
  gallery.buckets["plato-7680"] = {
    items: live.map((p) => structuredClone(p)),
    loading: false,
    error: null,
    loaded: true,
  };
  localGalleryList.mockResolvedValue({ images: [], target: null });
  localGalleryTrashList.mockResolvedValue({ images: [], target: null });
  apiJsonTo.mockImplementation(async (_target: unknown, path: string) => {
    if (path === "/api/gallery") return live.map((p) => structuredClone(p));
    if (path.startsWith("/api/gallery/collections/")) return { filenames: [] };
    return undefined;
  });
  org.listCollections.mockResolvedValue([
    { ...smurfs, hidden: hiddenCollection },
    { ...riverStudies },
  ]);
  org.listTags.mockResolvedValue([
    { name: "smurf", count: 2 },
    { name: "blue", count: 1 },
    { name: "outdoor", count: 1 },
  ]);
  org.listTrash.mockResolvedValue(trashItems.map((p) => structuredClone(p)));
  org.createCollection.mockImplementation(async (_t: unknown, body: { name: string }) => ({
    ...smurfs,
    id: `col-${body.name.toLowerCase()}`,
    name: body.name,
    slug: body.name.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    count: 0,
  }));
  org.updateCollection.mockImplementation(
    async (_t: unknown, id: string, body: { name?: string; hidden?: boolean }) => ({
      ...(id === "col-smurfs" ? smurfs : riverStudies),
      ...body,
      ...(body.name ? { name: body.name, slug: body.name.toLowerCase() } : {}),
    }),
  );
  org.updateCollectionHidden.mockImplementation(
    async (_t: unknown, id: string, hidden: boolean) => ({
      ...(id === "col-smurfs" ? smurfs : riverStudies),
      hidden,
    }),
  );

  const wrapper = mount(LibraryView, {
    global: {
      plugins: [pinia, router],
      stubs: { AuthedMedia: authedMediaStub, HistoryDrawer: historyDrawerStub },
    },
  });
  await flushPromises();
  return { wrapper, gallery, router, hosts };
}

function menuEntry(label: string) {
  return useContextMenuStore().entries.find(
    (entry): entry is Exclude<MenuEntry, { separator: true }> =>
      !("separator" in entry) && entry.label === label,
  );
}

function tileFor(wrapper: Awaited<ReturnType<typeof mountView>>["wrapper"], filename: string) {
  return wrapper.get(`[data-filename="${filename}"]`);
}

const key = (k: string, init: KeyboardEventInit = {}) =>
  window.dispatchEvent(new KeyboardEvent("keydown", { key: k, cancelable: true, ...init }));

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  apiFetchTo.mockResolvedValue(new Response());
  org.patchGalleryImage.mockResolvedValue(null);
  org.organizeGallery.mockResolvedValue(undefined);
  org.restoreTrashed.mockResolvedValue({ restored: 1 });
  org.emptyTrash.mockResolvedValue({ purged: 1 });
  org.deleteCollection.mockResolvedValue(undefined);
});

afterEach(() => {
  document.body.innerHTML = "";
});

describe("scopes + capability gating", () => {
  it("shows Everything | Albums | Trash with counts when a machine can organize and trash", async () => {
    const { wrapper } = await mountView();
    const control = wrapper.get("[data-test='library-scope']");
    const labels = control
      .findAll("button")
      .map((b) => `${b.find(".ms-seg__label").text()} ${b.find(".ms-seg__sub").text()}`);
    expect(labels).toEqual(["Everything 4", "Albums 2", "Trash 1"]);
    expect(wrapper.get("[data-test='library-count']").text()).toBe("4 pictures · 4.0 MB");
    expect(wrapper.find("[data-test='library-chip-row']").exists()).toBe(true);
    wrapper.unmount();
  });

  it("collapses to Prints only (no chip row, no ♥) on an old server", async () => {
    const { wrapper } = await mountView("/library", { organize: false, trash: false });
    expect(wrapper.find("[data-test='library-scope']").exists()).toBe(false);
    // The chip row survives for the host chips, but carries no ♥ or tags.
    expect(wrapper.find("[data-test='favorites-chip']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='tag-chip']")).toHaveLength(0);
    expect(wrapper.findAll("[data-test='tile-favorite']")).toHaveLength(0);
    await tileFor(wrapper, smurf04.filename).trigger("contextmenu");
    expect(menuEntry("Favourite")).toBeUndefined();
    expect(menuEntry("Delete")).toBeDefined();
    wrapper.unmount();
  });
});

describe("Prints tiles + chip row", () => {
  it("reads title · model · seed on the edge strip and ♥ on favorites", async () => {
    const { wrapper } = await mountView();
    const strip = tileFor(wrapper, smurf04.filename).get("[data-test='edge-strip']");
    expect(strip.text().replace(/\s+/g, " ")).toBe("smurf 04 · flux-dev:q8 · seed 4");
    expect(
      tileFor(wrapper, smurf04.filename)
        .get("[data-test='tile-favorite']")
        .attributes("aria-pressed"),
    ).toBe("true");
    expect(
      tileFor(wrapper, plain.filename)
        .get("[data-test='tile-favorite']")
        .attributes("aria-pressed"),
    ).toBe("false");
    wrapper.unmount();
  });

  it("the tile ♥ toggles favorite on every copy without opening the print", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).get("[data-test='tile-favorite']").trigger("click");
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [plain.filename],
      favorite: true,
    });
    expect(wrapper.findComponent({ name: "Lightbox" }).exists()).toBe(false);
    wrapper.unmount();
  });

  it("♥ Favourites and tag chips filter the grid (AND) and sync to ?fav / ?tag", async () => {
    const { wrapper, router } = await mountView();
    await wrapper.get("[data-test='favorites-chip']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(1);
    expect(router.currentRoute.value.query.fav).toBe("1");

    await wrapper.get("[data-test='favorites-chip']").trigger("click");
    await wrapper.get("[data-test='tag-chip'][data-tag='smurf']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(2);
    await wrapper.get("[data-test='tag-chip'][data-tag='blue']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(1);
    expect(router.currentRoute.value.query.tag).toBe("smurf,blue");
    expect(router.currentRoute.value.query.fav).toBeUndefined();

    await wrapper.get("[data-test='clear-filters']").trigger("click");
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(4);
    expect(router.currentRoute.value.query.tag).toBeUndefined();
    wrapper.unmount();
  });

  it("restores ♥ / tag filters from a deep link", async () => {
    const { wrapper, gallery } = await mountView("/library?fav=1&tag=smurf");
    expect(gallery.favoritesOnly).toBe(true);
    expect(gallery.tagFilter).toEqual(["smurf"]);
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(1);
    wrapper.unmount();
  });
});

describe("tile context menu", () => {
  it("adds Favourite, Rename…, Tags ▸, Add to album ▸, and Move to trash", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, smurf03.filename).trigger("contextmenu");
    expect(menuEntry("Favourite")).toMatchObject({ checked: false });
    expect(menuEntry("Rename…")).toBeDefined();
    const tags = menuEntry("Tags")!;
    expect(tags.children!.map((c) => ("separator" in c ? "—" : c.label))).toEqual([
      "smurf",
      "blue",
      "outdoor",
      "—",
      "New tag…",
    ]);
    expect(tags.children![0]).toMatchObject({ checked: true });
    const collections = menuEntry("Add to album")!;
    expect(collections.children!.map((c) => ("separator" in c ? "—" : c.label))).toEqual([
      "River studies",
      "Smurfs",
      "—",
      "New album…",
    ]);
    expect(collections.children![1]).toMatchObject({ checked: true });
    expect(menuEntry("Move to trash")).toMatchObject({ danger: true });
    expect(menuEntry("Use these settings")).toBeDefined();
    expect(menuEntry("Copy prompt")).toBeDefined();

    // Checking a tag adds it to the print on its host.
    useContextMenuStore().activate(tags.children![1]!);
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [smurf03.filename],
      add_tags: ["blue"],
    });
    wrapper.unmount();
  });

  it("Rename… opens the shared dialog and PATCHes the title", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).trigger("contextmenu");
    useContextMenuStore().activate(menuEntry("Rename…")!);
    await flushPromises();
    const dialog = wrapper.get("[data-test='rename-dialog']");
    expect(dialog.text()).toContain("Rename print");
    await dialog.get("input").setValue("Bottled storm");
    await dialog.get("[data-test='rename-save']").trigger("click");
    await flushPromises();
    expect(org.patchGalleryImage).toHaveBeenCalledWith(PLATO, plain.filename, {
      title: "Bottled storm",
    });
    wrapper.unmount();
  });

  it("Move to trash hides the print behind a 6 s undo toast, then DELETEs (the host trashes)", async () => {
    const { wrapper, gallery } = await mountView();
    vi.useFakeTimers();
    try {
      await tileFor(wrapper, plain.filename).trigger("contextmenu");
      useContextMenuStore().activate(menuEntry("Move to trash")!);
      await flushPromises();
      expect(gallery.filtered.some((e) => e.item.filename === plain.filename)).toBe(false);
      const toast = useToastStore().items.at(-1)!;
      expect(toast.message).toBe("Moved “prompt 1” to trash");
      expect(toast.action?.label).toBe("Undo");
      expect(apiFetchTo).not.toHaveBeenCalled();
      vi.advanceTimersByTime(6000);
      await flushPromises();
      expect(apiFetchTo).toHaveBeenCalledWith(PLATO, `/api/gallery/image/${plain.filename}`, {
        method: "DELETE",
      });
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });
});

describe("bulk bar", () => {
  it("enters selection naturally with Command-click and Shift-click", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, river.filename).trigger("click", { metaKey: true });

    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("2 / 4 selected");
    expect(tileFor(wrapper, smurf04.filename).get("button").attributes("aria-pressed")).toBe(
      "true",
    );
    expect(tileFor(wrapper, river.filename).get("button").attributes("aria-pressed")).toBe("true");

    await tileFor(wrapper, plain.filename).trigger("pointerdown", {
      pointerId: 40,
      pointerType: "mouse",
      button: 0,
      isPrimary: true,
      shiftKey: true,
    });
    window.dispatchEvent(
      new PointerEvent("pointerup", {
        pointerId: 40,
        pointerType: "mouse",
        isPrimary: true,
        shiftKey: true,
      }),
    );
    await tileFor(wrapper, plain.filename).trigger("click", { shiftKey: true });
    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("3 / 4 selected");
    wrapper.unmount();
  });

  it("the Select All chord selects only the prints in the active filter", async () => {
    const { wrapper } = await mountView();
    await wrapper.get("[data-test='tag-chip'][data-tag='smurf']").trigger("click");
    await flushPromises();

    // jsdom is not a Tauri window, so `CURRENT_PLATFORM` is "unknown" and the
    // primary modifier is Control — the same answer `primaryModifierPressed`
    // gives the ⌘⇧N tests below. Dispatching ⌘A here would test macOS from a
    // non-macOS platform and simply not match.
    key("a", { ctrlKey: true });
    await wrapper.vm.$nextTick();

    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("2 / 2 selected");
    expect(wrapper.findAll("[data-test='select-indicator']").map((node) => node.text())).toEqual([
      "✓",
      "✓",
    ]);
    wrapper.unmount();
  });

  it("drag-selects and drag-deselects every visible tile crossed", async () => {
    const { wrapper } = await mountView();
    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    const tiles = [smurf04, smurf03, river].map((print) => tileFor(wrapper, print.filename));
    const previous = Object.getOwnPropertyDescriptor(document, "elementsFromPoint");
    Object.defineProperty(document, "elementsFromPoint", {
      configurable: true,
      value: vi.fn((x: number) => [
        x < 100 ? tiles[0]!.element : x < 200 ? tiles[1]!.element : tiles[2]!.element,
      ]),
    });
    try {
      await tiles[0]!.trigger("pointerdown", {
        pointerId: 41,
        pointerType: "mouse",
        button: 0,
        isPrimary: true,
        clientX: 20,
        clientY: 200,
      });
      window.dispatchEvent(
        new PointerEvent("pointermove", {
          pointerId: 41,
          pointerType: "mouse",
          isPrimary: true,
          clientX: 260,
          clientY: 200,
        }),
      );
      window.dispatchEvent(
        new PointerEvent("pointerup", { pointerId: 41, pointerType: "mouse", isPrimary: true }),
      );
      await wrapper.vm.$nextTick();
      expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("3 / 4 selected");

      await tiles[0]!.trigger("pointerdown", {
        pointerId: 42,
        pointerType: "mouse",
        button: 0,
        isPrimary: true,
        clientX: 20,
        clientY: 200,
      });
      window.dispatchEvent(
        new PointerEvent("pointermove", {
          pointerId: 42,
          pointerType: "mouse",
          isPrimary: true,
          clientX: 150,
          clientY: 200,
        }),
      );
      window.dispatchEvent(
        new PointerEvent("pointerup", { pointerId: 42, pointerType: "mouse", isPrimary: true }),
      );
      await wrapper.vm.$nextTick();
      expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("1 / 4 selected");
      expect(tileFor(wrapper, river.filename).get("button").attributes("aria-pressed")).toBe(
        "true",
      );
    } finally {
      if (previous) Object.defineProperty(document, "elementsFromPoint", previous);
      else Reflect.deleteProperty(document, "elementsFromPoint");
      wrapper.unmount();
    }
  });

  it("right-click targets the selected group and resets to an unselected print", async () => {
    const { wrapper } = await mountView();
    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, plain.filename).trigger("click", { metaKey: true });

    await tileFor(wrapper, smurf04.filename).trigger("contextmenu");
    expect(menuEntry("Favourite 2 selected")).toBeDefined();
    expect(menuEntry("Move 2 selected to trash")).toBeDefined();
    expect(menuEntry("Rename…")).toBeUndefined();

    await tileFor(wrapper, river.filename).trigger("contextmenu");
    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("1 / 4 selected");
    expect(menuEntry("Move to trash")).toBeDefined();
    wrapper.unmount();
  });

  it("preserves the selected group when macOS Control-click opens its context menu", async () => {
    const { wrapper } = await mountView();
    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, plain.filename).trigger("click", { metaKey: true });

    await tileFor(wrapper, smurf04.filename).trigger("pointerdown", {
      pointerId: 43,
      pointerType: "mouse",
      button: 0,
      isPrimary: true,
      ctrlKey: true,
    });
    await tileFor(wrapper, smurf04.filename).trigger("contextmenu", { ctrlKey: true });

    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("2 / 4 selected");
    expect(menuEntry("Favourite 2 selected")).toBeDefined();
    expect(menuEntry("Move 2 selected to trash")).toBeDefined();
    wrapper.unmount();
  });

  it("labels mixed trash-capability selections as permanent delete", async () => {
    const { wrapper, gallery, hosts } = await mountView();
    const legacy = base("legacy-host-print.png", 8, 8);
    apiJsonTo.mockImplementation(async (target: { baseUrl?: string }, path: string) => {
      if (path === "/api/gallery") {
        return target?.baseUrl === "http://legacy:7680"
          ? [structuredClone(legacy)]
          : live.map((print) => structuredClone(print));
      }
      if (path.startsWith("/api/gallery/collections/")) return { filenames: [] };
      return undefined;
    });
    hosts.extras.push({
      id: "legacy-7680",
      label: "legacy",
      url: "http://legacy:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    hosts.capabilities["legacy-7680"] = {
      gallery: { can_delete: true, organize: false, trash: null },
    };
    gallery.buckets["legacy-7680"] = {
      items: [legacy],
      loading: false,
      error: null,
      loaded: true,
    };
    await flushPromises();

    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, legacy.filename).trigger("click", { metaKey: true });
    await tileFor(wrapper, smurf04.filename).trigger("contextmenu");

    expect(menuEntry("Delete 2 selected")).toBeDefined();
    expect(menuEntry("Move 2 selected to trash")).toBeUndefined();
    wrapper.unmount();
  });

  it("moves the selection to the trash behind one undo toast; undo restores every print", async () => {
    const { wrapper, gallery } = await mountView();
    vi.useFakeTimers();
    try {
      await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
      await tileFor(wrapper, river.filename).trigger("click");
      await tileFor(wrapper, plain.filename).trigger("click", { metaKey: true });
      const bar = wrapper.get("[data-test='bulk-action-bar']");
      expect(bar.text()).toContain("2 / 4 selected");
      expect(bar.get("[data-test='bulk-delete']").text()).toContain("Move 2 pictures to trash");
      await bar.get("[data-test='bulk-delete']").trigger("click");
      await flushPromises();
      expect(gallery.filtered).toHaveLength(2);
      const toast = useToastStore().items.at(-1)!;
      expect(toast.message).toBe("Moved 2 pictures to trash");
      useToastStore().runAction(toast.id);
      await flushPromises();
      expect(gallery.filtered).toHaveLength(4);
      vi.advanceTimersByTime(6000);
      await flushPromises();
      expect(apiFetchTo).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });

  it("♥ Favourite favourites all when any is unfavourited; Tag applies to the selection", async () => {
    const { wrapper } = await mountView();
    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, plain.filename).trigger("click", { metaKey: true });
    const bar = wrapper.get("[data-test='bulk-action-bar']");
    await bar.get("[data-test='bulk-favorite']").trigger("click");
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [smurf04.filename, plain.filename],
      favorite: true,
    });

    await bar.get("[data-test='bulk-tags']").trigger("click");
    const panel = document.body.querySelector("[data-test='bulk-tags-panel']")!;
    // Intersection over the selection: plain has no tags.
    expect(panel.querySelectorAll("[data-test='tag-chip']")).toHaveLength(0);
    const input = panel.querySelector("[data-test='tag-input']") as HTMLInputElement;
    input.value = "keep";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [smurf04.filename, plain.filename],
      add_tags: ["keep"],
    });
    wrapper.unmount();
  });

  it("Add to album shows mixed state and creating a new one adds the selection", async () => {
    const { wrapper } = await mountView();
    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click");
    await tileFor(wrapper, plain.filename).trigger("click", { metaKey: true });
    await wrapper.get("[data-test='bulk-collections']").trigger("click");
    const panel = document.body.querySelector("[data-test='bulk-collections-panel']")!;
    expect(panel.querySelector("[data-slug='smurfs']")!.getAttribute("aria-checked")).toBe("mixed");
    (panel.querySelector("[data-test='collection-new']") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    const input = panel.querySelector("[data-test='collection-new-input']") as HTMLInputElement;
    input.value = "Halcyon";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await flushPromises();
    expect(org.createCollection).toHaveBeenCalledWith(PLATO, { name: "Halcyon" });
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [smurf04.filename, plain.filename],
      add_to_collections: ["col-halcyon"],
    });
    expect(useToastStore().items.at(-1)?.message).toBe("Added 2 pictures to “Halcyon”");
    wrapper.unmount();
  });
});

describe("Albums scope", () => {
  it("lists a card per merged collection with logical counts, and drills in via ?c=", async () => {
    const { wrapper, router } = await mountView("/library?scope=collections");
    const cards = wrapper.findAll("[data-test='collection-card']");
    expect(cards.map((c) => c.find("[data-test='collection-name']").text())).toEqual([
      "River studies",
      "Smurfs",
    ]);
    expect(cards[1]!.find("[data-test='collection-meta']").text()).toBe("2 pictures · plato");
    expect(wrapper.get("[data-test='library-count']").text()).toBe("2 albums");
    expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(false);

    // The collection endpoint records membership order (when prints were
    // added), which is deliberately the reverse of generation time here.
    apiJsonTo.mockImplementation(async (_target: unknown, path: string) => {
      if (path === "/api/gallery") return live.map((p) => structuredClone(p));
      if (path.startsWith("/api/gallery/collections/")) {
        return { filenames: [smurf03.filename, smurf04.filename] };
      }
      return undefined;
    });
    await cards[1]!.trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query).toMatchObject({ scope: "collections", c: "smurfs" });
    expect(wrapper.get("[data-test='crumb-here']").text()).toBe("Smurfs");
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(2);
    expect(wrapper.findAll(".ms-lib-tile").map((tile) => tile.attributes("data-filename"))).toEqual(
      [smurf04.filename, smurf03.filename],
    );
    expect(wrapper.get("[data-test='collection-chip']").text()).toContain("Smurfs");

    // Inside the collection the tile menu offers Use as cover / Remove.
    await tileFor(wrapper, smurf03.filename).trigger("contextmenu");
    expect(menuEntry("Use as cover")).toBeDefined();
    useContextMenuStore().activate(menuEntry("Remove from album")!);
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [smurf03.filename],
      remove_from_collections: ["col-smurfs"],
    });

    await wrapper.get("[data-test='crumb-back']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.query.c).toBeUndefined();
    expect(wrapper.findAll("[data-test='collection-card']")).toHaveLength(2);
    wrapper.unmount();
  });

  it("creates from the New album card and deletes via a plain confirm", async () => {
    const { wrapper, gallery } = await mountView("/library?scope=collections");
    await wrapper.get("[data-test='new-collection-label']").trigger("click");
    const input = wrapper.get("[data-test='new-collection-input']");
    await input.setValue("Film grain tests");
    await input.trigger("keydown", { key: "Enter" });
    await flushPromises();
    expect(org.createCollection).toHaveBeenCalledWith(PLATO, { name: "Film grain tests" });
    expect(gallery.mergedCollections.map((c) => c.name)).toContain("Film grain tests");

    await wrapper.get("[data-test='collection-card'][data-slug='smurfs']").trigger("contextmenu");
    useContextMenuStore().activate(menuEntry("Delete album…")!);
    await flushPromises();
    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Delete album “Smurfs”?");
    expect(dialog.text()).toContain("Its pictures stay in My images.");
    await dialog.get("[data-test='confirm-accept']").trigger("click");
    await flushPromises();
    expect(org.deleteCollection).toHaveBeenCalledWith(PLATO, "col-smurfs");
    expect(gallery.mergedCollections.map((c) => c.name)).not.toContain("Smurfs");
    // The prints are untouched.
    expect(gallery.merged).toHaveLength(4);
    wrapper.unmount();
  });

  it("hides collection members from Prints and search while preserving drill-in", async () => {
    const { wrapper, gallery } = await mountView("/library?scope=collections");
    await wrapper.get("[data-test='collection-card'][data-slug='smurfs']").trigger("contextmenu");
    useContextMenuStore().activate(menuEntry("Hide from Library")!);
    await flushPromises();
    expect(org.updateCollectionHidden).toHaveBeenCalledWith(PLATO, "col-smurfs", true);
    expect(
      gallery.mergedCollections.find((collection) => collection.slug === "smurfs")?.hidden,
    ).toBe(true);
    expect(wrapper.get("[data-test='collection-hidden-badge']").text()).toBe("Hidden");

    gallery.scope = "prints";
    gallery.query = "smurf";
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(0);

    gallery.query = "";
    gallery.scope = "collections";
    gallery.collectionSlug = "smurfs";
    await flushPromises();
    expect(wrapper.findAll(".ms-lib-tile")).toHaveLength(2);
    wrapper.unmount();
  });

  it("⌘⇧N opens the New album dialog from Everything", async () => {
    const { wrapper } = await mountView();
    key("N", { ctrlKey: true, shiftKey: true });
    await flushPromises();
    expect(wrapper.get("[data-test='rename-dialog']").text()).toContain("New album");
    wrapper.unmount();
  });
});

describe("Trash scope", () => {
  it("lists trashed prints with the banner, a purge chip, and hover Restore / Delete forever", async () => {
    const { wrapper } = await mountView("/library?scope=trash");
    expect(wrapper.get("[data-test='library-count']").text()).toBe("1 picture in trash · 1.0 MB");
    expect(wrapper.get("[data-test='trash-banner-summary']").text()).toBe(
      "Prints stay in the trash 30 d before purge",
    );
    const tile = tileFor(wrapper, trashed.filename);
    expect(tile.get("[data-test='purge-chip']").text().replace(/\s+/g, " ")).toBe("Purges in 3 d");
    expect(tile.find("[data-test='edge-strip']").exists()).toBe(false);
    await tile.get("[data-test='trash-restore']").trigger("click");
    await flushPromises();
    expect(org.restoreTrashed).toHaveBeenCalledWith(PLATO, [trashed.filename]);
    expect(useToastStore().items.at(-1)?.message).toBe("Restored 1 picture");
    wrapper.unmount();
  });

  it("Empty trash confirms with the plain dialog naming the hosts, then purges", async () => {
    const { wrapper } = await mountView("/library?scope=trash");
    await wrapper.get("[data-test='empty-trash']").trigger("click");
    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Empty trash?");
    expect(dialog.text()).toContain(
      "Delete 1 picture in the trash on plato forever? This can't be undone.",
    );
    expect(dialog.find("input").exists()).toBe(false);
    await dialog.get("[data-test='confirm-accept']").trigger("click");
    await flushPromises();
    expect(org.emptyTrash).toHaveBeenCalledWith(PLATO);
    expect(wrapper.find("[data-test='empty-trash']").attributes("disabled")).toBeDefined();
    expect(wrapper.text()).toContain("Trash is empty");
    wrapper.unmount();
  });

  it("Delete in Trash = delete forever behind a confirm; the banner link routes to Machines", async () => {
    const { wrapper, router } = await mountView("/library?scope=trash");
    await tileFor(wrapper, trashed.filename).trigger("click");
    key("Delete");
    await flushPromises();
    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Delete “Grain test 01” forever?");
    await dialog.get("[data-test='confirm-accept']").trigger("click");
    await flushPromises();
    expect(apiFetchTo).toHaveBeenCalledWith(
      PLATO,
      `/api/gallery/image/${trashed.filename}?permanent=true`,
      { method: "DELETE" },
    );

    await wrapper.get("[data-test='trash-banner-link']").trigger("click");
    await flushPromises();
    expect(router.currentRoute.value.path).toBe("/machines/plato-7680");
    wrapper.unmount();
  });

  it("the tile menu offers Restore / Delete forever only", async () => {
    const { wrapper } = await mountView("/library?scope=trash");
    await tileFor(wrapper, trashed.filename).trigger("contextmenu");
    const labels = useContextMenuStore().entries.map((e) => ("separator" in e ? "—" : e.label));
    expect(labels).toEqual(["Restore", "Copy prompt", "Copy seed", "—", "Delete forever"]);
    wrapper.unmount();
  });
});

describe("keyboard", () => {
  it("F toggles favorite on the selected print; ⌘⌫ asks to delete forever", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).trigger("click");
    key("f");
    await flushPromises();
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [plain.filename],
      favorite: true,
    });
    key("Backspace", { ctrlKey: true });
    await flushPromises();
    expect(wrapper.get("[data-test='confirm-dialog']").text()).toContain("forever?");
    wrapper.unmount();
  });

  it("T enters select mode around the selected print and opens the Tag popover", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).trigger("click");
    key("t");
    await flushPromises();
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='bulk-action-bar']").text()).toContain("1 / 4 selected");
    expect(document.body.querySelector("[data-test='bulk-tags-panel']")).not.toBeNull();
    // Escape closes the popover first, then leaves select mode.
    key("Escape");
    await wrapper.vm.$nextTick();
    expect(document.body.querySelector("[data-test='bulk-tags-panel']")).toBeNull();
    expect(wrapper.find("[data-test='bulk-action-bar']").exists()).toBe(true);
    key("Escape");
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='bulk-action-bar']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("ignores the single-key shortcuts while typing in the search field", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).trigger("click");
    const search = wrapper.get("input[type='search']").element as HTMLInputElement;
    search.focus();
    search.dispatchEvent(new KeyboardEvent("keydown", { key: "f", bubbles: true }));
    await flushPromises();
    expect(org.organizeGallery).not.toHaveBeenCalled();
    wrapper.unmount();
  });
});

describe("Lightbox wiring", () => {
  it("renames, favorites, tags, and toggles collections through the store", async () => {
    const { wrapper } = await mountView();
    await tileFor(wrapper, plain.filename).trigger("dblclick");
    const lightbox = wrapper.getComponent({ name: "Lightbox" });
    expect(lightbox.props("canOrganize")).toBe(true);
    expect(lightbox.props("canTrash")).toBe(true);
    lightbox.vm.$emit("rename", "Bottled storm");
    lightbox.vm.$emit("favorite", true);
    lightbox.vm.$emit("tags", { add: ["keep"], remove: [] });
    lightbox.vm.$emit("collections", { slug: "smurfs", checked: true });
    await flushPromises();
    expect(org.patchGalleryImage).toHaveBeenCalledWith(PLATO, plain.filename, {
      title: "Bottled storm",
    });
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [plain.filename],
      favorite: true,
    });
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [plain.filename],
      add_tags: ["keep"],
    });
    expect(org.organizeGallery).toHaveBeenCalledWith(PLATO, {
      filenames: [plain.filename],
      add_to_collections: ["col-smurfs"],
    });
    wrapper.unmount();
  });

  it("a ?print deep link resets the scope and filters so the print cannot hide", async () => {
    const { wrapper, gallery } = await mountView(
      "/library?scope=trash&fav=1&print=mold-flux-4.png",
    );
    expect(gallery.scope).toBe("prints");
    expect(gallery.favoritesOnly).toBe(false);
    expect(wrapper.getComponent({ name: "Lightbox" }).props("item").filename).toBe(plain.filename);
    wrapper.unmount();
  });

  it("a notification deep link opens a video inside its hidden collection", async () => {
    const { wrapper, gallery, router } = await mountView(
      `/library?print=${encodeURIComponent(smurf04.filename)}`,
      { hiddenCollection: true },
    );

    expect(gallery.scope).toBe("collections");
    expect(gallery.collectionSlug).toBe("smurfs");
    expect(wrapper.getComponent({ name: "Lightbox" }).props("item").filename).toBe(
      smurf04.filename,
    );
    await vi.waitFor(() =>
      expect(router.currentRoute.value.query).toEqual({ scope: "collections", c: "smurfs" }),
    );
    wrapper.unmount();
  });

  it("retains a hidden-video notification until late capabilities and collections settle", async () => {
    const { wrapper, gallery, router, hosts } = await mountView(
      `/library?print=${encodeURIComponent(smurf04.filename)}`,
      { hiddenCollection: true, deferCapabilities: true },
    );

    expect(router.currentRoute.value.query.print).toBe(smurf04.filename);
    expect(wrapper.findComponent({ name: "Lightbox" }).exists()).toBe(false);

    hosts.capabilities["plato-7680"] = {
      gallery: { can_delete: true, organize: true, trash: null },
    };
    await vi.waitFor(() => expect(gallery.collectionsByHost["plato-7680"]?.loaded).toBe(true));
    await vi.waitFor(() => expect(gallery.collectionSlug).toBe("smurfs"));
    expect(wrapper.getComponent({ name: "Lightbox" }).props("item").filename).toBe(
      smurf04.filename,
    );
    wrapper.unmount();
  });

  it("opens a delayed notification target when a same-count refresh adds it", async () => {
    const late = { ...plain, filename: "late-h3-video.mp4", timestamp: 50 };
    const { wrapper, gallery, router } = await mountView(
      `/library?print=${encodeURIComponent(late.filename)}`,
    );

    expect(router.currentRoute.value.query.print).toBe(late.filename);
    expect(wrapper.findComponent({ name: "Lightbox" }).exists()).toBe(false);
    gallery.buckets["plato-7680"]!.items.splice(0, 1, late);

    await vi.waitFor(() =>
      expect(wrapper.getComponent({ name: "Lightbox" }).props("item").filename).toBe(late.filename),
    );
    expect(gallery.buckets["plato-7680"]!.items).toHaveLength(live.length);
    wrapper.unmount();
  });
});

describe("Trash-aware This-Mac media + file actions", () => {
  const localTrashed: GalleryImage = {
    ...base("mold-flux-local-9.png", 8, 42),
    trashed_at: NOW - 120,
    purge_at: NOW + 14 * 86_400 - 120,
  };

  async function mountWithLocalTrash() {
    const mounted = await mountView("/library?scope=trash");
    localGalleryTrashList.mockResolvedValue({
      images: [structuredClone(localTrashed)],
      target: null,
      retentionDays: 14,
    });
    await mounted.gallery.fetchTrash("local");
    await flushPromises();
    await mounted.wrapper.vm.$nextTick();
    return mounted;
  }

  it("resolves a trashed This-Mac tile's media into `.trash/` via the view-aware protocol URL", async () => {
    const { wrapper } = await mountWithLocalTrash();
    const media = tileFor(wrapper, localTrashed.filename).get("[data-test='authed-media']");
    // A tile is a thumbnail request against the native cache; the trash hint
    // rides along so the offline render reads `.trash/` first.
    expect(media.attributes("data-path")).toBe(
      `mold-thumb://localhost/local/${encodeURIComponent(localTrashed.filename)}?view=trash`,
    );
    // Host-backed trash rows stay on the API path (their server resolves
    // trashed rows itself).
    const hostMedia = tileFor(wrapper, trashed.filename).get("[data-test='authed-media']");
    expect(hostMedia.attributes("data-path")).toMatch(
      new RegExp(`^/api/gallery/thumbnail/${encodeURIComponent(trashed.filename)}\\?size=`),
    );
    wrapper.unmount();
  });

  it("Reveal in the Trash lightbox targets the `.trash/` copy of a This-Mac print", async () => {
    const { wrapper } = await mountWithLocalTrash();
    await tileFor(wrapper, localTrashed.filename).trigger("dblclick");
    const lightbox = wrapper.getComponent({ name: "Lightbox" });
    expect(lightbox.props("trashed")).toBe(true);
    const reveal = lightbox.findAll("button").find((b) => b.text() === "Show the file");
    expect(reveal, "trashed local prints keep a working Reveal").toBeDefined();
    await reveal!.trigger("click");
    await flushPromises();
    expect(vi.mocked((await import("../lib/ipc")).ipc.revealOutputFile)).toHaveBeenCalledWith(
      localTrashed.filename,
      true,
    );
    wrapper.unmount();
  });

  it("the Trash banner names This device from the offline retention instead of claiming no trash", async () => {
    const { wrapper, gallery } = await mountWithLocalTrash();
    expect(gallery.retentionByHost.map((h) => [h.key, h.retentionDays])).toEqual([
      ["local", 14],
      ["plato-7680", 30],
    ]);
    expect(wrapper.get("[data-test='trash-banner-summary']").text()).toContain("14 d");
    wrapper.unmount();
  });
});

describe("selection-derived organize gating", () => {
  /** A second, organize-incapable host (old server) holding one print. */
  function addLegacyHost(mounted: Awaited<ReturnType<typeof mountView>>, image: GalleryImage) {
    // The view refetches a newly appeared host's bucket over HTTP — answer
    // okra's /api/gallery with its single print, plato keeps the shared set.
    apiJsonTo.mockImplementation(
      async (target: { baseUrl?: string } | null, path: string): Promise<unknown> => {
        if (path === "/api/gallery") {
          return target?.baseUrl === "http://okra:7680"
            ? [structuredClone(image)]
            : live.map((p) => structuredClone(p));
        }
        if (path.startsWith("/api/gallery/collections/")) return { filenames: [] };
        return undefined;
      },
    );
    mounted.hosts.extras.push({
      id: "okra-7680",
      label: "okra",
      url: "http://okra:7680",
      apiKey: null,
      status: "ready",
      error: null,
      instanceId: null,
    });
    mounted.hosts.capabilities["okra-7680"] = { gallery: { can_delete: true } };
    mounted.gallery.buckets["okra-7680"] = {
      items: [structuredClone(image)],
      loading: false,
      error: null,
      loaded: true,
    };
  }

  const legacyOnly = base("mold-flux-okra-1.png", 35, 77);

  it("disables Favourite / Tag / Album for selections holding a picture with no organize-capable copy", async () => {
    const mounted = await mountView();
    const { wrapper } = mounted;
    addLegacyHost(mounted, legacyOnly);
    await flushPromises();

    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, legacyOnly.filename).trigger("click");
    await tileFor(wrapper, smurf04.filename).trigger("click", { metaKey: true });
    const bar = wrapper.get("[data-test='bulk-action-bar']");
    for (const control of ["bulk-collections", "bulk-tags", "bulk-favorite"]) {
      const button = bar.get(`[data-test='${control}']`);
      expect(button.attributes("disabled"), control).toBeDefined();
      expect(button.attributes("title")).toContain("okra");
    }

    // Dropping the blocked print re-enables the controls.
    await tileFor(wrapper, legacyOnly.filename).trigger("click", { metaKey: true });
    for (const control of ["bulk-collections", "bulk-tags", "bulk-favorite"]) {
      const button = bar.get(`[data-test='${control}']`);
      expect(button.attributes("disabled"), control).toBeUndefined();
    }
    wrapper.unmount();
  });

  it("refuses to create a collection when zero selected copies can be added (no empty-collection side effect)", async () => {
    const mounted = await mountView();
    const { wrapper } = mounted;
    addLegacyHost(mounted, legacyOnly);
    await flushPromises();

    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    await tileFor(wrapper, legacyOnly.filename).trigger("click");
    // The bulk control is disabled; ⌘⇧N still opens the dialog around the
    // selection, so the guard must refuse honestly.
    key("N", { ctrlKey: true, shiftKey: true });
    await flushPromises();
    const dialog = wrapper.get("[data-test='rename-dialog']");
    expect(dialog.text()).toContain("New album");
    await dialog.get("input").setValue("Ghost shelf");
    await dialog.get("[data-test='rename-save']").trigger("click");
    await flushPromises();
    expect(org.createCollection).not.toHaveBeenCalled();
    expect(useToastStore().items.at(-1)?.message).toContain("no album was created");
    wrapper.unmount();
  });
});

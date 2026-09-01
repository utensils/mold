import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";

const { apiFetchTo, localGalleryDelete, localGalleryList } = vi.hoisted(() => ({
  apiFetchTo: vi.fn().mockResolvedValue(new Response()),
  localGalleryDelete: vi.fn().mockResolvedValue(undefined),
  localGalleryList: vi.fn(),
}));
const createFramewiseUpscaleMock = vi.hoisted(() => vi.fn());
const getFramewiseUpscaleMock = vi.hoisted(() => vi.fn());
const transitionFramewiseUpscaleMock = vi.hoisted(() => vi.fn());
const upscaleLibraryImageMock = vi.hoisted(() => vi.fn());

vi.mock("@studio/api/videoUpscale", () => ({
  createFramewiseUpscale: createFramewiseUpscaleMock,
  getFramewiseUpscale: getFramewiseUpscaleMock,
  transitionFramewiseUpscale: transitionFramewiseUpscaleMock,
  upscaleLibraryImage: upscaleLibraryImageMock,
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
import { useGenerateFormStore } from "../stores/generateForm";
import { useToastStore } from "../stores/toasts";
import type { GalleryImage } from "../lib/api/types";
import { installMemoryLocalStorage } from "../lib/testSupport/memoryLocalStorage";
import { clearSessionScrollForTests } from "@studio/lib/libraryOrganization";

installMemoryLocalStorage();

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
  route = "/library",
  localServer = false,
) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/library", component: stub },
      { path: "/create", component: stub },
    ],
  });
  await router.push(route);

  const pinia = createPinia();
  setActivePinia(pinia);
  const connection = useConnectionStore();
  connection.info = localServer
    ? { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" }
    : null;
  connection.status = localServer ? "ready" : "error";
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
  seed?.(gallery);
  if (localServer) {
    useHostsStore().capabilities.local = {
      video_upscale: { available: true },
    } as never;
  }
  localGalleryList.mockResolvedValue({
    images: [...(gallery.buckets.local?.items ?? prints)],
    target: localServer ? { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" } : null,
  });

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
  clearSessionScrollForTests();
  vi.clearAllMocks();
  localStorage.clear();
  apiFetchTo.mockResolvedValue(new Response());
  createFramewiseUpscaleMock.mockResolvedValue({
    id: "vup-desktop-1",
    state: "queued",
    completed_frames: 0,
    total_frames: 1,
    disclosure:
      "Framewise upscale processes each frame independently; temporal flicker may remain.",
  });
  getFramewiseUpscaleMock.mockResolvedValue({
    id: "vup-desktop-1",
    state: "failed",
    completed_frames: 0,
    total_frames: 1,
    error: "test stop",
    disclosure: "Framewise upscale",
  });
});

describe("LibraryView session scroll", () => {
  it("restores its own scroller after leaving and returning", async () => {
    const first = await mountView();
    const firstScroller = first.wrapper.get("div[style='contain: strict;']").element as HTMLElement;
    firstScroller.scrollTop = 420;
    first.wrapper.unmount();

    const second = await mountView();
    const secondScroller = second.wrapper.get("div[style='contain: strict;']")
      .element as HTMLElement;
    await vi.waitFor(() => expect(secondScroller.scrollTop).toBe(420));
    second.wrapper.unmount();
  });
});

describe("LibraryView notification deep links", () => {
  it("opens the exact saved print named by the route after the gallery loads", async () => {
    const { wrapper } = await mountView(undefined, undefined, "/library?print=second.png");

    expect(wrapper.getComponent({ name: "Lightbox" }).props("item").filename).toBe("second.png");
  });
});

describe("LibraryView delete keyboard handling", () => {
  it("routes a selected video's context-menu Delete through the full bulk selection", async () => {
    const videos = prints.map((print, index) => ({
      ...print,
      filename: `${index + 1}.mp4`,
    }));
    localGalleryList.mockResolvedValueOnce({ images: videos, target: null });
    const { wrapper, gallery } = await mountView(undefined, (g) => {
      g.buckets.local!.items = videos;
    });

    await wrapper.get('[aria-label="Toggle select mode"]').trigger("click");
    const tiles = wrapper.findAll(".ms-lib-tile");
    await tiles[0]!.trigger("click");
    await tiles[1]!.trigger("click", { metaKey: true });
    expect(wrapper.get('[data-test="bulk-action-bar"]').text()).toContain("2 / 2 selected");

    await tiles[0]!.trigger("contextmenu");
    const menu = useContextMenuStore();
    const deleteEntry = menu.entries.find(
      (entry) => !("separator" in entry) && entry.label === "Delete 2 selected",
    );
    expect(deleteEntry).toBeDefined();
    menu.activate(deleteEntry!);
    await flushPromises();

    const confirm = wrapper
      .get('[data-test="bulk-action-bar"]')
      .findAll("button")
      .find((button) => button.text().includes("Delete 2 prints?"));
    expect(confirm).toBeDefined();
    await confirm!.trigger("click");
    await flushPromises();

    expect(localGalleryDelete).toHaveBeenCalledTimes(2);
    expect(localGalleryDelete).toHaveBeenCalledWith("1.mp4");
    expect(localGalleryDelete).toHaveBeenCalledWith("2.mp4");
    expect(gallery.filtered).toHaveLength(0);
    wrapper.unmount();
  });

  it("loads a remote gallery image as the Create source from its context menu", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "remote-source.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    apiFetchTo.mockResolvedValueOnce(new Response(new Uint8Array([65, 66, 67])));
    const { wrapper, router } = await mountView(remotePrint);

    const tile = wrapper.findAll("button").find((button) => button.text().includes("S 9"));
    expect(tile).toBeDefined();
    await tile!.trigger("contextmenu");

    const menu = useContextMenuStore();
    const sourceEntry = menu.entries.find(
      (entry) => !("separator" in entry) && entry.label === "Use as source",
    )!;
    expect(sourceEntry).toMatchObject({ disabled: false });
    menu.activate(sourceEntry);
    await flushPromises();

    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: "http://plato:7680", apiKey: "secret" },
      "/api/gallery/image/remote-source.png",
    );
    expect(useGenerateFormStore().form.sourceImage).toBe("QUJD");
    expect(useGenerateFormStore().form.sourceImageName).toBe("remote-source.png");
    expect(useGenerateFormStore().form.sourceFit).toEqual({ mode: "crop-fill" });
    expect(router.currentRoute.value.path).toBe("/create");
    expect(useToastStore().items.at(-1)?.message).toBe("Loaded as source");
    wrapper.unmount();
  });

  it("appends a Library source image to Ref2VA's dedicated ordered references", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "ordered-subject.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    apiFetchTo.mockResolvedValueOnce(new Response(new Uint8Array([65, 66, 67])));
    const { wrapper, router } = await mountView(remotePrint);
    const form = useGenerateFormStore().form;
    form.model = "minimax-h3-ref2va:comfy-pruned-int8";
    form.family = "minimax-h3";

    const tile = wrapper.findAll("button").find((button) => button.text().includes("S 9"));
    expect(tile).toBeDefined();
    await tile!.trigger("contextmenu");
    const menu = useContextMenuStore();
    const sourceEntry = menu.entries.find(
      (entry) => !("separator" in entry) && entry.label === "Use as source",
    )!;
    menu.activate(sourceEntry);
    await flushPromises();
    await vi.waitFor(() => {
      expect(form.h3Authoring?.references).toHaveLength(1);
    });

    expect(form.sourceImage).toBeNull();
    expect(form.h3Authoring?.references[0]?.reference).toMatchObject({
      kind: "image",
      media: { authority: "inline", data: "QUJD" },
      provenance: {
        name: "ordered-subject.png",
        sha256: "b5d4045c3f466fa91fe2cc6abe79232a1a57cdf104f7a26e716e0a1e2789df78",
      },
      mime_type: "image/png",
      width: 1024,
      height: 1024,
    });
    expect(router.currentRoute.value.path).toBe("/create");
    wrapper.unmount();
  });

  it.each(["Delete", "Backspace"])(
    "prevents %s navigation and only DELETEs after the undo window",
    async (key) => {
      const { wrapper, gallery, router } = await mountView();
      vi.useFakeTimers();
      try {
        window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));

        const press = new KeyboardEvent("keydown", { key, cancelable: true });
        expect(window.dispatchEvent(press)).toBe(false);
        await flushPromises();

        // Optimistically hidden, undo toast up, no server call and no navigation.
        expect(localGalleryDelete).not.toHaveBeenCalled();
        expect(gallery.filtered.map((entry) => entry.item.filename)).toEqual(["second.png"]);
        expect(useToastStore().items.at(-1)?.message).toContain("Deleted first.png");
        expect(router.currentRoute.value.path).toBe("/library");

        // The DELETE fires only when the 6 s window lapses.
        vi.advanceTimersByTime(6000);
        await flushPromises();
        expect(localGalleryDelete).toHaveBeenCalledWith("first.png");
        expect(router.currentRoute.value.path).toBe("/library");
      } finally {
        vi.useRealTimers();
        wrapper.unmount();
      }
    },
  );

  it("still commits the delete when the undo toast is dismissed mid-window", async () => {
    const { wrapper, gallery } = await mountView();
    vi.useFakeTimers();
    try {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "Delete", cancelable: true }));
      await flushPromises();
      expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["second.png"]);

      // Manually dismiss the undo toast (its ✕ / body click) before the window
      // lapses. The commit rides an independent timer, so this must NOT strand
      // a hidden-but-undeleted print — the delete still fires exactly once.
      const toasts = useToastStore();
      const undo = toasts.items.at(-1)!;
      expect(undo.action?.label).toBe("Undo");
      toasts.dismiss(undo.id);
      await flushPromises();
      expect(localGalleryDelete).not.toHaveBeenCalled();

      vi.advanceTimersByTime(6000);
      await flushPromises();
      expect(localGalleryDelete).toHaveBeenCalledWith("first.png");
      expect(localGalleryDelete).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });

  it("undo restores the print without a server call", async () => {
    const { wrapper, gallery } = await mountView();
    vi.useFakeTimers();
    try {
      window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));
      const press = new KeyboardEvent("keydown", { key: "Delete", cancelable: true });
      window.dispatchEvent(press);
      await flushPromises();
      expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["second.png"]);

      const undo = useToastStore().items.at(-1)!;
      expect(undo.action?.label).toBe("Undo");
      useToastStore().runAction(undo.id);
      await flushPromises();

      // The print is back and no DELETE ran, even after the window would lapse.
      expect(gallery.filtered.map((e) => e.item.filename)).toEqual(["first.png", "second.png"]);
      vi.advanceTimersByTime(6000);
      await flushPromises();
      expect(localGalleryDelete).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });

  it("routes the committed delete to the selected remote host", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "remote.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    const { wrapper, gallery } = await mountView(remotePrint);
    vi.useFakeTimers();
    try {
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

      // Hidden immediately, but nothing hits the host until the window lapses.
      expect(apiFetchTo).not.toHaveBeenCalled();
      expect(gallery.filtered.some((entry) => entry.item.filename === "remote.png")).toBe(false);

      vi.advanceTimersByTime(6000);
      await flushPromises();
      expect(apiFetchTo).toHaveBeenCalledWith(
        { baseUrl: "http://plato:7680", apiKey: "secret" },
        "/api/gallery/image/remote.png",
        { method: "DELETE" },
      );
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });

  it("restores the print and surfaces the error when the committed delete fails", async () => {
    const remotePrint: GalleryImage = {
      ...prints[0]!,
      filename: "remote.png",
      timestamp: 3,
      metadata: { ...prints[0]!.metadata, seed: 9 },
    };
    apiFetchTo.mockRejectedValueOnce(new Error("Forbidden"));
    const { wrapper, gallery } = await mountView(remotePrint);
    vi.useFakeTimers();
    try {
      const tile = wrapper.findAll("button").find((button) => button.text().includes("S 9"));
      await tile!.trigger("contextmenu");
      const menu = useContextMenuStore();
      const deleteEntry = menu.entries.find(
        (entry) => !("separator" in entry) && entry.label === "Delete",
      )!;
      menu.activate(deleteEntry);
      await flushPromises();

      vi.advanceTimersByTime(6000);
      await flushPromises();

      expect(
        useToastStore().items.some((t) => t.message === "Forbidden" && t.kind === "error"),
      ).toBe(true);
      expect(gallery.filtered.some((entry) => entry.item.filename === "remote.png")).toBe(true);
    } finally {
      vi.useRealTimers();
      wrapper.unmount();
    }
  });

  it("leaves Backspace native while the library search field is being edited", async () => {
    const { wrapper } = await mountView();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft", cancelable: true }));
    const search = wrapper.get("input[type='search']").element as HTMLInputElement;
    search.focus();

    const event = new KeyboardEvent("keydown", { key: "Backspace", cancelable: true });
    expect(search.dispatchEvent(event)).toBe(true);
    expect(localGalleryDelete).not.toHaveBeenCalled();
    wrapper.unmount();
  });
});

describe("LibraryView source reuse", () => {
  it("uses the same native-authority loader from the Lightbox as the gallery picker", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([65, 66, 67]), {
        status: 200,
        headers: { "Content-Type": "image/png" },
      }),
    );
    const { wrapper, router } = await mountView();
    const tile = wrapper.findAll("button").find((button) => button.text().includes("S 1"));
    expect(tile).toBeDefined();
    await tile!.trigger("dblclick");
    await wrapper.get("[data-test='lightbox-use-source']").trigger("click");
    await flushPromises();

    expect(fetchMock).toHaveBeenCalledWith("mold-local://localhost/first.png");
    expect(useGenerateFormStore().form.sourceImage).toBe("QUJD");
    expect(useGenerateFormStore().form.sourceFit).toEqual({ mode: "crop-fill" });
    expect(router.currentRoute.value.path).toBe("/create");
    wrapper.unmount();
  });

  it("enables Use as source for video and attaches it to the LTX source-video field", async () => {
    const video = {
      ...prints[0]!,
      filename: "clip.mp4",
      format: "mp4",
      metadata: { ...prints[0]!.metadata, model: "ltx-2.3-22b-dev:fp8" },
    } as GalleryImage;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array([65, 66, 67]), {
        status: 200,
        headers: { "Content-Type": "video/mp4" },
      }),
    );
    const { wrapper, router } = await mountView(undefined, (gallery) => {
      gallery.buckets.local!.items = [video];
    });
    const tile = wrapper.get(".ms-lib-tile");
    await tile.trigger("contextmenu");
    const source = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Use as source",
    );
    expect(source).toMatchObject({ disabled: false });
    useContextMenuStore().activate(source!);
    await flushPromises();

    expect(useGenerateFormStore().form.sourceVideo).toEqual({
      filename: "clip.mp4",
      base64: "QUJD",
    });
    expect(router.currentRoute.value.path).toBe("/create");
    expect(useToastStore().items.at(-1)?.message).toBe("Loaded as source video");
    wrapper.unmount();
  });

  it("queues an existing local Library video from its context menu", async () => {
    const video = {
      ...prints[0]!,
      filename: "existing-clip.mp4",
      format: "mp4",
    } as GalleryImage;
    const { wrapper } = await mountView(
      undefined,
      (gallery) => {
        gallery.buckets.local!.items = [video];
      },
      "/library",
      true,
    );

    await wrapper.get(".ms-lib-tile").trigger("contextmenu");
    const upscale = useContextMenuStore().entries.find(
      (entry) => !("separator" in entry) && entry.label === "Framewise upscale",
    );
    expect(upscale).toMatchObject({ disabled: false });
    useContextMenuStore().activate(upscale!);
    await flushPromises();
    (document.querySelector("[data-test='start-upscale']") as HTMLButtonElement).click();
    await flushPromises();

    expect(createFramewiseUpscaleMock).toHaveBeenCalledWith(
      { baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" },
      "existing-clip.mp4",
      "real-esrgan-x4plus:fp16",
    );
    expect(useToastStore().items.at(-1)?.message).toContain(
      "Framewise upscale queued (vup-desktop-1)",
    );
    wrapper.unmount();
  });
});

describe("LibraryView header + NEW badges", () => {
  it("titles the workspace Library and counts prints across all hosts", async () => {
    const { wrapper } = await mountView();
    const header = wrapper.get("header");
    expect(header.text()).toContain("Library");
    expect(wrapper.get("[data-test='library-count']").text()).toBe("2 prints");
    wrapper.unmount();
  });

  it("restores, applies, and persists the toolbar thumbnail-size slider", async () => {
    localStorage.setItem("mold.gallery.thumbnailSize.v1", "280");
    const { wrapper } = await mountView();
    const slider = wrapper.get<HTMLInputElement>('input[aria-label="Thumbnail size"]');

    expect(slider.element.value).toBe("280");
    expect(wrapper.get(".ms-lib-tile").attributes("style")).toContain("height: 280px");

    // A drag delivers many ticks; the grid follows each one immediately while
    // the localStorage write settles 250 ms after the last tick.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      await slider.setValue("300");
      await slider.setValue("320");
      expect(wrapper.get(".ms-lib-tile").attributes("style")).toContain("height: 320px");
      expect(localStorage.getItem("mold.gallery.thumbnailSize.v1")).toBe("280");
      vi.advanceTimersByTime(250);
      expect(localStorage.getItem("mold.gallery.thumbnailSize.v1")).toBe("320");

      // Leaving mid-settle still persists the last value.
      await slider.setValue("340");
      wrapper.unmount();
      expect(localStorage.getItem("mold.gallery.thumbnailSize.v1")).toBe("340");
    } finally {
      vi.useRealTimers();
    }
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

  it("badges upscaled image tiles without relying only on their filename", async () => {
    const { wrapper } = await mountView(undefined, (gallery) => {
      const first = gallery.buckets.local?.items[0];
      if (first) {
        first.filename = "renamed.png";
        first.metadata = {
          ...first.metadata,
          width: 2048,
          height: 2048,
          generation_width: 512,
          generation_height: 512,
          upscale_model: "real-esrgan-x4plus",
        };
      }
    });

    expect(wrapper.get('[data-test="upscaled-badge"]').text()).toBe("Upscaled");
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
